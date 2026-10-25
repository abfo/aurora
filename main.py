import base64
import json
import logging
import os
import struct
import threading
import time
import wave
from collections import deque
from datetime import datetime, timedelta
from typing import NamedTuple
import numpy as np
import requests
from settings import settings
from logging_config import configure_logging
from wake_word.detector import WakeWordDetector
from wake_word import collect
import pyaudio
import asyncio
from tools.base import load_plugins, Tool
from analytics import Analytics
from audio_manager import AudioManager, ScheduledAudio
from ui.base import AssistantUIBase, AssistantUIState
from live_session import LiveSession, connect_live_session, LIVE_SAMPLERATE
from backend_agent import (
    BackendAgent,
    TranscriptBuffer,
    MODE_COOKING,
    MODE_NORMAL,
    MODE_TRAINING,
)

_AUDIO_PLAYBACK_CHUNK = 1024
# ~85 ms per output callback. Smaller than GPT-Live's 100 ms chunks, so each
# pull takes a modest bite out of the jitter buffer rather than most of it.
_OUTPUT_FRAMESPERBUFFER = 2048
_BYTES_PER_MS = LIVE_SAMPLERATE * 2 // 1000
# GPT-Live streams silence between utterances as exact digital zeros, while
# speech chunks measure far above this RMS, so one threshold separates them.
_VOICED_RMS = 300
# A chunk no louder than this is silence that can be stretched or skipped
# without anyone hearing it. Deliberately stricter than _VOICED_RMS, so the
# quiet tail of a word is never touched.
_SILENT_PEAK = 16
# Let the jitter buffer run this far over target before trimming silence, so
# latency can't creep up if GPT-Live runs slightly faster than the sound card.
_PLAYBACK_TRIM_MS = 200
# Smaller input buffer (~43 ms @ 24 kHz) so wake-word detection stays responsive
# and post-wake audio is buffered at fine granularity for the handoff to GPT-Live.
_INPUT_FRAMESPERBUFFER = 1024
# Seconds of audio kept before the wake word fires and replayed into the session
# so the start of the utterance ("Aurora when is...") is never clipped.
_WAKE_PREROLL_SECONDS = 0.8
_WAKE_PREROLL_FRAMES = int(_WAKE_PREROLL_SECONDS * LIVE_SAMPLERATE / _INPUT_FRAMESPERBUFFER)
# How long the playback queue must stay empty before an action that ends the
# session (sleep, or a switch into cooking or training) is acted on, so Aurora is
# never cut off mid-word. GPT-Live has no end-of-speech event to wait for.
_PLAYBACK_QUIET_SECONDS = 1.2
# Cap on that wait, in case speech keeps trickling in.
_SLEEP_GRACE_SECONDS = 8.0
# How long to give Aurora to start speaking a result before accepting that this
# one is silent. The backend usually decides to sleep in the same breath as it
# reports the result, so at that point she has not started talking yet.
_SPEECH_START_GRACE_SECONDS = 3.0
# Aurora counts as speaking for this long after her last voiced audio, so the
# UI does not flicker between words and the optional mic mute covers the echo
# still coming out of the speaker.
_TALK_HOLD_SECONDS = 0.5

_COOKING_INSTRUCTIONS = """

# Cooking Mode

IMPORTANT: you are currently helping the user cook. The recipe is included below. Follow the recipe instructions
step by step and make sure the user has completed each step before moving to the next one. Do not skip steps unless
asked to by the user. Answer any questions in the context of the recipe. If the recipe calls for a timer, set one for 
the user. If the user wants to stop cooking, use the stop_cooking tool.

The recipe:

"""

def main():
    configure_logging(settings.log_level, settings.log_file)
    log = logging.getLogger("aurora")
    log.info("Log level=%s, log_file=%s", logging.getLevelName(log.getEffectiveLevel()), settings.log_file)

    # Initialize UI (lazy imports to avoid hardware deps on Windows)
    if settings.ui == "Debug":
        from ui.debug import DebugUI
        ui = DebugUI(log)
    elif settings.ui == "Braincraft":
        try:
            from ui.braincraft import BraincraftUI
            ui = BraincraftUI(log)
        except Exception:
            log.exception("Braincraft UI not available on this platform")
            return
    else:
        log.warning(f"Unknown UI implementation: {settings.ui}")
        return
    log.info("UI=%s", settings.ui)
    
    # We need internet
    _wait_for_internet_connection()
    ui.update_state(AssistantUIState.LOAD_INTERNET, reason="Internet initialized")

    # create audio manager and load tool plugins with it
    audio_manager = AudioManager(log)
    analytics = Analytics()
    tools = load_plugins(log=log, audio_manager=audio_manager, analytics=analytics)

    # Two prompts, because GPT-Live splits the job in two: the live prompt is the
    # voice (persona, speaking style, when to delegate), the backend prompt is the
    # agent that actually runs the tools.
    agent_instructions = "You are a helpful assistant."
    if settings.agent_instructions_path:
        try:
            with open(settings.agent_instructions_path, "r") as f:
                agent_instructions = f.read()
                log.info(f"Loaded agent instructions from {settings.agent_instructions_path}")
        except FileNotFoundError:
            log.warning(f"Agent instructions file not found: {settings.agent_instructions_path}")

    backend_instructions = (
        "You are the backend task handler for a voice assistant. Call a tool whenever one fits "
        "the request, and reply with one short sentence for the assistant to say out loud."
    )
    if settings.backend_instructions_path:
        try:
            with open(settings.backend_instructions_path, "r") as f:
                backend_instructions = f.read()
                log.info(f"Loaded backend instructions from {settings.backend_instructions_path}")
        except FileNotFoundError:
            log.warning(f"Backend instructions file not found: {settings.backend_instructions_path}")

    ui.update_state(AssistantUIState.LOAD_DISPLAY, reason="Display initialized")

    try:
        # open audio and log available devices
        audio = pyaudio.PyAudio()
        _log_audio_devices(audio)
        ui.update_state(AssistantUIState.LOAD_AUDIO, reason="Audio initialized")

        # Run the assistant: one persistent mic stream feeds wake-word detection
        # while sleeping and the GPT-Live session while talking (see _run_assistant).
        asyncio.run(_run_assistant(
            audio,
            agent_instructions,
            backend_instructions,
            log,
            tools,
            audio_manager,
            ui,
            analytics
        ))

    except Exception as e:
        log.exception(f"Error initializing audio: {e}")

    finally:
        if audio:
            audio.terminate()
        if ui:
            ui.shutdown()

def _drain_queue(q: asyncio.Queue) -> None:
    """Discard any buffered frames (e.g. stale audio before we start sleeping)."""
    try:
        while True:
            q.get_nowait()
    except asyncio.QueueEmpty:
        pass

async def _run_assistant(
        audio: pyaudio.PyAudio,
        agent_instructions: str,
        backend_instructions: str,
        log: logging.Logger,
        tools: list[Tool],
        audio_manager: AudioManager,
        ui: AssistantUIBase,
        analytics: Analytics):
    """Own a single always-open mic stream and alternate between listening for
    the wake word and running a GPT-Live conversation.

    The same stream and frame queue are used throughout, so audio spoken right
    after the wake word (while the websocket is still connecting) is buffered in
    the queue and flushed once the session is ready - no gap, and no
    closing/reopening of the input stream. The same buffering covers the
    reconnect when a session switches mode.
    """
    loop = asyncio.get_running_loop()
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue()
    # mic_gate controls whether the callback enqueues frames. It stays on for the
    # whole conversation - GPT-Live handles interruptions itself - and is only
    # closed while an alarm is playing or training clips are being recorded.
    mic_gate = {"capture": True}

    input_stream = await _open_input_stream_async(audio, loop, frame_queue, lambda: mic_gate["capture"])
    input_stream.start_stream()

    try:
        while True:
            # Play any due scheduled audio (timers/alarms) first.
            scheduled_audio = audio_manager.get_audio()
            if scheduled_audio:
                ui.set_timer_text(audio_manager.audio_to_text())
                ui.update_state(AssistantUIState.TALKING, reason="Playing scheduled audio")
                mic_gate["capture"] = False
                await asyncio.to_thread(_play_scheduled_audio, audio, scheduled_audio, ui)

            # Listen for the wake word on the live mic.
            ui.update_state(AssistantUIState.SLEEPING, reason="Listening for wake word")
            mic_gate["capture"] = True
            _drain_queue(frame_queue)

            detector = WakeWordDetector(
                settings.wake_word_model_path,
                threshold=settings.wake_word_threshold,
            )

            outcome = await _wait_for_wake_word(detector, frame_queue, audio_manager, ui, log)

            detector.delete()

            if outcome == "shutdown":
                log.info("User requested shutdown")
                break
            if outcome not in ("woke", "train"):
                # due_audio or transient error: loop back to play audio / retry.
                continue

            # "train" skips the wake word entirely and opens the session straight
            # into training mode; everything downstream is otherwise identical.
            training_mode = outcome == "train"
            reason = "Training requested" if training_mode else "Wake word detected"
            log.info(reason)
            _drain_queue(frame_queue)
            # Show LISTENING immediately on wake so the user has feedback while
            # the GPT-Live session connects in the background.
            ui.update_state(AssistantUIState.LISTENING, reason=reason)
            # Hand the live queue to the conversation. Frames captured from here
            # on (including during connect) are buffered and sent once connected.
            #
            # A conversation runs until GPT-Live closes, and may hand back an
            # action. Cooking and training need a different live prompt, and that
            # prompt is fixed at session.start, so those actions are served by
            # opening the next session here rather than reconfiguring this one.
            plan = _training_plan() if training_mode else _normal_plan(agent_instructions)
            while plan is not None:
                action = await run_conversation(
                    audio,
                    frame_queue,
                    mic_gate,
                    log,
                    tools,
                    audio_manager,
                    ui,
                    analytics,
                    backend_instructions,
                    plan
                )

                next_plan = None
                kind = (action or {}).get("action")

                if kind == "train":
                    next_plan = _training_plan()
                elif kind == "cook":
                    next_plan = _cooking_plan(agent_instructions, action.get("recipe", ""))
                elif kind == "stop_cooking":
                    next_plan = _normal_plan(agent_instructions)
                elif kind == "capture":
                    # Recording needs the mic stream and the UI owned here, so it
                    # runs between sessions rather than inside one.
                    ui.update_state(AssistantUIState.TOOL_CALLING, reason="Wake word training")
                    try:
                        await collect.collect_training_samples(
                            frame_queue,
                            mic_gate,
                            ui,
                            action.get("name", ""),
                            log,
                            collect_dir=settings.wake_word_collect_dir,
                        )
                    except Exception as e:
                        log.exception("Wake word training capture failed: %s", e)
                    # Fall through; the loop returns to SLEEPING and re-arms the
                    # wake word with the normal prompt again.

                plan = next_plan
    finally:
        if input_stream:
            input_stream.stop_stream()
            input_stream.close()

async def _wait_for_wake_word(
        detector: WakeWordDetector,
        frame_queue: asyncio.Queue,
        audio_manager: AudioManager,
        ui: AssistantUIBase,
        log: logging.Logger) -> str:
    """Consume mic frames and feed the detector until the wake word fires.

    Returns one of: "woke", "train", "shutdown", "due_audio", "error".

    "train" means the user asked for wake-word training from the UI (the
    joystick button) rather than by speaking to Aurora.
    """
    next_timer_update = datetime.now() + timedelta(seconds=1)
    preroll: deque[bytes] = deque(maxlen=_WAKE_PREROLL_FRAMES)
    # The training button is the same one that cancels a session or an alarm, so
    # it may well still be held when we arrive here. Start disarmed: a press only
    # counts once the button has been seen released during this idle period.
    train_armed = False
    while True:
        try:
            frame = await frame_queue.get()
            preroll.append(frame)
            samples = struct.unpack_from("h" * (len(frame) // 2), frame)
            if detector.process(samples) >= 0:
                # When training is enabled, save the exact 1.2s window that fired
                # as a negative-candidate clip (reviewed/sorted later). Guarded so
                # a disk error never blocks waking. The buffer is already in
                # memory, so this adds negligible latency before the handoff.
                if settings.wake_word_training_enabled:
                    try:
                        pcm = detector.window_pcm_bytes()
                        await asyncio.to_thread(
                            collect.save_activation_clip,
                            pcm,
                            settings.wake_word_collect_dir,
                            log,
                        )
                    except Exception as e:
                        log.exception("Failed to save activation clip: %s", e)
                # Replay the buffered lead-in (incl. the wake word) so the start
                # of the utterance reaches the live session uncut. Any frames
                # already queued are newer than the pre-roll, so re-insert the
                # pre-roll first, then those, to preserve chronological order.
                newer = []
                try:
                    while True:
                        newer.append(frame_queue.get_nowait())
                except asyncio.QueueEmpty:
                    pass
                for f in preroll:
                    frame_queue.put_nowait(f)
                for f in newer:
                    frame_queue.put_nowait(f)
                return "woke"
            if ui.is_shutdown_pressed():
                return "shutdown"
            # The joystick button starts wake-word training directly, skipping
            # the wake word and the spoken request. Gated on the same master
            # switch as the start_wake_word_training tool so a stray press does
            # nothing on a normal build.
            train_pressed = settings.wake_word_training_enabled and ui.is_train_pressed()
            if not train_pressed:
                # Arm on release. These are undebounced level reads, so without
                # this a button still held from cancelling an alarm - or from the
                # press that ended the last capture - would fire straight away.
                train_armed = True
            elif train_armed:
                train_armed = False
                return "train"
            if audio_manager.has_due_audio():
                return "due_audio"
            if datetime.now() > next_timer_update:
                next_timer_update = datetime.now() + timedelta(seconds=1)
                if audio_manager.has_any_audio():
                    ui.set_timer_text(audio_manager.audio_to_text())
        except Exception as e:
            log.exception(f"Error in wake word loop: {e}")
            await asyncio.sleep(5)
            return "error"

def _normal_plan(agent_instructions: str) -> dict:
    """Aurora as usual: the full live prompt and every configured tool."""
    return {"mode": MODE_NORMAL, "instructions": agent_instructions, "opening": None}


def _cooking_plan(agent_instructions: str, recipe: str) -> dict:
    """Cooking mode, with the recipe baked into the live prompt.

    The recipe goes to the voice model rather than the backend because stepping
    the user through it is conversation, not a task - the backend keeps the
    tools it needs for timers along the way.
    """
    return {
        "mode": MODE_COOKING,
        "instructions": f"{agent_instructions}{_COOKING_INSTRUCTIONS}{recipe}",
        "opening": (
            "Cooking mode has started. Say one short line, then take the user "
            "through the first step of the recipe."
        ),
    }


def _training_plan() -> dict:
    """Wake-word training, on the dedicated short prompt and a tiny tool set."""
    return {
        "mode": MODE_TRAINING,
        "instructions": collect.TRAINING_PROMPT,
        "opening": (
            "Wake word training has started. Greet the user in one short line "
            "and ask for their first name."
        ),
    }


class _Chunk(NamedTuple):
    """One output delta, classified once on arrival."""
    pcm: bytes
    voiced: bool   # contains speech
    silent: bool   # digital silence, safe to stretch or skip


class _Playback:
    """Assistant audio, buffered here and pulled by PortAudio on its own thread.

    GPT-Live sends output as one continuous stream for the whole session: a
    100 ms chunk roughly every 100 ms, digital silence between utterances, and
    never ahead of real time. Two things that used to come for free therefore
    have to be done here.

    A jitter buffer. The Realtime API sent audio faster than real time, so the
    old blocking writes built up a cushion in the device buffer. GPT-Live never
    gets ahead, so without a deliberate cushion any network hiccup or event-loop
    stall empties the device buffer and the speaker clicks. Playback holds back
    until PLAYBACK_BUFFER_MS is queued, and the stream runs in callback mode, so
    PortAudio pulls audio when it needs it rather than waiting for the event
    loop to push it.

    Speech detection. A chunk arriving says nothing about whether Aurora is
    talking, because silence arrives too. Each chunk is classified by energy as
    it is queued, and :attr:`speaking` means voiced audio is still queued or
    played within the last _TALK_HOLD_SECONDS. The idle timer, the UI state,
    the session-ending wait and the optional mic mute all read it from here.
    """

    def __init__(self, log: logging.Logger):
        self._log = log
        # The event loop enqueues and PortAudio's thread pulls, so everything
        # below the lock is shared between them.
        self._lock = threading.Lock()
        self._chunks: deque[_Chunk] = deque()
        self._head = 0           # bytes of _chunks[0] already played
        self._level = 0          # bytes queued and not yet played
        self._pending = b""      # odd trailing byte, carried into the next delta
        self._target = settings.playback_buffer_ms * _BYTES_PER_MS
        self._low_water = self._target // 2
        self._ceiling = self._target + _PLAYBACK_TRIM_MS * _BYTES_PER_MS
        self._buffering = True   # holding back until the cushion is full
        self._refilling = False  # rebuilding a depleted cushion during silence
        self._queued_voiced = 0
        self._last_voiced = 0.0
        self._created = time.monotonic()
        # Logged when the session ends, to tune PLAYBACK_BUFFER_MS on hardware.
        self._underruns = 0
        self._device_underflows = 0
        self._inserted_ms = 0.0
        self._dropped_ms = 0.0

    def enqueue(self, pcm: bytes) -> bool:
        """Queue one output delta. Returns True if it contains speech."""
        # The callback hands PortAudio whole 16-bit samples; one stray byte
        # would shift every sample after it into noise.
        pcm = self._pending + pcm
        whole = len(pcm) - len(pcm) % 2
        pcm, self._pending = pcm[:whole], pcm[whole:]
        if not pcm:
            return False

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        chunk = _Chunk(pcm, rms > _VOICED_RMS, int(np.abs(samples).max()) <= _SILENT_PEAK)

        with self._lock:
            self._chunks.append(chunk)
            self._level += len(pcm)
            if chunk.voiced:
                self._queued_voiced += 1
        return chunk.voiced

    def clear(self) -> None:
        """Drop anything not yet played (cancel button, session ending)."""
        with self._lock:
            self._chunks.clear()
            self._head = 0
            self._level = 0
            self._pending = b""
            self._queued_voiced = 0
            self._buffering = True
            self._refilling = False

    @property
    def speaking(self) -> bool:
        return (self._queued_voiced > 0
                or time.monotonic() - self._last_voiced < _TALK_HOLD_SECONDS)

    def quiet_for(self) -> float:
        """Seconds since Aurora last spoke; 0 while she is speaking."""
        if self.speaking:
            return 0.0
        return time.monotonic() - (self._last_voiced or self._created)

    def summary(self) -> str:
        return (f"{self._underruns} underruns, {self._device_underflows} device underflows, "
                f"{self._inserted_ms:.0f} ms silence inserted, {self._dropped_ms:.0f} ms silence skipped")

    def callback(self, in_data, frame_count, time_info, status_flags):
        """PortAudio output callback. Runs on PortAudio's thread and never waits.

        A device underflow means this callback itself ran late (CPU or GIL
        contention); an underrun means the jitter buffer ran dry (network).
        They are counted separately because they have different fixes.
        """
        if status_flags & pyaudio.paOutputUnderflow:
            self._device_underflows += 1
        need = frame_count * 2
        with self._lock:
            out = self._pull(need)
        # Always return the full length: PyAudio treats a short buffer as the
        # end of the stream.
        return (out, pyaudio.paContinue)

    def _pull(self, need: int) -> bytes:
        if self._buffering:
            if self._level < self._target:
                return bytes(need)
            self._buffering = False

        out = bytearray()
        while len(out) < need:
            if not self._chunks:
                # Ran dry mid-stream. Pad with silence and rebuild the whole
                # cushion before playing again, rather than stuttering along
                # chunk by chunk.
                self._underruns += 1
                self._buffering = True
                out.extend(bytes(need - len(out)))
                break

            chunk = self._chunks[0]
            if self._head == 0 and chunk.silent:
                # Between utterances nobody can hear the cushion being adjusted,
                # so this is where clock drift between GPT-Live and the sound
                # card is corrected: stretch the silence when running low (with
                # hysteresis, so it refills fully), skip it when running long.
                if self._level < self._low_water:
                    self._refilling = True
                if self._refilling:
                    if self._level < self._target:
                        gap = need - len(out)
                        out.extend(bytes(gap))
                        self._inserted_ms += gap / _BYTES_PER_MS
                        break
                    self._refilling = False
                elif self._level > self._ceiling:
                    self._chunks.popleft()
                    self._level -= len(chunk.pcm)
                    self._dropped_ms += len(chunk.pcm) / _BYTES_PER_MS
                    continue

            take = min(need - len(out), len(chunk.pcm) - self._head)
            out.extend(chunk.pcm[self._head:self._head + take])
            self._head += take
            self._level -= take
            if chunk.voiced:
                self._last_voiced = time.monotonic()
            if self._head >= len(chunk.pcm):
                self._chunks.popleft()
                self._head = 0
                if chunk.voiced:
                    self._queued_voiced -= 1

        return bytes(out)


async def run_conversation(
        audio: pyaudio.PyAudio,
        frame_queue: asyncio.Queue,
        mic_gate: dict,
        log: logging.Logger,
        tools: list[Tool],
        audio_manager: AudioManager,
        ui: AssistantUIBase,
        analytics: Analytics,
        backend_instructions: str,
        plan: dict):
    """Run one GPT-Live session to completion and return what to do next.

    ``plan`` carries the live prompt the session opens with and the mode the
    backend agent runs in (see the _*_plan helpers). The session ends when the
    backend asks it to, when a timer comes due, when the user presses cancel, or
    when nobody has said anything for a while; the action it ends on is returned
    to the caller, which may open the next session.
    """
    session = None
    output_stream = None
    tasks: list[asyncio.Task] = []
    # Filled in by the delegation handler with the action this session ends on.
    outcome: dict = {}
    # Delegations in flight. The idle timer must not close the session while the
    # backend is still working, however quiet it is in here.
    inflight: set[asyncio.Task] = set()
    # Last time anyone said anything, either direction.
    activity = {"last": time.monotonic()}

    transcript = TranscriptBuffer()
    agent = BackendAgent(
        instructions=backend_instructions,
        tools=tools,
        log=log,
        analytics=analytics,
        mode=plan["mode"],
    )

    # Nothing gates the mic during a session any more: GPT-Live is full duplex
    # and decides for itself when to stop talking and listen. mic_gate stays on
    # and belongs to the outer loop (alarm playback, training capture).
    mic_gate["capture"] = True

    # The output stream pulls from this, so it has to exist before the stream.
    playback = _Playback(log)

    log.info("Connecting to GPT-Live and opening output stream...")
    # Either half can fail - no network, or no audio device - and whichever one
    # succeeded has to be let go of, or a flaky connection slowly leaks PyAudio
    # streams across wake cycles.
    session, output_stream = await asyncio.gather(
        connect_live_session(plan["instructions"], log),
        _open_output_stream_async(audio, playback.callback),
        return_exceptions=True,
    )
    if isinstance(session, BaseException) or isinstance(output_stream, BaseException):
        failure = session if isinstance(session, BaseException) else output_stream
        log.error("Could not start the conversation: %s", failure)
        if not isinstance(output_stream, BaseException):
            try:
                output_stream.close()
            except Exception:
                pass
        if not isinstance(session, BaseException):
            await session.disconnect()
        return None

    try:
        log.info("Starting audio...")
        output_stream.start_stream()

        if plan.get("opening"):
            # Nothing has been said in this session yet - it was opened by a
            # button press or a mode change, not by someone speaking - so the
            # live model needs telling that it is its turn.
            await session.append_instructions(plan["opening"])

        tasks = [
            asyncio.create_task(_send_audio_loop(session, frame_queue, playback)),
            asyncio.create_task(_due_audio_loop(session, playback, audio_manager, ui)),
            asyncio.create_task(_ui_state_loop(ui, playback, inflight)),
            asyncio.create_task(_close_when_quiet_loop(session, playback, outcome, log)),
            asyncio.create_task(_idle_loop(session, playback, activity, inflight, outcome, plan, log)),
        ]

        log.info("Ready for conversation.")
        analytics.report_event("Conversation")
        if plan["mode"] == MODE_TRAINING:
            analytics.report_event("WakeWordTraining")

        async for event in session.events():
            kind = event.get("type")

            # error, log it
            if kind == "error":
                log.warning("GPT-Live error: %s", event.get("error"))

            # output audio, queue it for playback. This arrives all session
            # long, silence included, so only speech counts as activity.
            elif kind == "session.output_audio.delta":
                delta = event.get("delta")
                if delta and playback.enqueue(base64.b64decode(delta)):
                    activity["last"] = time.monotonic()

            # captions, both directions. These are the only record of what was
            # said, and the delegation handler builds its request from them.
            elif kind == "session.output_transcript.delta":
                transcript.add_assistant(event.get("delta") or "")
                activity["last"] = time.monotonic()

            elif kind == "session.input_transcript.delta":
                transcript.add_user(event.get("delta") or "")
                activity["last"] = time.monotonic()

            # GPT-Live wants something done: run it on the backend agent. This
            # goes on its own task so audio keeps flowing while it works.
            elif kind == "session.delegation.created":
                delegation = event.get("delegation") or {}
                if delegation.get("target") != "client":
                    log.debug("Ignoring delegation targeted at %s", delegation.get("target"))
                    continue
                delegation_id = delegation.get("id") or ""
                log.info("Delegation created: %s", delegation_id)
                activity["last"] = time.monotonic()
                task = asyncio.create_task(_handle_delegation(
                    agent, session, delegation_id, transcript, outcome, audio_manager, ui, log
                ))
                inflight.add(task)
                task.add_done_callback(inflight.discard)

            elif kind == "session.closed":
                log.info("Session closed: reason=%s usage=%ss", event.get("reason"),
                         (event.get("usage") or {}).get("seconds"))
                break

    except Exception as e:
        log.info(f"Error or user shutdown in conversation: {e}")
    finally:
        log.info("Conversation over, cleaning up.")
        for task in list(inflight) + tasks:
            task.cancel()
        if tasks or inflight:
            await asyncio.gather(*(list(inflight) + tasks), return_exceptions=True)
        # Release the speaker first: closing the session can take a while to
        # drain, and there is nothing left worth playing.
        playback.clear()
        try:
            output_stream.stop_stream()
            output_stream.close()
        except Exception:
            log.debug("Output stream already closed", exc_info=True)
        log.info("Playback: %s", playback.summary())
        # Close gracefully so the final usage is reported, then let go of the
        # socket. The receive loop above has stopped, so draining the remaining
        # events here is safe.
        await session.request_close()
        await session.drain_until_closed()
        await session.disconnect()
        # The input stream stays open for the next wake-word cycle, and mic_gate
        # is left on for the outer loop to manage.

    return outcome.get("action")


async def _handle_delegation(
        agent: BackendAgent,
        session: LiveSession,
        delegation_id: str,
        transcript: TranscriptBuffer,
        outcome: dict,
        audio_manager: AudioManager,
        ui: AssistantUIBase,
        log: logging.Logger):
    """Run one delegated task and record any action it produced.

    The delegation event carries no request text, so the task is reconstructed
    from the transcript captured so far.
    """
    try:
        action = await agent.handle(session, delegation_id, transcript.recent())
    except Exception as e:
        log.exception("Delegation %s failed: %s", delegation_id, e)
        return

    # A tool may have added or removed a timer, so resync the display. Mirrors
    # the resync done in the main and wake-word loops.
    ui.set_timer_text(audio_manager.audio_to_text())

    # First action wins: a second opinion about which mode to be in would only
    # fight with the first.
    if action and "action" not in outcome:
        log.info("Session will end on action: %s", action.get("action"))
        outcome["action"] = action


async def _send_audio_loop(session: LiveSession, frame_queue: asyncio.Queue, playback: _Playback):
    """Stream mic frames to GPT-Live.

    Frames go up continuously. GPT-Live listens and speaks at the same time and
    handles interruptions itself, so there is no response-driven gate here any
    more. MUTE_INPUT_WHILE_SPEAKING is the one exception, for hardware where the
    speaker bleeds into the mic badly enough that Aurora hears herself - it
    costs the ability to interrupt her, which is why it is off by default.

    Muting swaps the mic for digital silence rather than dropping frames or
    using the server's mute: GPT-Live expects an unbroken stream paced like a
    live mic, and this leaves no server-side state that could get stuck.
    """
    while True:
        try:
            frame = await frame_queue.get()
            if not frame:
                continue
            if settings.mute_input_while_speaking and playback.speaking:
                frame = bytes(len(frame))
            await session.append_audio(frame)
        except Exception:
            pass


async def _due_audio_loop(session: LiveSession, playback: _Playback,
                          audio_manager: AudioManager, ui: AssistantUIBase):
    """End the session when a timer comes due or the user presses cancel.

    Both want the speaker now, so this closes the session outright instead of
    asking Aurora to wrap up, and drops whatever she had left to say.
    """
    while True:
        await asyncio.sleep(1)
        if audio_manager.has_due_audio() or ui.is_cancel_pressed():
            playback.clear()
            await session.request_close()
            return
        if audio_manager.has_any_audio():
            ui.set_timer_text(audio_manager.audio_to_text())


async def _close_when_quiet_loop(session: LiveSession, playback: _Playback,
                                 outcome: dict, log: logging.Logger):
    """Close the session once an action is pending and Aurora has finished.

    There is no end-of-speech event, so this waits on voiced audio in the
    playback buffer. Two waits, not one: the backend decides to sleep (or to
    change mode) in the same breath as it reports the result, which GPT-Live is
    still turning into speech at that point, so closing on the first quiet
    moment would cut Aurora off before she had said anything at all.
    """
    while not outcome:
        await asyncio.sleep(0.2)

    waited = 0.0
    while not playback.speaking and waited < _SPEECH_START_GRACE_SECONDS:
        await asyncio.sleep(0.2)
        waited += 0.2

    waited = 0.0
    while playback.quiet_for() < _PLAYBACK_QUIET_SECONDS and waited < _SLEEP_GRACE_SECONDS:
        await asyncio.sleep(0.2)
        waited += 0.2

    log.info("Closing the session on action: %s", (outcome.get("action") or {}).get("action"))
    await session.request_close()


async def _idle_loop(session: LiveSession, playback: _Playback, activity: dict,
                     inflight: set, outcome: dict, plan: dict, log: logging.Logger):
    """Close the session after a long silence.

    This replaces the old watchdog, which hung off response.done - an event
    GPT-Live does not have. A session nobody speaks to again has to be noticed
    here or it would stay open indefinitely. Cooking mode is exempt, where long
    silences are exactly what is expected.
    """
    if plan["mode"] == MODE_COOKING or settings.idle_timeout_seconds <= 0:
        return

    while True:
        await asyncio.sleep(1)
        if inflight or outcome or playback.speaking:
            continue
        idle = time.monotonic() - activity["last"]
        if idle >= settings.idle_timeout_seconds:
            log.info("Nothing said for %.0fs, ending the session", idle)
            await session.request_close()
            return


async def _ui_state_loop(ui: AssistantUIBase, playback: _Playback, inflight: set):
    """Keep the UI showing what Aurora is actually doing.

    Replaces the old response.created / response.done transitions, which have no
    GPT-Live equivalent: talking is voiced audio queued or playing, tool calling
    is a delegation in flight, and anything else is listening.
    """
    shown = None
    while True:
        await asyncio.sleep(0.2)
        if inflight:
            state, reason = AssistantUIState.TOOL_CALLING, "Backend working"
        elif playback.speaking:
            state, reason = AssistantUIState.TALKING, "Assistant speaking"
        else:
            state, reason = AssistantUIState.LISTENING, "Listening"
        if state is not shown:
            shown = state
            ui.update_state(state, reason=reason)


def _make_audio_callback(loop, frame_queue: asyncio.Queue, should_capture):
    def _callback(in_data, frame_count, time_info, status_flags):
        # Push from PortAudio thread into the asyncio loop without blocking
        try:
            if should_capture():
                loop.call_soon_threadsafe(frame_queue.put_nowait, in_data)
        except Exception:
            pass
        return (in_data, pyaudio.paContinue)
    return _callback

async def _open_input_stream_async(audio: pyaudio.PyAudio, loop, frame_queue: asyncio.Queue, should_capture):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=LIVE_SAMPLERATE,
            input=True,
            frames_per_buffer=_INPUT_FRAMESPERBUFFER,
            input_device_index=settings.input_device_id,
            stream_callback=_make_audio_callback(loop, frame_queue, should_capture),
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

async def _open_output_stream_async(audio: pyaudio.PyAudio, stream_callback):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=LIVE_SAMPLERATE,
            output=True,
            frames_per_buffer=_OUTPUT_FRAMESPERBUFFER,
            output_device_index=settings.output_device_id,
            stream_callback=stream_callback,
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

def _play_scheduled_audio(audio: pyaudio.PyAudio, scheduled_audio: ScheduledAudio, ui: AssistantUIBase):
    log = logging.getLogger("aurora")
    try:
        wf = wave.open(scheduled_audio.path, 'rb')

        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output_device_index=settings.output_device_id,
            output=True)
        
        data = wf.readframes(_AUDIO_PLAYBACK_CHUNK)
        while data != b'':
            stream.write(data)
            data = wf.readframes(_AUDIO_PLAYBACK_CHUNK)

            # audio playback is interrupted
            if ui.is_cancel_pressed():
                break
        
        stream.stop_stream()
        stream.close()
        wf.close()

        if (scheduled_audio.delete_after_play):
            try:
                os.remove(scheduled_audio.path)
            except FileNotFoundError:
                log.debug("Audio file already removed: %s", scheduled_audio.path)
            except Exception:
                log.exception("Failed to remove audio file: %s", scheduled_audio.path)

    except Exception as e:
        log.exception("Failed to play scheduled audio: %s", e)

def _wait_for_internet_connection():
    while True:
        try:
            requests.get('https://api.openai.com/').status_code
            break
        except:
            time.sleep(5)
            pass

# log available audio devices
def _log_audio_devices(audio: pyaudio.PyAudio):
    """Log available PyAudio devices with safe formatting and error guards."""
    log = logging.getLogger("aurora")
    try:
        device_count = audio.get_device_count()
        log.debug("Detected %s audio devices", device_count)
    except Exception:
        log.exception("Failed to get audio device count")
        return

    for i in range(device_count):
        try:
            info = audio.get_device_info_by_index(i)
            name = info.get('name', f'Device {i}')
            max_input = int(info.get('maxInputChannels') or 0)
            max_output = int(info.get('maxOutputChannels') or 0)

            if max_input > 0:
                log.info("Input Device id %s - %s (channels=%s)", i, name, max_input)
            if max_output > 0:
                log.info("Output Device id %s - %s (channels=%s)", i, name, max_output)
            if max_input == 0 and max_output == 0:
                log.debug("Device id %s - %s has no I/O channels reported", i, name)
        except Exception:
            log.exception("Failed to get info for audio device index %s", i)

if __name__ == "__main__":
    main()