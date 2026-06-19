import base64
import json
import logging
import os
import struct
import time
import wave
from collections import deque
from datetime import datetime, timedelta
import requests
from settings import settings
from logging_config import configure_logging
from wake_word.detector import WakeWordDetector
from wake_word import collect
import pyaudio
import asyncio
import websockets
from tools.base import load_plugins, Tool
from analytics import Analytics
from audio_manager import AudioManager, ScheduledAudio
from ui.base import AssistantUIBase, AssistantUIState

_AUDIO_PLAYBACK_CHUNK = 1024
_REALTIME_SAMPLERATE = 24000
_REALTIME_FRAMESPERBUFFER = 4096
# Smaller input buffer (~43 ms @ 24 kHz) so wake-word detection stays responsive
# and post-wake audio is buffered at fine granularity for the realtime handoff.
_INPUT_FRAMESPERBUFFER = 1024
# Seconds of audio kept before the wake word fires and replayed into the session
# so the start of the utterance ("Aurora when is...") is never clipped.
_WAKE_PREROLL_SECONDS = 0.8
_WAKE_PREROLL_FRAMES = int(_WAKE_PREROLL_SECONDS * _REALTIME_SAMPLERATE / _INPUT_FRAMESPERBUFFER)
_WS_PING_INTERVAL_SECONDS = 30
_WS_PING_TIMEOUT_SECONDS = 25
_REALTIME_WATCHDOG_TIMEOUT_SECONDS = 120

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

    agent_instructions = "You are a helpful assistant."
    if settings.agent_instructions_path:
        try:
            with open(settings.agent_instructions_path, "r") as f:
                agent_instructions = f.read()
                log.info(f"Loaded agent instructions from {settings.agent_instructions_path}")
        except FileNotFoundError:
            log.warning(f"Agent instructions file not found: {settings.agent_instructions_path}")

    ui.update_state(AssistantUIState.LOAD_DISPLAY, reason="Display initialized")

    try:
        # open audio and log available devices
        audio = pyaudio.PyAudio()
        _log_audio_devices(audio)
        ui.update_state(AssistantUIState.LOAD_AUDIO, reason="Audio initialized")

        # Run the assistant: one persistent mic stream feeds wake-word detection
        # while sleeping and the realtime session while talking (see _run_assistant).
        asyncio.run(_run_assistant(
            audio,
            agent_instructions,
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
        log: logging.Logger,
        tools: list[Tool],
        audio_manager: AudioManager,
        ui: AssistantUIBase,
        analytics: Analytics):
    """Own a single always-open mic stream and alternate between listening for
    the wake word and running a realtime conversation.

    The same stream and frame queue are used throughout, so audio spoken right
    after the wake word (while the realtime websocket is still connecting) is
    buffered in the queue and flushed once the session is ready - no gap, and
    no closing/reopening of the input stream.
    """
    loop = asyncio.get_running_loop()
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue()
    # mic_gate controls whether the callback enqueues frames. Off while the
    # assistant is talking (avoids it hearing itself) and during alarm playback.
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
            if outcome != "woke":
                # due_audio or transient error: loop back to play audio / retry.
                continue

            log.info("Wake word detected")
            # Show LISTENING immediately on wake so the user has feedback while
            # the realtime session connects in the background.
            ui.update_state(AssistantUIState.LISTENING, reason="Wake word detected")
            # Hand the live queue to the conversation. Frames captured from here
            # on (including during connect) are buffered and sent once connected.
            action = await run_realtime_conversation(
                audio,
                frame_queue,
                mic_gate,
                agent_instructions,
                log,
                tools,
                audio_manager,
                ui,
                analytics
            )

            # The conversation may ask us to run wake-word training capture once
            # the session has ended (it needs the mic stream / UI we own here).
            if action and action.get("action") == "train_wake_word":
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
                # Fall through; the loop returns to SLEEPING and re-arms the wake
                # word, and the next conversation uses the normal prompt again.
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

    Returns one of: "woke", "shutdown", "due_audio", "error".
    """
    next_timer_update = datetime.now() + timedelta(seconds=1)
    preroll: deque[bytes] = deque(maxlen=_WAKE_PREROLL_FRAMES)
    while True:
        try:
            frame = await frame_queue.get()
            preroll.append(frame)
            samples = struct.unpack_from("h" * (len(frame) // 2), frame)
            if detector.process(samples) >= 0:
                # Replay the buffered lead-in (incl. the wake word) so the start
                # of the utterance reaches the realtime session uncut. Any frames
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

async def run_realtime_conversation(
        audio: pyaudio.PyAudio,
        frame_queue: asyncio.Queue,
        mic_gate: dict,
        agent_instructions: str,
        log: logging.Logger,
        tools: list[Tool],
        audio_manager: AudioManager,
        ui: AssistantUIBase,
        analytics: Analytics):
    watchdog_task = None
    due_audio_task = None
    send_audio_task = None
    output_stream = None
    ws = None
    # Optional action for the outer loop to run after this session ends (e.g.
    # wake-word training capture). None means just go back to sleep.
    post_action = None

    # The input stream is already open and capturing; mic_gate stays True so that
    # everything the user says after the wake word accumulates in frame_queue
    # while we connect, then gets flushed by _send_audio_loop once the WS is up.
    mic_gate["capture"] = True
    watchdog_control = {"reset_event": asyncio.Event()}

    connect_task = asyncio.create_task(_connect_realtime(agent_instructions, tools))
    output_stream_task = asyncio.create_task(_open_output_stream_async(audio))

    log.info("Connecting to OpenAI Realtime API and opening output stream...")
    ws, output_stream = await asyncio.gather(connect_task, output_stream_task)

    log.info("Starting audio...")
    # Flushes the buffered post-wake audio (and then live frames) to the session.
    send_audio_task = asyncio.create_task(_send_audio_loop(ws, frame_queue))
    output_stream.start_stream()
    due_audio_task = asyncio.create_task(_due_audio_loop(ws, audio_manager, ui))
    
    # Start watchdog timer
    log.info("Starting watchdog timer...")
    watchdog_task = asyncio.create_task(_watchdog_timer(watchdog_control, log, ws))

    log.info("Ready for conversation.")

    analytics.report_event("Conversation")

    try:
        while True:
            async for response in ws:
                res_1=json.loads(response)
                #log.info(f"Received: {res_1.get('type')}")

                # error, log it
                if res_1.get("type") == "error":
                    error = res_1.get("error")
                    log.warning(f'Realtime Error: {error}')

                # generating a response, update state
                if res_1.get("type") == "response.created":
                    mic_gate["capture"] = False
                    ui.update_state(AssistantUIState.TALKING, reason="Assistant responding")

                # finished a response, update state and reset watchdog
                if res_1.get("type") == "response.done":
                    # Reset watchdog timer since assistant finished speaking
                    watchdog_control["reset_event"].set()
                    if ui.state != AssistantUIState.TOOL_CALLING: # if tool calling don't update
                        asyncio.create_task(_reopen_mic_delayed(mic_gate, 500, ui))

                # audio chunk received, play it
                if res_1.get("type") == "response.output_audio.delta":
                    base64_audio_data = res_1.get("delta")
                    if base64_audio_data:
                        pcm_data = base64.b64decode(base64_audio_data)
                        output_stream.write(pcm_data)

                # need to call a function (tool call)
                if res_1.get("type") == "response.function_call_arguments.done":
                    ui.update_state(AssistantUIState.TOOL_CALLING, reason="Starting a tool call")
                    function_name = res_1.get("name")
                    arguments = (res_1.get("arguments"))
                    call_id = res_1.get("call_id")
                    output = None

                    if function_name == "go_to_sleep":
                        # throw an exception
                        log.info("User asked us to go to sleep, ending session.")
                        analytics.report_event("Sleep")
                        raise Exception("User asked us to go to sleep...")
                    elif function_name == "start_wake_word_training":
                        # Swap the session into training mode (dedicated prompt +
                        # minimal tools); Aurora then collects a name and explains
                        # the rules before calling begin_wake_word_capture.
                        log.info("Entering wake word training mode.")
                        analytics.report_event("WakeWordTraining")
                        await _enter_training_session(ws)
                        output = "Entered wake word training mode."
                    elif function_name == "begin_wake_word_capture":
                        # End this session and run the capture routine in the outer
                        # loop, which owns the mic stream and UI.
                        try:
                            args = json.loads(arguments)
                        except Exception:
                            args = {}
                        name = args.get("name") or ""
                        log.info("Starting wake word capture for '%s'.", name)
                        post_action = {"action": "train_wake_word", "name": name}
                        raise Exception("Beginning wake word capture...")
                    elif function_name == "start_cooking":
                        recipe_tool = next((tool for tool in tools if tool.name == "list_recipes"), None)
                        if recipe_tool and recipe_tool.is_configured():
                            try:
                                args = json.loads(arguments)
                            except Exception:
                                output = "Invalid arguments payload"
                            else:
                                recipe_name = args.get("recipe_name")
                                if not recipe_name:
                                    output = "Missing required argument: recipe_name"
                                else:
                                    recipes_dir = recipe_tool._recipes_dir or recipe_tool._resolve_recipes_dir()
                                    if not recipes_dir:
                                        output = "Recipes folder is not configured or does not exist"
                                    else:
                                        recipe_path = (recipes_dir / recipe_name).resolve()
                                        if not recipe_path.is_file() or recipe_path.suffix.lower() != ".md":
                                            output = f"Recipe file not found: {recipe_name}"
                                        else:
                                            try:
                                                with open(recipe_path, "r", encoding="utf-8") as f:
                                                    recipe_content = f.read()
                                                await _update_realtime_session(ws, agent_instructions, recipe_content, tools)

                                                # Stop watchdog timer
                                                log.info("Stopping watchdog timer...")
                                                if watchdog_task:
                                                    watchdog_task.cancel()
                                                    watchdog_task = None

                                                output = f"Started cooking session with recipe {recipe_name}."
                                                log.info(f"Started cooking session with recipe {recipe_name}.")
                                                analytics.report_event("Cooking")
                                            except Exception as e:
                                                log.exception(f"Failed to read recipe file {recipe_name}: {e}")
                                                output = f"Failed to read recipe file {recipe_name}: {e}"
                        else:
                            output = "Recipe tool is not available."
                    elif function_name == "stop_cooking":
                        await _update_realtime_session(ws, agent_instructions, "", tools)
                        # Start watchdog timer
                        log.info("Restarting watchdog timer...")
                        watchdog_task = asyncio.create_task(_watchdog_timer(watchdog_control, log, ws))
                        output = "Stopped cooking session."
                        log.info("Stopped cooking session.")
                    else:
                        for tool in tools:
                            try:
                                output = tool.handle(function_name, arguments)
                                if output:
                                    log.info(f"Tool {tool.name} handled function call {function_name}")
                                    break
                            except Exception as e:
                                log.exception(f"Error in tool {tool.name}: {e}")

                    if output:
                        message = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "output": output,
                                "call_id": call_id
                            }
                        }
                        await ws.send(json.dumps(message))

                        # force generation after tool call
                        message = {
                                "type": "response.create"
                            }
                        await ws.send(json.dumps(message))

    except Exception as e:
        log.info(f"Error or user shutdown in conversation: {e}")
    finally:
        log.info("Conversation over, cleaning up.")
        # Stop sending to the (closing) session; the input stream stays open for
        # the next wake-word cycle. mic_gate is reset by the outer loop.
        mic_gate["capture"] = False
        if watchdog_task:
            watchdog_task.cancel()
        if due_audio_task:
            due_audio_task.cancel()
        if send_audio_task:
            send_audio_task.cancel()
        if output_stream:
            output_stream.stop_stream()
            output_stream.close()
        if ws:
            await ws.close()

    return post_action

async def _watchdog_timer(watchdog_control: dict, log: logging.Logger, ws: websockets.ClientConnection):
    """Watchdog timer that throws an exception if assistant doesn't finish speaking within the timeout."""
    while True:
        try:
            await asyncio.sleep(1)
            # Wait for either timeout or reset event
            await asyncio.wait_for(watchdog_control["reset_event"].wait(), timeout=_REALTIME_WATCHDOG_TIMEOUT_SECONDS)
            # If we get here, the reset event was set, so clear it and continue
            watchdog_control["reset_event"].clear()
        except asyncio.TimeoutError:
            log.info(f"Realtime conversation timeout: Assistant did not speak within {_REALTIME_WATCHDOG_TIMEOUT_SECONDS} seconds")
            await _trigger_sleep(ws)
            break

async def _update_realtime_session(ws: websockets.ClientConnection, agent_instructions: str, recipe: str, tools: list[Tool]):
    all_tools = [tool.manifest() for tool in tools]

    full_instructions = None
    if recipe:
        full_instructions = f"{agent_instructions}{_COOKING_INSTRUCTIONS}{recipe}"
    else:
        full_instructions = agent_instructions

    # always include the go_to_sleep tool
    all_tools.append({
        "name": "go_to_sleep",
        "type": "function",
        "description": "Puts the assistant to sleep and ends the current session.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })

    # always offer wake-word training. The description is the trigger: the model
    # calls this when a user asks to help train/teach the assistant to recognize
    # their voice or the wake word. Handling it swaps in the dedicated training
    # prompt and tool set (see _enter_training_session).
    all_tools.append({
        "name": "start_wake_word_training",
        "type": "function",
        "description": (
            "Start wake-word training. Call this when the user asks to help train or "
            "teach Aurora to recognize their voice or the wake word (e.g. 'help me train "
            "you to recognize my voice', 'train your wake word')."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })

    ## if a recipe folder is configured add start/stop cooking tools
    recipe_tool = next((tool for tool in tools if tool.name == "list_recipes"), None)
    if recipe_tool and recipe_tool.is_configured():
        all_tools.append({
            "name": "start_cooking",
            "type": "function",
            "description": "Starts a cooking session with the specified recipe. Use the list_recipes tool to see available recipes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "The filename of the recipe to start (including .md extension)."
                    }
                },
                "required": ["recipe_name"]
            }
        })
        all_tools.append({
            "name": "stop_cooking",
            "type": "function",
            "description": "Stops the current cooking session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        })

    event = {
        "type": "session.update",
        "session": {
            "model": settings.realtime_model,
            "type": "realtime",
            "audio": {
                "input" : {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "noise_reduction" : {
                        "type": "far_field"
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "create_response": True,
                        "interrupt_response": False,
                        "eagerness": "auto"
                    },
                    "transcription": None,
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "speed": 1,
                    "voice": settings.agent_voice
                }
            },
            "instructions": full_instructions,
            "output_modalities": ['audio'],
            "tools": all_tools,
            "tool_choice": "auto",
        }
    }
    await ws.send(json.dumps(event))

async def _enter_training_session(ws: websockets.ClientConnection):
    """Swap the live session into wake-word training mode.

    Fully replaces the instructions with the dedicated training prompt and the
    tool set with just begin_wake_word_capture + go_to_sleep, so the model stays
    on-script while it collects a name and explains the rules. The regular prompt
    is restored automatically on the next conversation (each one reconnects with
    agent_instructions), so there is no "stop training" tool to call.
    """
    training_tools = [
        {
            "name": "begin_wake_word_capture",
            "type": "function",
            "description": (
                "Begin recording the wake-word training clips. Call this once the user "
                "has given their name and is ready to start saying the words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The first name of the person doing the training."
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "go_to_sleep",
            "type": "function",
            "description": "Puts the assistant to sleep and ends the current session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
    ]

    event = {
        "type": "session.update",
        "session": {
            "model": settings.realtime_model,
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "noise_reduction": {
                        "type": "far_field"
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "create_response": True,
                        "interrupt_response": False,
                        "eagerness": "auto"
                    },
                    "transcription": None,
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "speed": 1,
                    "voice": settings.agent_voice
                }
            },
            "instructions": collect.TRAINING_PROMPT,
            "output_modalities": ['audio'],
            "tools": training_tools,
            "tool_choice": "auto",
        }
    }
    await ws.send(json.dumps(event))

async def _connect_realtime(agent_instructions: str, tools: list[Tool]):
    additional_headers = {
        "Authorization": f"Bearer {settings.openai_api_key}"
    }
    
    ws = await websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={settings.realtime_model}",
        additional_headers=additional_headers,
        ping_interval=_WS_PING_INTERVAL_SECONDS,
        ping_timeout=_WS_PING_TIMEOUT_SECONDS
    )
    
    await _update_realtime_session(ws, agent_instructions, "", tools)

    return ws

async def _trigger_sleep(ws: websockets.ClientConnection):
    try:
        # tell the assistant to go to sleep
        message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Call the go_to_sleep function"
                        }
                    ]
                }
            }
        await ws.send(json.dumps(message))
        messageRespond = {
                "type": "response.create"
            }
        await ws.send(json.dumps(messageRespond))

        # in case that doesn't work, repeat the message
        await asyncio.sleep(1)
        message2 = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "This is important - you need to go to sleep now."
                        }
                    ]
                }
            }
        await ws.send(json.dumps(message2))
        messageRespond2 = {
                "type": "response.create"
            }
        await ws.send(json.dumps(messageRespond2))
    except Exception:
        pass
    
async def _due_audio_loop(ws: websockets.ClientConnection, audio_manager:AudioManager, ui: AssistantUIBase):
    while True:
        await asyncio.sleep(1)
        if audio_manager.has_due_audio() or ui.is_cancel_pressed():
            # go to sleep if we have due audio
            await _trigger_sleep(ws)
        elif audio_manager.has_any_audio():
            # otherwise update any time text
            ui.set_timer_text(audio_manager.audio_to_text())

async def _send_audio_loop(ws: websockets.ClientConnection, frame_queue: asyncio.Queue):
    while True:
        try:
            frame = await frame_queue.get()
            if frame:
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame).decode('utf-8')
                }
                await ws.send(json.dumps(audio_event))       
        except Exception:
            pass

async def _reopen_mic_delayed(mic_gate, delay_ms: int, ui: AssistantUIBase):
    await asyncio.sleep(delay_ms / 1000)
    mic_gate["capture"] = True
    ui.update_state(AssistantUIState.LISTENING, reason="Assistant finished")

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
            rate=_REALTIME_SAMPLERATE,
            input=True,
            frames_per_buffer=_INPUT_FRAMESPERBUFFER,
            input_device_index=settings.input_device_id,
            stream_callback=_make_audio_callback(loop, frame_queue, should_capture),
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

async def _open_output_stream_async(audio: pyaudio.PyAudio):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_REALTIME_SAMPLERATE,
            output=True,
            frames_per_buffer=_REALTIME_FRAMESPERBUFFER,
            output_device_index=settings.output_device_id,
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