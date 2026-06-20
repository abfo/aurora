"""Guided, in-app collection of wake-word training samples.

This module powers the voice-driven "help me train you to recognize my voice"
flow. It holds the dedicated training-mode prompt (swapped into the realtime
session so the small model stays tightly on-script), plus the routine that
records positive ("Aurora") and negative ("Alexa" / general speech) clips from
the already-open mic stream and writes them to disk for later training.

Clips are saved as 24 kHz mono int16 WAV (matching wake_word/config and the
training pipeline) to ``<collect_dir>/positives`` and ``<collect_dir>/negatives``
with ``<speaker>_<uuid>.wav`` filenames. The speaker prefix lets the training
dataset group train/val splits by voice (see prepare_dataset._speaker_group),
and the uuid guarantees repeated sessions never overwrite earlier recordings.

When wake-word training is enabled, runtime activations are also captured here
via :func:`save_activation_clip` into ``<collect_dir>/activations`` (same format)
to be reviewed later and sorted into positives/ or negatives/.

Nothing is uploaded or trained automatically - the files are left on disk to be
gathered and fed into ``wake_word/scripts/train.py`` manually.
"""

from __future__ import annotations

import asyncio
import os
import uuid
import wave

from wake_word import config

# Prompt that fully replaces Aurora's normal instructions while training mode is
# active. Kept short and imperative because the realtime model is not very smart
# and must stay on-script.
TRAINING_PROMPT = """You are Aurora, a friendly cartoon squid home assistant, now in WAKE WORD TRAINING MODE.

Your ONLY job right now is to collect a short set of voice recordings to help you
better recognize when someone says your name. Do not answer unrelated questions or
get sidetracked - if the user asks for anything else, gently steer them back to
training (or, if they clearly want to stop, call go_to_sleep).

Follow these steps:
1. Ask who is doing the training and capture their first name.
2. Briefly explain the routine, in your own voice:
   - They will say "Aurora" or "Hey Aurora" 10 times. The lights turn GREEN each
     time - say it once after each green flash.
   - Then they will say a different phrase like "Alexa" or "What's up" 5 times. The
     lights turn RED each time - say it once after each red flash.
   - Tell them you will go quiet while recording and will wake up again when it's
     done, and to just watch the lights and speak right after each colour change.
3. When they confirm they are ready, call begin_wake_word_capture with their name.

Keep it warm but brief. Do not narrate that you are calling a tool."""

# How many of each kind of clip to capture per session.
POSITIVE_COUNT = 10
NEGATIVE_COUNT = 5

# Length of each recorded clip, and the small pauses around it.
CLIP_SECONDS = 1.5
LEAD_IN_SECONDS = 0.7   # pause after the colour cue so the speaker can react
GAP_SECONDS = 0.6       # lights-off gap between clips so cues read as discrete

# Subfolders under the collection dir; map to the training data layout.
POSITIVE_LABEL = "positives"
NEGATIVE_LABEL = "negatives"
ACTIVATION_LABEL = "activations"


def sanitize_speaker(name: str | None) -> str:
    """Turn a spoken name into a safe filename speaker tag.

    Lowercased, ``[a-z0-9]`` only (spaces/underscores dropped) so the
    first-underscore split in prepare_dataset._speaker_group recovers the
    speaker cleanly. Falls back to ``"trainer"`` if nothing usable remains.
    """
    cleaned = "".join(c for c in (name or "").lower() if c.isascii() and c.isalnum())
    return cleaned or "trainer"


def _drain_queue(q: asyncio.Queue) -> None:
    """Discard any buffered frames so a capture starts on fresh audio."""
    try:
        while True:
            q.get_nowait()
    except asyncio.QueueEmpty:
        pass


def _write_wav(path: str, pcm: bytes) -> None:
    """Write raw int16 PCM as a 24 kHz mono WAV (see record_samples.py)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm)


def save_activation_clip(pcm: bytes, collect_dir: str, log) -> str:
    """Write one wake-word activation as a negative-candidate WAV under
    ``<collect_dir>/activations``. Same 24 kHz mono int16 format as the guided
    negatives; reviewed and sorted into positives/ or negatives/ manually. The
    ``activation_`` prefix groups these consistently in train/val splits (see
    prepare_dataset._speaker_group)."""
    out_dir = os.path.join(collect_dir, ACTIVATION_LABEL)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"activation_{uuid.uuid4().hex}.wav")
    _write_wav(path, pcm)
    log.info("Saved wake-word activation clip %s", path)
    return path


async def _capture_clip(frame_queue: asyncio.Queue) -> bytes:
    """Collect roughly CLIP_SECONDS of audio from the live mic queue."""
    target_samples = int(config.SAMPLE_RATE * CLIP_SECONDS)
    collected = 0
    chunks: list[bytes] = []
    while collected < target_samples:
        frame = await frame_queue.get()
        if not frame:
            continue
        chunks.append(frame)
        collected += len(frame) // 2  # int16 -> 2 bytes per sample
    return b"".join(chunks)


async def collect_training_samples(
    frame_queue: asyncio.Queue,
    mic_gate: dict,
    ui,
    name: str,
    log,
    *,
    collect_dir: str,
) -> None:
    """Record POSITIVE_COUNT positive then NEGATIVE_COUNT negative clips.

    Reuses the assistant's existing always-open mic stream via ``frame_queue``
    (no second PyAudio stream is opened). Drives the UI with green/red cues per
    clip; the Debug UI logs a text cue instead (Windows has no lights).
    """
    speaker = sanitize_speaker(name)
    log.info("Starting wake word training capture for speaker '%s' -> %s", speaker, collect_dir)

    plan = [(POSITIVE_LABEL, POSITIVE_COUNT), (NEGATIVE_LABEL, NEGATIVE_COUNT)]

    # Make sure capture is on and the output folders exist.
    mic_gate["capture"] = True
    for label, _ in plan:
        os.makedirs(os.path.join(collect_dir, label), exist_ok=True)

    saved = {POSITIVE_LABEL: 0, NEGATIVE_LABEL: 0}
    try:
        for label, count in plan:
            out_dir = os.path.join(collect_dir, label)
            for i in range(count):
                ui.show_training_prompt(label, i + 1, count)
                # Give the speaker a beat to react to the colour change, then
                # start from clean audio so we don't capture the previous clip.
                await asyncio.sleep(LEAD_IN_SECONDS)
                _drain_queue(frame_queue)

                pcm = await _capture_clip(frame_queue)

                path = os.path.join(out_dir, f"{speaker}_{uuid.uuid4().hex}.wav")
                await asyncio.to_thread(_write_wav, path, pcm)
                saved[label] += 1
                log.info("Saved training clip %s", path)

                ui.clear_training_lights()
                await asyncio.sleep(GAP_SECONDS)
    finally:
        ui.clear_training_lights()

    log.info(
        "Wake word training capture complete: %d positives, %d negatives in %s",
        saved[POSITIVE_LABEL], saved[NEGATIVE_LABEL], collect_dir,
    )
