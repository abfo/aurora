"""Record real wake word / negative samples from a microphone.

Run this once per household member to capture how each person actually says
"Aurora" (and, for hard negatives, "Alexa"). Real recordings are the single
biggest lever on real-world accuracy.

Usage::

    # Each person records ~15 "Aurora" clips:
    python wake_word/scripts/record_samples.py --speaker kate --label positives --count 15

    # A few "Alexa" clips so the model learns to reject it:
    python wake_word/scripts/record_samples.py --speaker kate --label hard_negatives --count 8

Files are saved as 24 kHz mono WAV to wake_word/data/<label>/<speaker>_<n>.wav,
which also tags the speaker for cross-speaker train/val splits.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import wave

import _bootstrap  # noqa: F401

from prepare_dataset import DATA_DIR
from wake_word import config

LABELS = ["positives", "hard_negatives", "negatives"]
PROMPTS = {
    "positives": 'Say "Aurora"',
    "hard_negatives": 'Say "Alexa" (or a similar word)',
    "negatives": "Say any normal sentence",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Record wake word samples from the mic.")
    ap.add_argument("--speaker", required=True, help="Short speaker tag, e.g. kate")
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--count", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--device", type=int, default=None, help="PyAudio input device index")
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()

    import pyaudio

    out_dir = os.path.join(args.data_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    # Continue numbering after any existing files for this speaker.
    existing = glob.glob(os.path.join(out_dir, f"{args.speaker}_*.wav"))
    start_idx = len(existing)

    pa = pyaudio.PyAudio()
    frames_per_buffer = config.FRAME_LENGTH
    n_frames = int(config.SAMPLE_RATE * args.seconds / frames_per_buffer)

    print(f"\nRecording {args.count} '{args.label}' clips for speaker '{args.speaker}'.")
    print(f"Prompt: {PROMPTS[args.label]}\n")

    try:
        for i in range(args.count):
            idx = start_idx + i
            for c in (3, 2, 1):
                print(f"  clip {i + 1}/{args.count} in {c}...", end="\r", flush=True)
                time.sleep(0.6)
            print(f"  clip {i + 1}/{args.count}: RECORDING -- {PROMPTS[args.label]}   ")

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frames_per_buffer,
                input_device_index=args.device,
            )
            chunks = []
            for _ in range(n_frames):
                chunks.append(stream.read(frames_per_buffer, exception_on_overflow=False))
            stream.stop_stream()
            stream.close()

            path = os.path.join(out_dir, f"{args.speaker}_{idx:03d}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(config.SAMPLE_RATE)
                wf.writeframes(b"".join(chunks))
            print(f"    saved {path}")
            time.sleep(0.3)
    finally:
        pa.terminate()

    print("\nDone. Re-run with a different --speaker for each household member.")


if __name__ == "__main__":
    main()
