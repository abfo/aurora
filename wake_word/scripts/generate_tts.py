"""Bootstrap wake word training data with OpenAI text-to-speech.

Generates many spoken variants of the wake word ("Aurora"), the key hard
negative ("Alexa") and assorted general speech, across all available TTS voices
and a range of speaking styles, so the model generalises across voices before
you have collected many real recordings.

Reuses your existing OPENAI_API_KEY (from .env / settings). Clips are short, so
this is inexpensive, but use --dry-run first to see the plan and counts.

Usage::

    python wake_word/scripts/generate_tts.py --dry-run
    python wake_word/scripts/generate_tts.py
    python wake_word/scripts/generate_tts.py --voices alloy,nova --variations 2
"""

from __future__ import annotations

import argparse
import io
import os

import _bootstrap  # noqa: F401
import numpy as np
import soundfile as sf

from prepare_dataset import DATA_DIR, HARD_NEG_DIR, NEG_DIR, POSITIVE_DIR, _resample_linear
from settings import settings
from wake_word import config

DEFAULT_VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]

# (category, phrases)
PHRASES = {
    POSITIVE_DIR: [
        "Aurora",
        "Hey Aurora",
        "Aurora?",
        "Okay Aurora",
        "Aurora, are you there?",
        "Aurora, what's the weather?",
    ],
    HARD_NEG_DIR: [
        "Alexa",
        "Hey Alexa",
        "Alexa, play some music",
        "Alexa, set a timer",
        "Aura",
        "Laura",
        "an order",
        "or a",
    ],
    NEG_DIR: [
        "What time is it?",
        "Let's have dinner soon",
        "Can you pass the salt?",
        "The weather is lovely today",
        "I'm heading to the store",
        "Turn off the lights please",
    ],
}

STYLES = [
    "Speak naturally and clearly.",
    "Speak quickly and casually.",
    "Speak slowly and calmly.",
    "Speak in a cheerful, upbeat tone.",
    "Speak with a British accent.",
    "Speak softly, as if across the room.",
]


def _save_resampled(audio_bytes: bytes, out_path: str) -> None:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = _resample_linear(data, sr, config.SAMPLE_RATE)
    sf.write(out_path, data, config.SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate wake word data via OpenAI TTS.")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                    help="Comma-separated TTS voices.")
    ap.add_argument("--variations", type=int, default=1,
                    help="Speaking-style variations per (voice, phrase).")
    ap.add_argument("--model", default="gpt-4o-mini-tts")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    rng = np.random.default_rng(args.seed)

    plan = []
    for category, phrases in PHRASES.items():
        for voice in voices:
            for p_idx, phrase in enumerate(phrases):
                for v in range(args.variations):
                    style = STYLES[rng.integers(len(STYLES))]
                    fname = f"tts-{voice}_{category[:3]}{p_idx:02d}{v:02d}.wav"
                    plan.append((category, voice, phrase, style, fname))

    n_by_cat = {c: sum(1 for x in plan if x[0] == c) for c in PHRASES}
    print("Plan:")
    for c, n in n_by_cat.items():
        print(f"  {c}: {n} clips")
    print(f"  total: {len(plan)} clips across {len(voices)} voices")

    if args.dry_run:
        print("\n--dry-run: nothing generated.")
        return

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set (see .env).")

    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)

    for category in PHRASES:
        os.makedirs(os.path.join(args.data_dir, category), exist_ok=True)

    done = 0
    for category, voice, phrase, style, fname in plan:
        out_path = os.path.join(args.data_dir, category, fname)
        if os.path.exists(out_path):
            done += 1
            continue
        try:
            resp = client.audio.speech.create(
                model=args.model,
                voice=voice,
                input=phrase,
                instructions=style,
                response_format="wav",
            )
            _save_resampled(resp.read(), out_path)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(plan)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  failed {voice}/{phrase!r}: {exc}")

    print(f"Generated {done} clips into {args.data_dir}")


if __name__ == "__main__":
    main()
