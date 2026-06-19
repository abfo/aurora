"""Generate synthetic audio to smoke-test the training/inference pipeline.

This is NOT real wake word data -- it fabricates separable signals so we can
verify the end-to-end plumbing (wav -> features -> train -> onnx -> detector)
runs and learns, without needing a microphone or the OpenAI API.

    python wake_word/scripts/_make_synth_data.py --out wake_word/data_synth

Positives, hard-negatives and negatives use distinct spectral signatures.
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np
import soundfile as sf

from wake_word import config

SR = config.SAMPLE_RATE


def _tone(freqs, dur, rng, env=True):
    t = np.arange(int(SR * dur)) / SR
    x = np.zeros_like(t)
    for f in freqs:
        x += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    if env:
        # Syllabic amplitude modulation to look speech-like.
        x *= 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3, 6) * t))
    x += 0.01 * rng.standard_normal(len(x))
    x /= (np.max(np.abs(x)) + 1e-9)
    return (0.6 * x).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_synth"))
    ap.add_argument("--per-speaker", type=int, default=8)
    ap.add_argument("--speakers", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    cats = {
        "positives": [430, 900, 2100],
        "hard_negatives": [520, 1500],
        "negatives": [300, 1200, 3000],
    }
    for cat in list(cats) + ["background"]:
        os.makedirs(os.path.join(args.out, cat), exist_ok=True)

    for cat, freqs in cats.items():
        for s in range(args.speakers):
            for i in range(args.per_speaker):
                jitter = [f * rng.uniform(0.95, 1.05) for f in freqs]
                wav = _tone(jitter, dur=1.0, rng=rng)
                path = os.path.join(args.out, cat, f"s{s}_{i:03d}.wav")
                sf.write(path, wav, SR, subtype="PCM_16")

    # A couple of background noise files for augmentation.
    for i in range(3):
        noise = (0.2 * rng.standard_normal(int(SR * 3))).astype(np.float32)
        sf.write(os.path.join(args.out, "background", f"noise_{i}.wav"), noise, SR, subtype="PCM_16")

    print(f"Synthetic data written to {args.out}")


if __name__ == "__main__":
    main()
