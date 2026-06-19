"""Load audio from wake_word/data, augment it, and build feature tensors.

Directory convention (all wavs, any sample rate -- they are resampled to 16 kHz
mono on load)::

    wake_word/data/
        positives/        "aurora" / "hey aurora"          -> label 1
        hard_negatives/   "alexa" + phonetically near words -> label 0
        negatives/        general speech / chatter / silence-> label 0
        background/       noise / music / room tone (used only for augmentation)

Short clips become a single centered window; long files are sliced into
overlapping windows. Optional file-name speaker tags ("kate_001.wav") are used
to group train/val splits so we measure cross-speaker generalization.

This module exposes :func:`build_dataset` for train.py / evaluate.py and can also
be run directly to cache a dataset to .npz.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
import numpy as np

from wake_word import config, features

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - training dependency
    sf = None

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

POSITIVE_DIR = "positives"
HARD_NEG_DIR = "hard_negatives"
NEG_DIR = "negatives"
BACKGROUND_DIR = "background"

# label 1 = wake word, label 0 = everything else.
CATEGORY_LABELS = {POSITIVE_DIR: 1, HARD_NEG_DIR: 0, NEG_DIR: 0}


@dataclass
class Sample:
    features: np.ndarray   # (1, n_mels, NUM_FRAMES)
    label: int
    group: str             # speaker / source tag for grouped splits
    category: str          # subdir name (for per-category metrics)


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    n_dst = int(round(len(x) * dst_sr / src_sr))
    if n_dst <= 1:
        return np.zeros(0, dtype=np.float32)
    src_t = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    dst_t = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def load_wav_16k_mono(path: str) -> np.ndarray:
    """Load any wav as float32 mono at 16 kHz in roughly [-1, 1]."""
    if sf is None:
        raise RuntimeError("soundfile is required: pip install -r wake_word/requirements-train.txt")
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return _resample_linear(data, sr, config.SAMPLE_RATE)


def _speaker_group(path: str, category: str) -> str:
    """Derive a grouping tag from the filename prefix, e.g. 'kate_001.wav'."""
    name = os.path.basename(path)
    if "_" in name:
        return f"{category}:{name.split('_', 1)[0]}"
    return f"{category}:{name}"


def iter_windows(wave: np.ndarray, hop: int | None = None):
    """Yield fixed-length windows. Short clips -> one centered window."""
    w = config.WINDOW_SAMPLES
    if len(wave) <= w:
        yield wave
        return
    hop = hop or w // 2
    for start in range(0, len(wave) - w + 1, hop):
        yield wave[start : start + w]


# --- augmentation --------------------------------------------------------
def _augment(wave: np.ndarray, backgrounds: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    w = config.WINDOW_SAMPLES
    x = wave.astype(np.float32).copy()

    # Pad/trim to a full window first so shifts have room.
    if len(x) < w:
        x = np.concatenate([np.zeros(w - len(x), dtype=np.float32), x])
    elif len(x) > w:
        x = x[:w]

    # Random time shift (circular-ish via zero pad).
    shift = int(rng.integers(-w // 8, w // 8 + 1))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, dtype=np.float32), x[:-shift]])
    elif shift < 0:
        x = np.concatenate([x[-shift:], np.zeros(-shift, dtype=np.float32)])

    # Random gain.
    x *= float(rng.uniform(0.5, 1.4))

    # Mix in background noise at a random SNR.
    if backgrounds and rng.random() < 0.7:
        bg = backgrounds[rng.integers(len(backgrounds))]
        if len(bg) >= w:
            off = int(rng.integers(0, len(bg) - w + 1))
            noise = bg[off : off + w].astype(np.float32)
            sig_p = float(np.mean(x ** 2)) + 1e-9
            noise_p = float(np.mean(noise ** 2)) + 1e-9
            snr_db = float(rng.uniform(0.0, 20.0))
            scale = np.sqrt(sig_p / (noise_p * (10 ** (snr_db / 10.0))))
            x = x + scale * noise

    np.clip(x, -1.0, 1.0, out=x)
    return x


def _load_backgrounds() -> list[np.ndarray]:
    bg_dir = os.path.join(DATA_DIR, BACKGROUND_DIR)
    out = []
    for path in sorted(glob.glob(os.path.join(bg_dir, "*.wav"))):
        try:
            out.append(load_wav_16k_mono(path))
        except Exception:
            pass
    return out


def build_dataset(
    data_dir: str = DATA_DIR,
    augment: bool = True,
    augment_factor: int = 4,
    seed: int = 0,
) -> list[Sample]:
    """Build a list of :class:`Sample` from the data directory."""
    rng = np.random.default_rng(seed)
    backgrounds = _load_backgrounds() if augment else []
    samples: list[Sample] = []

    for category, label in CATEGORY_LABELS.items():
        cat_dir = os.path.join(data_dir, category)
        paths = sorted(glob.glob(os.path.join(cat_dir, "*.wav")))
        for path in paths:
            try:
                wave = load_wav_16k_mono(path)
            except Exception as exc:  # noqa: BLE001
                print(f"  skipped {path}: {exc}")
                continue
            if len(wave) == 0:
                continue
            group = _speaker_group(path, category)

            for window in iter_windows(wave):
                feat = features.waveform_to_model_input(window)[0]  # (1, n_mels, frames)
                samples.append(Sample(feat, label, group, category))

                if augment:
                    for _ in range(augment_factor):
                        aug = _augment(window, backgrounds, rng)
                        feat_a = features.waveform_to_model_input(aug)[0]
                        samples.append(Sample(feat_a, label, group, category))

    return samples


def to_arrays(samples: list[Sample]):
    X = np.stack([s.features for s in samples]).astype(np.float32)  # (N,1,n_mels,frames)
    y = np.array([s.label for s in samples], dtype=np.float32)
    groups = np.array([s.group for s in samples])
    categories = np.array([s.category for s in samples])
    return X, y, groups, categories


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and cache the wake word dataset.")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--augment-factor", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "dataset.npz"))
    args = ap.parse_args()

    samples = build_dataset(
        args.data_dir, augment=not args.no_augment, augment_factor=args.augment_factor
    )
    if not samples:
        print("No samples found. Add wavs under wake_word/data/ (see README).")
        return
    X, y, groups, categories = to_arrays(samples)
    np.savez_compressed(args.out, X=X, y=y, groups=groups, categories=categories)
    print(f"Saved {len(samples)} samples ({int(y.sum())} positive) -> {args.out}")


if __name__ == "__main__":
    main()
