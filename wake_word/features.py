"""Pure-numpy log-mel spectrogram frontend.

This is the single source of truth for turning a 16 kHz mono waveform into the
2-D feature the CNN consumes. It is imported by BOTH the training data pipeline
and the live :class:`~wake_word.detector.WakeWordDetector`, which guarantees
train/inference parity without pulling librosa onto the Raspberry Pi.

Only depends on numpy.
"""

from __future__ import annotations

import numpy as np

from wake_word import config

_EPS = 1e-6


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """HTK mel scale (simple and self-consistent)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    n_mels: int = config.N_MELS,
    fmin: float = config.FMIN,
    fmax: float = config.FMAX,
) -> np.ndarray:
    """Triangular mel filterbank of shape (n_mels, n_fft // 2 + 1)."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)

    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]

    weights = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    return weights.astype(np.float32)


# Precompute once at import (cheap, ~40x257 floats) and reuse on every window.
_MEL_FB = _mel_filterbank()
_HANN = np.hanning(config.N_FFT).astype(np.float32)


def num_frames(num_samples: int = config.WINDOW_SAMPLES) -> int:
    """Number of STFT frames for a signal of ``num_samples`` (no centering)."""
    if num_samples < config.N_FFT:
        return 0
    return 1 + (num_samples - config.N_FFT) // config.HOP_LENGTH


def waveform_to_logmel(waveform: np.ndarray) -> np.ndarray:
    """Convert a 1-D waveform into a normalized log-mel feature.

    Args:
        waveform: 1-D float array in roughly [-1, 1] (int16 is auto-scaled).

    Returns:
        float32 array of shape (n_mels, frames), per-window standardized
        (zero mean, unit variance) so the result is robust to input gain.
    """
    x = np.asarray(waveform)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)

    frames = num_frames(len(x))
    if frames <= 0:
        return np.zeros((config.N_MELS, 0), dtype=np.float32)

    # Frame the signal (frames, n_fft) via a strided view.
    idx = (
        np.arange(config.N_FFT)[None, :]
        + config.HOP_LENGTH * np.arange(frames)[:, None]
    )
    framed = x[idx] * _HANN  # (frames, n_fft)

    spectrum = np.fft.rfft(framed, n=config.N_FFT, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)  # (frames, bins)

    mel = power @ _MEL_FB.T              # (frames, n_mels)
    log_mel = np.log(mel + _EPS).T       # (n_mels, frames)

    # Per-window standardization -> gain invariant.
    mean = log_mel.mean()
    std = log_mel.std()
    log_mel = (log_mel - mean) / (std + _EPS)
    return log_mel.astype(np.float32)


def waveform_to_model_input(waveform: np.ndarray) -> np.ndarray:
    """Produce a (1, 1, n_mels, NUM_FRAMES) batch tensor for the CNN.

    The waveform is right-aligned/truncated/zero-padded to exactly
    ``config.WINDOW_SAMPLES`` so the feature has the fixed shape the model expects.
    """
    x = np.asarray(waveform)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)

    target = config.WINDOW_SAMPLES
    if len(x) < target:
        x = np.concatenate([np.zeros(target - len(x), dtype=np.float32), x])
    elif len(x) > target:
        x = x[-target:]

    feat = waveform_to_logmel(x)  # (n_mels, NUM_FRAMES)
    # Defensive: pad/trim the frame axis to NUM_FRAMES.
    if feat.shape[1] < config.NUM_FRAMES:
        pad = config.NUM_FRAMES - feat.shape[1]
        feat = np.pad(feat, ((0, 0), (0, pad)))
    elif feat.shape[1] > config.NUM_FRAMES:
        feat = feat[:, : config.NUM_FRAMES]
    return feat[None, None, :, :].astype(np.float32)
