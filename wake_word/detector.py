"""WakeWordDetector -- a Porcupine-compatible local wake word detector.

Drop-in replacement for ``pvporcupine``. It exposes the same surface the wake
word loop in ``main.py`` relies on::

    detector = WakeWordDetector(model_path)
    detector.sample_rate     # 16000
    detector.frame_length    # 512
    idx = detector.process(sample)   # sample = tuple of int16; >=0 means detected
    detector.delete()

Internally it keeps a sliding ~1.2s ring buffer, runs the ONNX model every
~100 ms, smooths the probability, and fires when it stays above threshold for a
couple of consecutive evaluations (with a short refractory period afterwards).

Runtime dependencies: numpy + onnxruntime only.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque

import numpy as np
import onnxruntime as ort

from wake_word import config, features

log = logging.getLogger("aurora.wakeword")


class WakeWordDetector:
    def __init__(
        self,
        model_path: str | None = None,
        threshold: float | None = None,
        metadata_path: str | None = None,
        sample_rate: int = config.SAMPLE_RATE,
        frame_length: int = config.FRAME_LENGTH,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_length = frame_length

        self.model_path = model_path or config.DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Wake word model not found: {self.model_path}. "
                "Train one with wake_word/scripts/train.py (see wake_word/README.md)."
            )

        # Resolve threshold: explicit arg > metadata file > package default.
        meta_path = metadata_path or os.path.splitext(self.model_path)[0] + ".json"
        meta_threshold = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta_threshold = meta.get("threshold")
            except Exception:
                log.warning("Could not read wake word metadata %s", meta_path)
        if threshold is not None:
            self.threshold = float(threshold)
        elif meta_threshold is not None:
            self.threshold = float(meta_threshold)
        else:
            self.threshold = config.DEFAULT_THRESHOLD

        # Single-threaded session keeps Pi CPU usage predictable.
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            self.model_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # Sliding window of raw int16 samples.
        self._buffer = np.zeros(config.WINDOW_SAMPLES, dtype=np.int16)
        self._filled = 0
        self._samples_since_eval = 0
        self._probs: deque[float] = deque(maxlen=config.SMOOTHING_WINDOW)
        self._consecutive = 0
        self._last_trigger = 0.0

        log.info(
            "WakeWordDetector loaded model=%s threshold=%.3f sr=%d frame=%d",
            self.model_path, self.threshold, self._sample_rate, self._frame_length,
        )

    # --- Porcupine-compatible surface ------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_length(self) -> int:
        return self._frame_length

    def process(self, pcm) -> int:
        """Feed one frame of int16 PCM. Returns 0 on detection, else -1."""
        frame = np.asarray(pcm, dtype=np.int16)

        n = len(frame)
        # Append to the ring buffer (shift left by n, drop oldest).
        if n >= config.WINDOW_SAMPLES:
            self._buffer[:] = frame[-config.WINDOW_SAMPLES:]
        else:
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = frame
        self._filled = min(self._filled + n, config.WINDOW_SAMPLES)
        self._samples_since_eval += n

        # Only evaluate every ~EVAL_HOP_SAMPLES, once the window is full.
        if self._filled < config.WINDOW_SAMPLES:
            return -1
        if self._samples_since_eval < config.EVAL_HOP_SAMPLES:
            return -1
        self._samples_since_eval = 0

        # Energy gate: skip essentially-silent windows (never a wake word).
        rms = float(np.sqrt(np.mean((self._buffer.astype(np.float32) / 32768.0) ** 2)))
        if rms < config.ENERGY_GATE_RMS:
            self._probs.clear()
            self._consecutive = 0
            return -1

        prob = self._infer(self._buffer)
        self._probs.append(prob)
        smoothed = float(np.mean(self._probs))

        # Refractory period: ignore detections shortly after the last one.
        now = time.monotonic()
        in_refractory = (now - self._last_trigger) < config.REFRACTORY_SECONDS

        if smoothed >= self.threshold:
            self._consecutive += 1
        else:
            self._consecutive = 0

        if self._consecutive >= config.TRIGGER_CONSECUTIVE and not in_refractory:
            self._last_trigger = now
            self._consecutive = 0
            self._probs.clear()
            log.debug("Wake word fired (smoothed prob=%.3f)", smoothed)
            return 0
        return -1

    def delete(self) -> None:
        self._session = None

    # --- internals -------------------------------------------------------
    def _infer(self, window: np.ndarray) -> float:
        feat = features.waveform_to_model_input(window)
        out = self._session.run(None, {self._input_name: feat})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def predict_proba(self, waveform: np.ndarray) -> float:
        """Convenience for offline tests: probability for a full waveform."""
        return self._infer(np.asarray(waveform))
