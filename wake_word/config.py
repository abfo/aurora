"""Shared constants for the Aurora wake word system.

These values define the audio frontend and the model input shape. They are
imported by both the training pipeline and the live detector, so changing one
here keeps train-time and inference-time behaviour in lock-step. If you change
any feature constant you MUST retrain the model.
"""

from __future__ import annotations

# --- Audio ---------------------------------------------------------------
# 24 kHz matches the OpenAI realtime API rate, so a single always-open mic
# stream feeds both the wake word detector and the realtime session with no
# resampling or stream reopen (see main.py).
SAMPLE_RATE = 24000          # Hz, mono.
FRAME_LENGTH = 1024          # nominal mic chunk (~43 ms @ 24 kHz).

# --- Analysis window -----------------------------------------------------
# The model classifies a sliding window of audio. 1.2s comfortably contains
# "Aurora" / "Hey Aurora" spoken at a natural pace.
WINDOW_SECONDS = 1.2
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)   # 28800

# --- Log-mel spectrogram -------------------------------------------------
# Sized to ~32 ms window / 10 ms hop, matching the speech band. FMAX is capped
# at 8 kHz (speech energy) rather than Nyquist so the features are insensitive
# to source bandwidth.
N_FFT = 768                  # ~32 ms @ 24 kHz
HOP_LENGTH = 240             # 10 ms hop @ 24 kHz
N_MELS = 40
FMIN = 20.0
FMAX = 8000.0

# Number of mel frames produced for a full window (see features.num_frames()).
# 1 + (28800 - 768) // 240 = 117
NUM_FRAMES = 1 + (WINDOW_SAMPLES - N_FFT) // HOP_LENGTH

# --- Inference behaviour -------------------------------------------------
# How often (in samples) the detector re-runs the model. ~100 ms keeps CPU low
# on the Pi while still being responsive.
EVAL_HOP_SAMPLES = 2400      # 100 ms

# Detection smoothing / debouncing. A hit requires the smoothed probability to
# stay above threshold for this many consecutive evaluations.
SMOOTHING_WINDOW = 3
TRIGGER_CONSECUTIVE = 2

# Default detection threshold. Deliberately permissive (favouring false
# positives over false negatives) because the OpenAI realtime session shuts
# down quickly if it does not hear a real conversation. Overridable via the
# WAKE_WORD_THRESHOLD setting and the metadata saved alongside the model.
DEFAULT_THRESHOLD = 0.5

# Minimum seconds between two triggers (refractory period) so a single
# utterance does not fire repeatedly.
REFRACTORY_SECONDS = 1.5

# Energy gate: windows quieter than this normalized RMS are treated as silence
# and never evaluated. Set low so it only rejects essentially-silent audio
# (which can never be a wake word) without hurting recall, and it saves CPU.
ENERGY_GATE_RMS = 0.005

# When auto-selecting a detection threshold during training, never go below this
# floor. Keeps a safety margin against garbage/near-silence even when the
# training data is cleanly separable.
MIN_AUTO_THRESHOLD = 0.3

# Threshold selection treats the hard-negative ("Alexa") trigger rate as the
# binding constraint and otherwise maximizes recall. Random speech triggering
# the wake word is acceptable (the realtime session just closes), but the
# household's Alexa devices must not. See train.py:_choose_threshold.
MAX_HARD_NEGATIVE_RATE = 0.05

# Default location of the exported model + metadata, relative to the repo root.
DEFAULT_MODEL_PATH = "wake_word/models/aurora.onnx"
DEFAULT_METADATA_PATH = "wake_word/models/aurora.json"
