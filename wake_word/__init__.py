"""Aurora self-trained wake word package.

This package replaces the previous Picovoice Porcupine dependency with a small,
fully self-contained wake word detector that we train ourselves.

Runtime (laptop + Raspberry Pi) only needs ``numpy`` and ``onnxruntime`` and uses:
    - :mod:`wake_word.config`   - shared feature/model constants
    - :mod:`wake_word.features` - pure-numpy log-mel frontend (train/inference parity)
    - :mod:`wake_word.detector` - WakeWordDetector, a drop-in replacement for Porcupine

Training (PC / Colab only) additionally uses :mod:`wake_word.model` and the scripts
under ``wake_word/scripts``. See ``wake_word/README.md``.

Note: importing :class:`WakeWordDetector` is done explicitly via
``from wake_word.detector import WakeWordDetector`` so that feature-only / training
code does not require onnxruntime to be installed.
"""
