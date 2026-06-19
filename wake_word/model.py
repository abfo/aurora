"""CNN definition for the Aurora wake word classifier.

Training only -- imports torch, which is NOT installed on the Raspberry Pi.
The live detector loads the exported ONNX file instead (see detector.py).

The network is intentionally tiny (~100-200k params): a few conv blocks, global
average pooling, then a single logit. It exports cleanly to ONNX and runs in well
under a millisecond on a Pi 4 CPU.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from wake_word import config


class WakeWordCNN(nn.Module):
    """Small 2-D CNN over a (1, n_mels, frames) log-mel image -> 1 logit."""

    def __init__(self, n_mels: int = config.N_MELS) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
            block(32, 64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)  # (B, 1) raw logit


class SigmoidWrapper(nn.Module):
    """Wraps the model so the exported ONNX graph outputs a probability.

    This keeps the detector dependency-light: onnxruntime returns the
    detection probability directly, no sigmoid needed on the Pi.
    """

    def __init__(self, model: WakeWordCNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x)).squeeze(-1)  # (B,)


def export_onnx(model: WakeWordCNN, path: str) -> None:
    """Export ``model`` (probability output) to ONNX at ``path``."""
    model.eval()
    wrapped = SigmoidWrapper(model).eval()
    dummy = torch.zeros(1, 1, config.N_MELS, config.NUM_FRAMES, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        dummy,
        path,
        input_names=["features"],
        output_names=["probability"],
        dynamic_axes={"features": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=17,
        verbose=False,
    )

    # The exporter may stash weights in an external "<path>.data" file. Inline
    # them so the model is a single self-contained file that's easy to deploy.
    ext_data = path + ".data"
    if os.path.exists(ext_data):
        import onnx
        model_proto = onnx.load(path)  # pulls in the external weights
        onnx.save_model(model_proto, path, save_as_external_data=False)
        os.remove(ext_data)
