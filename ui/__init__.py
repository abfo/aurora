"""UI package exposing the assistant UI base and simple debug implementation."""

from .base import AssistantUIBase, AssistantUIState
from .debug import DebugUI

# Try to expose BraincraftUI when available (e.g., on Raspberry Pi)
try:
    from .braincraft import BraincraftUI
except Exception:  # ImportError or hardware libs missing
    BraincraftUI = None  # type: ignore

__all__ = [
    "AssistantUIBase",
    "AssistantUIState",
    "DebugUI",
    "BraincraftUI",
]
