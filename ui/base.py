from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional


class AssistantUIState(Enum):
    """Conversation UI states for the assistant.

    - LOADING: Application is starting or resources initializing.
    - SLEEPING: Idle and waiting for wake word / user interaction.
    - TALKING: Assistant is speaking.
    - LISTENING: Assistant is recording/listening to the user.
    - TOOL_CALLING: Assistant is calling a tool/function.
    """

    LOAD_START = auto()
    LOAD_INTERNET = auto()
    LOAD_DISPLAY = auto()
    LOAD_AUDIO = auto()
    SLEEPING = auto()
    TALKING = auto()
    LISTENING = auto()
    TOOL_CALLING = auto()


class AssistantUIBase(ABC):
    """Base class for assistant UI implementations.

    Concrete UIs (CLI, Tkinter, Web, etc.) should subclass this and implement
    the abstract hooks. The base class stores state and provides helpers for
    safe state transitions and timer display updates.
    """

    def __init__(self) -> None:
        self._state: AssistantUIState = AssistantUIState.LOAD_START
        self._timer_text: str = ""

    # -------------------- Public API --------------------
    @property
    def state(self) -> AssistantUIState:
        return self._state

    def update_state(self, state: AssistantUIState, reason: Optional[str] = None) -> None:
        """Update the current UI state.

        Implementations can override on_state_changed to react (e.g., update
        LEDs, text, animations). This method is synchronous and should remain
        fast; do long work asynchronously in the subclass if needed.

        Args:
            state: New UI state.
            reason: Optional human-readable note for logging/diagnostics.
        """
        if not isinstance(state, AssistantUIState):
            raise TypeError("state must be an AssistantUIState")

        if state is self._state:
            # No-op if unchanged, but still allow subclass to observe.
            self.on_state_changed(self._state, state, reason)
            return

        prev = self._state
        self._state = state
        self.on_state_changed(prev, state, reason)

    def set_timer_text(self, text: str) -> None:
        """Set or clear the timer display text.

        Args:
            text: A short string to show in the UI. Empty string clears it.
        """
        self._timer_text = text or ""
        self.on_timer_text_changed(self._timer_text)

    def get_timer_text(self) -> str:
        """Return current timer display text (may be empty)."""
        return self._timer_text

    # -------------------- Hooks for subclasses --------------------
    def on_state_changed(
        self,
        previous: AssistantUIState,
        current: AssistantUIState,
        reason: Optional[str] = None,
    ) -> None:
        """Hook called whenever update_state is invoked.

        Default implementation is a no-op; subclasses may override.
        """

    def on_timer_text_changed(self, text: str) -> None:
        """Hook called when timer text changes. Default is no-op."""

    # -------------------- User controls --------------------
    @abstractmethod
    def is_cancel_pressed(self) -> bool:
        """Return True if the cancel button is currently pressed."""

    @abstractmethod
    def is_shutdown_pressed(self) -> bool:
        """Return True if the shutdown button is currently pressed."""

    # -------------------- Lifecycle --------------------
    @abstractmethod
    def shutdown(self) -> None:
        """Clean up UI resources before application exit."""
