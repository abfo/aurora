from __future__ import annotations

import logging

from .base import AssistantUIBase, AssistantUIState


class DebugUI(AssistantUIBase):
    """Debug implementation of the Assistant UI.

    Logs state transitions and timer text updates via the project's logging
    infrastructure. Buttons are considered never pressed.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self._log = logger or logging.getLogger("aurora.ui.debug")

    # Hooks
    def on_state_changed(
        self,
        previous: AssistantUIState,
        current: AssistantUIState,
        reason: str | None = None,
    ) -> None:
        if reason:
            self._log.info("UI state: %s -> %s (reason=%s)", previous.name, current.name, reason)
        else:
            self._log.info("UI state: %s -> %s", previous.name, current.name)

    def on_timer_text_changed(self, text: str) -> None:
        level = logging.DEBUG if text else logging.DEBUG
        self._log.log(level, "Timer text: %s", text if text else "<cleared>")

    # User controls
    def is_cancel_pressed(self) -> bool:
        return False

    def is_shutdown_pressed(self) -> bool:
        return False

    # Lifecycle
    def shutdown(self) -> None:
        # Nothing to clean up for the debug UI.
        pass
