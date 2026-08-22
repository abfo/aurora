from __future__ import annotations

import logging

from .base import AssistantUIBase, AssistantUIState

try:
    import msvcrt  # Windows-only; keyboard controls are disabled without it
except ImportError:  # pragma: no cover - non-Windows debug runs
    msvcrt = None

# Arrow keys arrive from msvcrt as a two-character sequence: a prefix byte
# followed by a letter identifying the direction.
_ARROW_PREFIXES = ("\x00", "\xe0")
_ARROW_UP = "H"
_ARROW_DOWN = "P"


class DebugUI(AssistantUIBase):
    """Debug implementation of the Assistant UI.

    Logs state transitions and timer text updates via the project's logging
    infrastructure. Console keys stand in for the Braincraft joystick so the
    same flows can be exercised on Windows: up arrow starts wake-word training
    (joystick up), down arrow shuts down (joystick down), Enter cancels
    (joystick press).
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self._log = logger or logging.getLogger("aurora.ui.debug")
        # Keyboard input is edge-based, but is_*_pressed() is polled as a level
        # from several places at once (the wake-word loop, the due-audio loop,
        # alarm playback). Latch each keypress here so exactly one poller
        # consumes it, instead of racing over a single "last key" value.
        self._pending: set[str] = set()
        if msvcrt:
            self._log.info("Debug UI keys: up=train, down=shutdown, enter=cancel")
        else:
            self._log.info("Debug UI keyboard controls unavailable on this platform")

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

    # Wake-word training cues (no lights on Windows; log a clear text cue)
    def show_training_prompt(self, label: str, index: int, total: int) -> None:
        kind = "POSITIVE" if label == "positives" else "NEGATIVE"
        self._log.info("TRAINING: speak %s sample %d/%d now", kind, index, total)

    def clear_training_lights(self) -> None:
        self._log.info("TRAINING: (clip captured)")

    # User controls
    def is_train_pressed(self) -> bool:
        return self._take("train")

    def is_cancel_pressed(self) -> bool:
        return self._take("cancel")

    def is_shutdown_pressed(self) -> bool:
        return self._take("shutdown")

    # Lifecycle
    def shutdown(self) -> None:
        # Nothing to clean up for the debug UI.
        pass

    # -------------------- Keyboard polling --------------------
    def _poll_keys(self) -> None:
        """Drain anything waiting on the console into the pending latch."""
        if not msvcrt:
            return
        try:
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in _ARROW_PREFIXES:
                    # Second half of an arrow-key sequence identifies which one.
                    if not msvcrt.kbhit():
                        continue
                    code = msvcrt.getwch()
                    if code == _ARROW_UP:
                        self._pending.add("train")
                    elif code == _ARROW_DOWN:
                        self._pending.add("shutdown")
                elif ch in ("\r", "\n"):
                    self._pending.add("cancel")
        except Exception:
            self._log.exception("Keyboard poll failed")

    def _take(self, action: str) -> bool:
        """Poll the console, then consume one pending press of ``action``."""
        self._poll_keys()
        if action in self._pending:
            self._pending.discard(action)
            self._log.info("Debug UI key: %s", action)
            return True
        return False
