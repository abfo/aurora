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
_ARROW_DOWN = "P"


class DebugUI(AssistantUIBase):
    """Debug implementation of the Assistant UI.

    Logs state transitions and timer text updates via the project's logging
    infrastructure. Console keys stand in for the Braincraft controls so the
    same flows can be exercised on Windows: Enter is the joystick press, which
    cancels during playback and starts wake-word training while idle, and down
    arrow shuts down (joystick down).
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
            self._log.info("Debug UI keys: enter=cancel/train, down=shutdown")
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

    # User controls. Enter is one button with two jobs, mirroring the joystick
    # press on the Braincraft: cancel during playback, start training while
    # idle. Only one of the two is ever polled at a time, so they can share it.
    def is_train_pressed(self) -> bool:
        return self._take("button", "train")

    def is_cancel_pressed(self) -> bool:
        return self._take("button", "cancel")

    def is_shutdown_pressed(self) -> bool:
        return self._take("shutdown", "shutdown")

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
                    if msvcrt.getwch() == _ARROW_DOWN:
                        self._pending.add("shutdown")
                elif ch in ("\r", "\n"):
                    self._pending.add("button")
        except Exception:
            self._log.exception("Keyboard poll failed")

    def _take(self, key: str, action: str) -> bool:
        """Poll the console, then consume one pending press of ``key``.

        ``action`` only labels the log line - it says what the press means in
        the current phase, since one key can drive more than one action.
        """
        self._poll_keys()
        if key in self._pending:
            self._pending.discard(key)
            self._log.info("Debug UI key: %s", action)
            return True
        return False
