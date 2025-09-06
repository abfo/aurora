from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from threading import RLock
from typing import List, Optional
import logging


@dataclass(order=True)
class ScheduledAudio:
    """A single audio item scheduled to play at a specific time.

    Attributes:
        due: When the audio should be played (local time).
        path: Full path to the audio file to play.
        name: Human-friendly name for the scheduled audio (used for listing/removing).
        delete_after_play: If True, caller may delete the file after playing (used by remove logic too).
    """

    due: datetime
    path: str
    name: str
    delete_after_play: bool = True


class AudioManager:
    """Manages a schedule of audio items with add/query/remove helpers.

    Methods mirror the previous functional API:
      - add_audio(due, path, name, delete_after_play=True)
      - has_due_audio(now=None)
      - has_any_audio()
      - get_audio(now=None) -> Optional[ScheduledAudio]
      - audio_to_text(now=None) -> str
      - list_audio() -> str (JSON)
      - remove_audio(name) -> None
    """

    def __init__(self, log: Optional[logging.Logger] = None) -> None:
        self._items: List[ScheduledAudio] = []
        self._lock = RLock()
        self._log = (log or logging.getLogger("aurora")).getChild("audio")

    # -- Mutations ---------------------------------------------------------
    def add_audio(self, due: datetime, path: str, name: str, delete_after_play: bool = True) -> None:
        """Add a new audio item and keep the list sorted by due time ascending."""
        with self._lock:
            self._items.append(ScheduledAudio(due=due, path=path, name=name, delete_after_play=delete_after_play))
            # sort by due ascending (soonest first)
            self._items.sort(key=lambda x: x.due)

    def remove_audio(self, name: str) -> None:
        """Remove all audio items with the given name; delete files for those marked delete_after_play."""
        with self._lock:
            to_remove = [it for it in self._items if it.name == name]
            for it in to_remove:
                try:
                    self._items.remove(it)
                except ValueError:
                    pass
                if it.delete_after_play:
                    try:
                        os.remove(it.path)
                    except FileNotFoundError:
                        self._log.debug("Audio file already removed: %s", it.path)
                    except Exception:
                        self._log.exception("Failed to remove audio file: %s", it.path)

    # -- Queries -----------------------------------------------------------
    def has_due_audio(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        with self._lock:
            return any(it.due <= now for it in self._items)

    def has_any_audio(self) -> bool:
        with self._lock:
            return len(self._items) > 0

    def get_audio(self, now: Optional[datetime] = None) -> Optional[ScheduledAudio]:
        """Pop and return the first due audio item, or None if none are due."""
        now = now or datetime.now()
        with self._lock:
            for it in list(self._items):
                if it.due <= now:
                    self._items.remove(it)
                    return it
        return None

    # -- Presentation ------------------------------------------------------
    def audio_to_text(self, now: Optional[datetime] = None) -> str:
        """Return a human-readable summary of timers with countdown mm:ss."""
        now = now or datetime.now()
        lines: list[str] = []
        with self._lock:
            for it in self._items:
                lines.append(f"{it.name} timer:\n")
                delta_seconds = (it.due - now).total_seconds()
                if delta_seconds > 0:
                    minutes = int(delta_seconds // 60)
                    seconds = int(delta_seconds % 60)
                    lines.append(f"{minutes:02}:{seconds:02}\n\n")
                else:
                    lines.append("00:00\n\n")
        return "".join(lines)

    def list_audio(self) -> str:
        """Return a JSON array of scheduled items with Name and Due fields."""
        with self._lock:
            tasks = [
                {"Name": it.name, "Due": it.due.strftime("%A, %B %d, %Y %I:%M:%S %p")}
                for it in self._items
            ]
        return json.dumps(tasks)

    # -- Utilities ---------------------------------------------------------
    def clear(self) -> None:
        """Remove all scheduled items without deleting files."""
        with self._lock:
            self._items.clear()
