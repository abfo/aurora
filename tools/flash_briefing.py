import logging
import os
import subprocess
import tempfile
import threading
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, Any

import requests

from settings import settings
from .base import Tool


# Media RSS namespace, used as a fallback when an <item> has no <enclosure>.
_MRSS_NS = "http://search.yahoo.com/mrss/"


class FlashBriefingTool(Tool):
    """Download the most recent episode of a configured podcast feed and play it
    immediately through the same audio queue that timer alarms use."""

    name = "play_flash_briefing"

    def is_configured(self) -> bool:
        return bool((settings.flash_briefing_url or "").strip())

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": (
                "Play the user's flash briefing: download the most recent episode of a "
                "configured news podcast and play it. Use for requests like 'play my flash "
                "briefing', 'play the news briefing', or 'give me my news briefing'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None
        if not self.audio_manager:
            return "Audio manager not available"

        # Do the network/ffmpeg work off the realtime event loop so we can return
        # immediately (mirrors how timer_set offloads audio generation).
        th = threading.Thread(target=self._download_and_schedule_blocking, daemon=True)
        th.start()

        self.analytics.report_event("Flash Briefing")
        return "Getting your flash briefing — it'll play in just a moment."

    # --- Background pipeline ------------------------------------------------
    def _download_and_schedule_blocking(self) -> None:
        try:
            mp3_path = os.path.join(tempfile.gettempdir(), "aurora_flash_briefing.mp3")
            wav_path = os.path.join(tempfile.gettempdir(), "aurora_flash_briefing.wav")

            episode_url, episode_title = self._fetch_latest_episode()
            if not episode_url:
                self.log.error("Flash briefing: no episode enclosure found in feed %s", settings.flash_briefing_url)
                return

            self.log.info("Flash briefing: downloading '%s' from %s", episode_title or "(untitled)", episode_url)
            self._download(episode_url, mp3_path)

            if not self._transcode(mp3_path, wav_path):
                return

            # Intermediate MP3 is no longer needed; keep only the reusable WAV.
            try:
                os.remove(mp3_path)
            except FileNotFoundError:
                pass
            except Exception:
                self.log.exception("Failed to remove intermediate flash briefing MP3: %s", mp3_path)

            self.audio_manager.add_audio(
                due=datetime.now(),
                path=wav_path,
                name="Flash Briefing",
                delete_after_play=False,  # fixed path, overwritten next time (not per-play)
            )
            self.log.info("Flash briefing scheduled for immediate playback: %s", wav_path)

        except Exception:
            self.log.exception("Error preparing flash briefing")

    def _fetch_latest_episode(self) -> tuple[Optional[str], Optional[str]]:
        """Return (audio_url, title) for the most recent feed item, or (None, None)."""
        response = requests.get(settings.flash_briefing_url, timeout=20)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        # RSS lists items newest-first, so the first <item> is the latest episode.
        item = root.find("./channel/item")
        if item is None:
            item = root.find(".//item")
        if item is None:
            return None, None

        title_el = item.find("title")
        title = title_el.text if title_el is not None else None

        enclosure = item.find("enclosure")
        if enclosure is not None and enclosure.get("url"):
            return enclosure.get("url"), title

        # Fallback: Media RSS <media:content url="...">
        media = item.find(f"{{{_MRSS_NS}}}content")
        if media is not None and media.get("url"):
            return media.get("url"), title

        return None, title

    def _download(self, url: str, dest_path: str) -> None:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    def _transcode(self, mp3_path: str, wav_path: str) -> bool:
        """Transcode MP3 to a small mono PCM WAV via ffmpeg, skipping the configured
        number of seconds from the start. Returns True on success."""
        cmd = ["ffmpeg", "-y"]
        # -ss before -i seeks on the input (fast) so the trimmed seconds are skipped.
        trim_seconds = max(0, settings.flash_briefing_trim_seconds)
        if trim_seconds:
            cmd += ["-ss", str(trim_seconds)]
        cmd += [
            "-i", mp3_path,
            "-ac", "1",
            "-ar", "24000",
            "-sample_fmt", "s16",
            wav_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except FileNotFoundError:
            self.log.error("Flash briefing: ffmpeg not installed. Install it (e.g. 'apt install ffmpeg').")
            return False
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", "replace") if e.stderr else ""
            self.log.error("Flash briefing: ffmpeg failed to transcode %s: %s", mp3_path, stderr)
            return False


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None, **kwargs) -> Tool:
    return FlashBriefingTool(log=log, audio_manager=audio_manager, **kwargs)
