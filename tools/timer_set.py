import json
import logging
import asyncio
import threading
import os
import uuid
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Any
from openai import OpenAI
from settings import settings
from .base import Tool


class TimerSetTool(Tool):
    name = "set_timer"

    def is_configured(self) -> bool:
        return True

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": "Set a named timer that is due after a number of seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A short human-friendly name for the timer",
                    },
                    "due_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of seconds from now when the timer should be due",
                    },
                },
                "required": ["name", "due_seconds"],
            },
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None
        if not self.audio_manager:
            return "Audio manager not available"
        
        arguments = json.loads(arguments)
        name = arguments.get("name")
        due_seconds = int(arguments.get("due_seconds"))
        due_time = datetime.now() + timedelta(seconds=max(0, due_seconds))

        # Kick off background generation/scheduling so we can return immediately
        try:
            self._start_async_audio_generation(name, due_time)
        except Exception as e:
            self.log.exception("Failed to start background audio generation for timer '%s'", name)
            return f"Failed to set timer {name}: {e}"

        return f'Set a {name} timer for {due_seconds} seconds.'

    # --- Async scaffolding -------------------------------------------------
    def _start_async_audio_generation(self, name: str, due_time: datetime) -> None:
        th = threading.Thread(
            target=self._generate_and_schedule_blocking,
            args=(name, due_time),
            daemon=True,
        )
        th.start()

    def _get_random_filename(self, directory, extension):
        filename = str(uuid.uuid4()) + '.' + extension
        filepath = os.path.join(directory, filename)
        while os.path.exists(filepath):
            filename = str(uuid.uuid4()) + '.' + extension
            filepath = os.path.join(directory, filename)
        return filepath

    def _generate_and_schedule_blocking(self, name: str, due_time: datetime) -> None:
        # Run the async coroutine in a dedicated event loop within this thread
        asyncio.run(self._generate_audio_and_schedule(name, due_time))

    async def _generate_audio_and_schedule(self, name: str, due_time: datetime) -> None:
        try:
            # Placeholder for async TTS/audio creation work
            await asyncio.sleep(0)
            
            client = OpenAI(api_key=settings.openai_api_key)
            filename = self._get_random_filename(tempfile.gettempdir(), "wav")

            ttsResponse = client.audio.speech.create(
                model="tts-1-hd",
                voice=settings.agent_voice,
                response_format="wav",
                input=f'beep beep beep beep this is your {name} timer beep beep beep beep this is your {name} timer beep beep beep beep it\'s still alarming beep beep beep beep push my button to switch this off BEEP BEEP BEEP BEEP last {name} timer warning BEEP BEEP BEEP BEEP BEEP BEEP BEEP BEEP ok, that\'s it, I tried by best - hope the {name} timer wasn\'t important'
            )

            ttsResponse.write_to_file(filename)

            self.audio_manager.add_audio(due=due_time, path=filename, name=name, delete_after_play=True)
            self.log.info("Scheduled timer '%s' at %s with audio %s", name, due_time.isoformat(), filename)
        except Exception:
            self.log.exception("Error generating/scheduling audio for timer '%s'", name)


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None) -> Tool:
    return TimerSetTool(log=log, audio_manager=audio_manager)
