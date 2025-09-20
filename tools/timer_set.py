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
                        "description": "A short human-friendly name for the timer. Do not include 'timer' in the name unless that is explicitly the name.",
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

        # Immediately schedule default audio if available for instant feedback
        default_audio_file = settings.default_timer_audio_file
        if default_audio_file and os.path.exists(default_audio_file):
            self.audio_manager.add_audio(
                due=due_time, 
                path=default_audio_file, 
                name=name, 
                delete_after_play=False  # Don't delete default file
            )
            self.log.info("Scheduled default timer audio for '%s' at %s", name, due_time.isoformat())

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

            responseInput = f"""Write a funny poem to announce that a timer has gone off. The poem need to include three parts:

1. The name of the timer (which in this case is '{name}').
2. Some wild speculation about what the timer is for (2-3 things). 
3. Some increasingly wild speculation about the consequences of ignoring the timer (2-3 things). 

Your poem will be converted to speech using an OpenAI text to speech model. Please format your output for best results as a text to speech input. Just include the poem text, no other commentary. Use punctuation and line breaks to indicate pauses and intonation. Make it funny and engaging!"""

            response = client.responses.create(
                model="gpt-5",
                input=responseInput
            )

            ttsResponse = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=settings.agent_voice,
                response_format="wav",
                input=response.output_text,
                instructions="You are reciting funny poetry. Speak quickly and with emotion, with appropriate pauses for effect."
            )

            ttsResponse.write_to_file(filename)

            # Try to replace existing default audio with custom audio
            # If no existing timer is found (e.g., timer already went off), just clean up the file
            if not self.audio_manager.replace_audio(name, filename, new_delete_after_play=True):
                # Timer not found - it may have already gone off and been deleted
                # Clean up the custom audio file since it won't be used
                try:
                    os.remove(filename)
                    self.log.info("Timer '%s' not found (may have already expired), cleaned up custom audio", name)
                except Exception:
                    self.log.exception("Failed to clean up unused custom audio file: %s", filename)
            else:
                self.log.info("Replaced default audio for timer '%s' with custom audio %s", name, filename)
        except Exception:
            self.log.exception("Error generating/scheduling audio for timer '%s'", name)


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None) -> Tool:
    return TimerSetTool(log=log, audio_manager=audio_manager)
