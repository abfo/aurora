import json
import logging
from typing import Optional, Any

import audio_manager

from .base import Tool


class TimerDeleteTool(Tool):
    name = "delete_timer"

    def is_configured(self) -> bool:
        return True

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": "Delete a named timer if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the timer to delete",
                    }
                },
                "required": ["name"],
            },
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None
        if not self.audio_manager:
            return "Audio manager not available"
        
        arguments = json.loads(arguments)
        name = arguments.get("name")
        self.audio_manager.remove_audio(name.lower())
        self.analytics.report_event("Delete Timer")    
        return f'Removed timer {name}'


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None, **kwargs) -> Tool:
    return TimerDeleteTool(log=log, audio_manager=audio_manager, **kwargs)
