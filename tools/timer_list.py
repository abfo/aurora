import logging
from typing import Optional, Any

import audio_manager

from .base import Tool


class TimerListTool(Tool):
    name = "list_timers"

    def is_configured(self) -> bool:
        return True

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": "List the currently scheduled timers in a human-friendly format.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None
        if not self.audio_manager:
            return "Audio manager not available"

        self.analytics.report_event("List Timers")    
        return self.audio_manager.list_audio()

def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None, **kwargs) -> Tool:
    return TimerListTool(log=log, audio_manager=audio_manager, **kwargs)
