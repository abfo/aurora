import logging
from datetime import datetime, date
from typing import Optional, Any
from settings import settings
from .base import Tool


class CheeseTool(Tool):
    """Simple tool to determine whose cheese night it is today.

    Alternates daily between two kids based on the current date. Requires both
    Kid Name A and Kid Name B settings to be populated.
    """

    name = "cheese"

    def is_configured(self) -> bool:
        try:
            a = (settings.kid_name_a or "").strip()
            b = (settings.kid_name_b or "").strip()
            if not a or not b:
                self.log.info("CheeseTool disabled: kid names not configured (KID_NAME_A/KID_NAME_B)")
            return bool(a and b)
        except Exception:
            self.log.exception("Error checking CheeseTool configuration")
            return False

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": (
                "Get's the kid whose cheese night it is today. This kid gets to be the first to take cheese at dinner, and they also get to choose which chore to do, like feeding the dog instead of the cat, so it's an advantage to have first cheese for the day."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None

        a = (settings.kid_name_a or "").strip()
        b = (settings.kid_name_b or "").strip()
        if not a or not b:
            return "Kid names are not configured. Please set KID_NAME_A and KID_NAME_B."

        unixoffset = datetime.now() - datetime(1970, 1, 1)
        kid = a if (unixoffset.days % 2 == 0) else b
        return kid


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None) -> Tool:
    # audio_manager is unused for this tool
    return CheeseTool(log=log, audio_manager=audio_manager)
