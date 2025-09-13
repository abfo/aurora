import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Any

import requests

from settings import settings
from .base import Tool


class NextTransitTool(Tool):
    """Tool to fetch predicted arrival times for a configured public transit route using the Bay Area 511 API."""

    def __init__(self, log: Optional[logging.Logger] = None, audio_manager: Any | None = None):
        super().__init__(log=log, audio_manager=audio_manager)
        friendly = (settings.bay_area_511_friendly_name or "transit").strip()
        # sanitize for tool name
        sanitized = "".join(c.lower() if c.isalnum() else "_" for c in friendly).strip("_")
        self.friendly_name = friendly
        self.name = f"next_{sanitized}"

    def is_configured(self) -> bool:
        try:
            return all(
                [
                    (settings.bay_area_511_api_key or "").strip(),
                    (settings.bay_area_511_agency or "").strip(),
                    (settings.bay_area_511_stop_code or "").strip(),
                    (settings.bay_area_511_friendly_name or "").strip(),
                ]
            )
        except Exception:
            self.log.exception("Error checking NextTransitTool configuration")
            return False

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": f"Get the predicted arrival time for the next {self.friendly_name}.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None

        api_key = settings.bay_area_511_api_key
        agency = settings.bay_area_511_agency
        stop_code = settings.bay_area_511_stop_code
        line_ref = (settings.bay_area_511_friendly_name or "").split()[0]

        url = (
            "https://api.511.org/transit/StopMonitoring?"
            f"api_key={api_key}&agency={agency}&stopCode={stop_code}&Format=json"
        )
        response = None
        try:
            response = requests.get(url)
            response.raise_for_status()
            raw = response.content.decode("utf-8-sig")
            data = json.loads(raw)

            predicted_arrival_times = []
            visits = (
                data.get("ServiceDelivery", {})
                .get("StopMonitoringDelivery", {})
                .get("MonitoredStopVisit", [])
            )
            for visit in visits:
                journey = visit.get("MonitoredVehicleJourney", {})
                if journey.get("LineRef") != line_ref:
                    continue
                expected_arrival_time = journey.get("MonitoredCall", {}).get(
                    "ExpectedArrivalTime"
                )
                if not expected_arrival_time:
                    continue
                eta = datetime.fromisoformat(expected_arrival_time.replace("Z", "+00:00"))
                eta = eta.astimezone(ZoneInfo("America/Los_Angeles"))
                now = datetime.now(ZoneInfo("America/Los_Angeles"))
                minutes = (eta - now).total_seconds() / 60
                predicted_arrival_times.append(minutes)

            if not predicted_arrival_times:
                return f"No upcoming {self.friendly_name} arrivals found."

            rounded_times = [f"{round(m)}mins" for m in predicted_arrival_times]
            return ", ".join(rounded_times)
        except Exception as err:
            self.log.exception("Failed to get transit prediction")
            return f"Failed to get {self.friendly_name} prediction: {err}"
        finally:
            if response is not None:
                response.close()


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None) -> Tool:
    return NextTransitTool(log=log, audio_manager=audio_manager)
