import json
import logging
from typing import Optional, Any
import requests
from settings import settings
from .base import Tool


class ControlLightTool(Tool):
    """Tool for controlling LIFX smart light bulbs via the LIFX HTTP API."""

    name = "control_light"

    def is_configured(self) -> bool:
        """Return True if LIFX is properly configured with auth token and at least one light."""
        try:
            if not settings.lifx_auth_token.strip():
                self.log.info("ControlLightTool disabled: LIFX auth token not configured (LIFX_AUTH_TOKEN)")
                return False
            
            # Parse the lights configuration
            lights_config = json.loads(settings.lifx_lights or "{}")
            if not lights_config:
                self.log.info("ControlLightTool disabled: no lights configured (LIFX_LIGHTS)")
                return False
                
            return True
        except json.JSONDecodeError:
            self.log.error("ControlLightTool disabled: invalid LIFX_LIGHTS JSON format")
            return False
        except Exception:
            self.log.exception("Error checking ControlLightTool configuration")
            return False

    def manifest(self) -> dict:
        """Return the tool manifest for the OpenAI Realtime API."""
        return {
            "name": self.name,
            "type": "function",
            "description": "Turns a light on or off by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "light_name": {
                        "type": "string",
                        "description": "The name of a light, i.e. cupboard, hallway, living room"
                    },
                    "light_state": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "The desired state of the light, on or off."
                    }
                },
                "required": ["light_name", "light_state"]
            }
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        """Handle the control_light tool call."""
        if tool_name != self.name:
            return None

        # Parse arguments
        try:
            args = json.loads(arguments)
        except Exception:
            return "Invalid arguments payload"

        light_name = args.get("light_name")
        light_state = args.get("light_state")

        if not light_name:
            return "Missing required argument: light_name"
        if not light_state:
            return "Missing required argument: light_state"
        if light_state.lower() not in {"on", "off"}:
            return "Invalid light_state; must be 'on' or 'off'"

        # Parse lights configuration and find the selector
        try:
            lights_config = json.loads(settings.lifx_lights or "{}")
        except json.JSONDecodeError:
            return "LIFX lights configuration is invalid JSON"

        selector = None
        light_name_lower = light_name.lower()
        
        # Look for exact match first, then partial matches
        for configured_name, configured_selector in lights_config.items():
            if configured_name.lower() == light_name_lower:
                selector = configured_selector
                break
        
        # If no exact match, try partial matching
        if selector is None:
            for configured_name, configured_selector in lights_config.items():
                if light_name_lower in configured_name.lower() or configured_name.lower() in light_name_lower:
                    selector = configured_selector
                    break

        if selector is None:
            configured_lights = ", ".join(lights_config.keys())
            return f"Light name '{light_name}' not recognized. Available lights: {configured_lights}"

        # Prepare API request
        auth_token = settings.lifx_auth_token.strip()
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }

        # Prepare request data based on desired state
        if light_state.lower() == 'on':
            data = {
                "power": "on",
                "color": "kelvin:3500",
                "brightness": 1.0,
                "duration": 2.0,
                "fast": False
            }
        else:
            data = {
                "power": "off",
                "duration": 2.0,
                "fast": False
            }

        # Make API request
        response = None
        try:
            request_url = f'https://api.lifx.com/v1/lights/{selector}/state'
            response = requests.put(request_url, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            
            # Check if all lights in the response succeeded
            for light in response_json.get("results", []):
                if light.get("status") != "ok":
                    return f'Failed to set light {light_name} to {light_state}'
                    
        except requests.exceptions.Timeout:
            return f'Request to LIFX API timed out for {light_name}'
        except requests.exceptions.RequestException as e:
            return f'Failed to send request to LIFX API for {light_name}: {e}'
        except json.JSONDecodeError:
            return f'Invalid response from LIFX API for {light_name}'
        except Exception as e:
            return f'Unexpected error controlling {light_name}: {e}'
        finally:
            if response is not None:
                response.close()

        return f'{light_name} is now {light_state}'


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None) -> Tool:
    return ControlLightTool(log=log, audio_manager=audio_manager)