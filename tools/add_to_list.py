import json
import logging
from typing import Optional, Any
from todoist_api_python.api import TodoistAPI
from settings import settings
from .base import Tool


class AddToListTool(Tool):
    name = "add_to_list"

    def is_configured(self) -> bool:
        # Load the plugin if we at least have an API key; project IDs can be set later.
        return bool(settings.todoist_api_key)

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": "Adds an item to the to do list or shopping list",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The name of the item to add to the list, i.e. 'bananas' or 'cut the cat's claws'",
                    },
                    "item_list": {
                        "type": "string",
                        "enum": ["todo", "shopping"],
                        "description": "The list to add to, either todo or shopping.",
                    },
                },
                "required": ["item_name", "item_list"],
            },
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None

        # Parse arguments
        try:
            args = json.loads(arguments)
        except Exception:
            return "Invalid arguments payload"

        item_name = args.get("item_name")
        item_list_raw = args.get("item_list")
        item_list = item_list_raw.lower() if isinstance(item_list_raw, str) else None   

        if not item_name:
            return "Missing required argument: item_name"
        if not item_list:
            return "Missing required argument: item_list"
        if item_list not in {"todo", "shopping"}:
            return "Invalid item_list; must be 'todo' or 'shopping'"

        # Ensure the relevant Todoist project is configured; plugin loads with only API key
        if item_list == "todo":
            project_id = settings.todoist_todo_project_id
            due_string = settings.todoist_todo_due_details
            if not project_id:
                return "Todoist 'To Do' project is not configured"
        else:  # shopping
            project_id = settings.todoist_shopping_project_id
            due_string = settings.todoist_shopping_due_details
            if not project_id:
                return "Todoist 'Shopping' project is not configured"

        api = TodoistAPI(settings.todoist_api_key)

        try:
            api.add_task(content=item_name, project_id=project_id, due_string=due_string)
        except Exception as err:
            # Try to dump server response for easier debugging
            status = None
            resp_text = None
            try:
                resp = getattr(err, "response", None)
                if resp is not None:
                    status = getattr(resp, "status_code", None)
                    # Prefer text; fall back to JSON dump if available
                    resp_text = getattr(resp, "text", None)
                    if not resp_text:
                        try:
                            resp_text = json.dumps(resp.json())
                        except Exception:
                            resp_text = None
            except Exception:
                pass

            # Log full details
            if status or resp_text:
                self.log.error("Todoist add_task failed: %s | status=%s | response=%s", err, status, resp_text)
            else:
                self.log.error("Todoist add_task failed: %s", err)

            details = f"Failed to call Todoist API: {err}"
            if status is not None:
                details += f" (status {status})"
            if resp_text:
                details += f" Response: {resp_text}"
            return details

        return f"{item_name} added to {item_list}"


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None) -> Tool:
    return AddToListTool(log=log, audio_manager=audio_manager)
