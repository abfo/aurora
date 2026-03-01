import json
import logging
from pathlib import Path
from typing import Optional, Any

from settings import settings, PROJECT_DIR
from .base import Tool


class RecipeListTool(Tool):
    name = "list_recipes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recipes_dir: Path | None = None

    def _resolve_recipes_dir(self) -> Path | None:
        # 1) Use configured folder if provided
        cfg = settings.recipes_folder
        if cfg:
            p = Path(cfg)
            if not p.is_absolute():
                p = (PROJECT_DIR / p).resolve()
            if p.is_dir():
                return p
        return None

    def is_configured(self) -> bool:
        self._recipes_dir = self._resolve_recipes_dir()
        return self._recipes_dir is not None

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": "List available recipe filenames (Markdown .md) from the configured recipes folder.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None

        recipes_dir = self._recipes_dir or self._resolve_recipes_dir()
        if not recipes_dir:
            return "Recipes folder is not configured or does not exist"

        try:
            files = [p.name for p in recipes_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md"]
            files.sort(key=str.casefold)
        except Exception as err:
            self.log.error("Failed to list recipes in %s: %s", recipes_dir, err)
            return f"Failed to list recipes: {err}"

        # Return JSON array of filenames for easy parsing by the model
        self.analytics.report_event("Recipe List")    
        return json.dumps(files)


def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None, **kwargs) -> Tool:
    return RecipeListTool(log=log, audio_manager=audio_manager, **kwargs)
