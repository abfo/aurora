import importlib
import inspect
import logging
import os
import pkgutil
from typing import Optional, Any


class Tool:
    """Synchronous tool interface for Realtime API function tools."""

    name: str = ""

    def __init__(self, log: Optional[logging.Logger] = None):
        self.log = (log or logging.getLogger("aurora")).getChild(self.__class__.__name__)

    def is_configured(self) -> bool:
        """Return True if the tool has valid configuration and can be enabled."""
        raise NotImplementedError

    def manifest(self) -> dict:
        """Return the tool manifest to include in the Realtime API tools array."""
        raise NotImplementedError

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        """Handle a tool call.

        Return None if this tool does not handle tool_name;
        otherwise return a string (either the tool result or an error message to speak back).
        """
        raise NotImplementedError


def load_plugins(log: Optional[logging.Logger] = None) -> list[Tool]:
    """Discover and load tools from the tools package, excluding base.py.

    Each tool module should export `create_tool(log) -> Tool`.
    """
    logger = (log or logging.getLogger("aurora")).getChild("plugins")
    tools: list[Tool] = []

    package_name = __name__.rsplit(".", 1)[0]  # 'tools'
    package = importlib.import_module(package_name)

    # Support both regular and namespace packages (which lack __file__)
    if hasattr(package, "__path__") and package.__path__ is not None:
        search_paths = list(package.__path__)
    else:
        # Fallback to directory of this file
        search_paths = [os.path.dirname(__file__)]

    for _, mod_name, ispkg in pkgutil.iter_modules(search_paths):
        if mod_name.startswith("_") or mod_name in {"base"}:
            continue
        full_name = f"{package_name}.{mod_name}"
        try:
            module = importlib.import_module(full_name)
            if hasattr(module, "create_tool") and inspect.isfunction(module.create_tool):
                tool = module.create_tool(log=logger)
                if not isinstance(tool, Tool):
                    logger.warning("Module %s create_tool did not return a Tool instance", full_name)
                    continue
                if tool.name == "":
                    logger.warning("Tool %s has empty name; skipping", full_name)
                    continue
                if tool.is_configured():
                    tools.append(tool)
                    logger.info("Loaded tool: %s", tool.name)
                else:
                    logger.info("Skipping unconfigured tool: %s", tool.name)
            else:
                logger.debug("Module %s has no create_tool(); skipping", full_name)
        except Exception as e:
            logger.exception("Failed to load tool module %s", full_name)
    # ensure unique names
    unique: dict[str, Tool] = {}
    for t in tools:
        if t.name in unique:
            logger.warning("Duplicate tool name '%s' found; keeping first, skipping others", t.name)
            continue
        unique[t.name] = t
    return list(unique.values())
