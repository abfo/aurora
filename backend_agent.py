"""Aurora's backend agent - the client side of GPT-Live delegation.

GPT-Live only listens and speaks. When it decides a request needs work done it
emits ``session.delegation.created``, and that event deliberately carries no
request text and no arguments: the application is expected to know what was
asked. So Aurora keeps a rolling transcript (see :class:`TranscriptBuffer`,
filled from the transcript delta events) and builds the backend request from it.

:class:`BackendAgent` then does what the Realtime voice model used to do itself -
pick a tool, run it, decide whether another is needed - using a normal Responses
call with the same tool manifests the plugins have always exposed. The result
goes back to GPT-Live as ``session.commentary.append``, which it paraphrases in
Aurora's voice.

Tools that change what Aurora is *doing* rather than answering a question
(sleep, cooking, wake-word training) are handled here too, but instead of being
spoken they are returned to main.py as an action for the conversation loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from analytics import Analytics
from live_session import LiveSession
from settings import settings
from tools.base import Tool

# How many tool rounds one delegation may take. Two is enough for the longest
# real chain Aurora has (list_recipes then start_cooking); the third is slack.
_MAX_TOOL_ROUNDS = 3

# How much transcript to hand the backend. Enough to interpret "the second one"
# or "make it five minutes instead", without turning every delegation into a
# long prompt.
_TRANSCRIPT_CHARS = 1200

_BACKEND_TIMEOUT_SECONDS = 30.0

# Session modes. The live prompt and the available tools both change with the
# mode, and because the live prompt is fixed at session.start, switching mode
# means reconnecting (see main.py).
MODE_NORMAL = "normal"
MODE_COOKING = "cooking"
MODE_TRAINING = "training"

_SLEEP_TOOL = {
    "name": "go_to_sleep",
    "type": "function",
    "description": "Ends the current session and puts the assistant to sleep.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_START_TRAINING_TOOL = {
    "name": "start_wake_word_training",
    "type": "function",
    "description": (
        "Start wake-word training. Call this when the user asks to help train or "
        "teach Aurora to recognize their voice or the wake word (e.g. 'help me train "
        "you to recognize my voice', 'train your wake word')."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_BEGIN_CAPTURE_TOOL = {
    "name": "begin_wake_word_capture",
    "type": "function",
    "description": (
        "Begin recording the wake-word training clips. Call this as soon as the "
        "user has given their name and the rules have been explained."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The first name of the person doing the training.",
            }
        },
        "required": ["name"],
    },
}

_START_COOKING_TOOL = {
    "name": "start_cooking",
    "type": "function",
    "description": "Starts a cooking session with the specified recipe. Use the list_recipes tool to see available recipes.",
    "parameters": {
        "type": "object",
        "properties": {
            "recipe_name": {
                "type": "string",
                "description": "The filename of the recipe to start (including .md extension).",
            }
        },
        "required": ["recipe_name"],
    },
}

_STOP_COOKING_TOOL = {
    "name": "stop_cooking",
    "type": "function",
    "description": "Stops the current cooking session.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class TranscriptBuffer:
    """Rolling record of who said what, built from transcript delta events.

    Deltas are fragments, not turns, so consecutive fragments from the same
    speaker are concatenated and only a speaker change starts a new turn. Old
    turns fall off the front once the buffer grows past ``max_chars``.
    """

    def __init__(self, max_chars: int = _TRANSCRIPT_CHARS):
        self._turns: list[list[str]] = []  # [role, text] pairs, oldest first
        self._max_chars = max_chars

    def add_user(self, delta: str) -> None:
        self._add("User", delta)

    def add_assistant(self, delta: str) -> None:
        self._add("Aurora", delta)

    def _add(self, role: str, delta: str) -> None:
        if not delta:
            return
        if self._turns and self._turns[-1][0] == role:
            self._turns[-1][1] += delta
        else:
            self._turns.append([role, delta])
        self._trim()

    def _trim(self) -> None:
        while len(self._turns) > 1 and sum(len(t[1]) for t in self._turns) > self._max_chars:
            self._turns.pop(0)

    def recent(self) -> str:
        """The conversation so far, as plain labelled lines."""
        return "\n".join(f"{role}: {text.strip()}" for role, text in self._turns if text.strip())

    def has_user_speech(self) -> bool:
        return any(role == "User" and text.strip() for role, text in self._turns)

    def clear(self) -> None:
        self._turns.clear()


class BackendAgent:
    """Runs delegated tasks with Aurora's tools and reports back to GPT-Live."""

    def __init__(
        self,
        instructions: str,
        tools: list[Tool],
        log: logging.Logger,
        analytics: Analytics,
        mode: str = MODE_NORMAL,
    ):
        self._instructions = instructions
        self._tools = tools
        self._log = log.getChild("backend")
        self._analytics = analytics
        self._mode = mode
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=_BACKEND_TIMEOUT_SECONDS)

    # -------------------- Tool set --------------------
    def tool_schemas(self) -> list[dict]:
        """The tools available to the backend in the current mode.

        Training mode is deliberately tiny: the guided flow only needs to take a
        name and start recording, and anything else on offer invites the model
        to wander off script.
        """
        if self._mode == MODE_TRAINING:
            return [_BEGIN_CAPTURE_TOOL, _SLEEP_TOOL]

        schemas = [tool.manifest() for tool in self._tools]
        schemas.append(_SLEEP_TOOL)

        if settings.wake_word_training_enabled:
            schemas.append(_START_TRAINING_TOOL)

        if self._mode == MODE_COOKING:
            schemas.append(_STOP_COOKING_TOOL)
        elif self._recipe_tool() is not None:
            schemas.append(_START_COOKING_TOOL)

        return schemas

    def _recipe_tool(self) -> Optional[Tool]:
        tool = next((t for t in self._tools if t.name == "list_recipes"), None)
        return tool if tool and tool.is_configured() else None

    # -------------------- Delegation --------------------
    async def handle(self, session: LiveSession, delegation_id: str, transcript: str) -> Optional[dict]:
        """Run one delegation to completion.

        Sends the spoken result back as commentary and returns a conversation
        action for main.py (sleep, cook, train...), or None if the task was
        purely informational.
        """
        task = (
            "This is the conversation so far between the user and Aurora, whose voice you are "
            "working behind. Handle the user's most recent request.\n\n"
            f"{transcript}"
        )
        conversation: list[dict] = [{"role": "user", "content": task}]
        action: Optional[dict] = None
        spoken = ""

        for round_number in range(_MAX_TOOL_ROUNDS):
            try:
                response = await asyncio.to_thread(
                    self._client.responses.create,
                    model=settings.backend_model,
                    instructions=self._instructions,
                    input=conversation,
                    tools=self.tool_schemas(),
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    store=False,
                )
            except Exception as e:
                self._log.exception("Backend model call failed: %s", e)
                await self._say(session, delegation_id, "That did not work - the request failed.")
                return action

            calls, text = _split_output(response)
            if text:
                spoken = text

            if not calls:
                break

            for call in calls:
                conversation.append({
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                })
                output, call_action = await self._run_call(session, delegation_id, call)
                conversation.append({
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": output,
                })
                if call_action:
                    # Later calls can still run, but the first action wins: a
                    # mode change and a second opinion about it would conflict.
                    action = action or call_action

            if round_number == _MAX_TOOL_ROUNDS - 1:
                self._log.warning("Delegation %s hit the tool round limit", delegation_id)

        # An action with nothing to say is normal (a bare go_to_sleep), and so is
        # a task whose whole result is the action (cooking mode). Only speak when
        # the backend actually produced words.
        if spoken:
            await self._say(session, delegation_id, spoken)
        elif action is None:
            self._log.info("Delegation %s produced no result", delegation_id)
            await self._say(session, delegation_id, "Nothing came back for that one.")

        return action

    async def _say(self, session: LiveSession, delegation_id: str, content: str) -> None:
        try:
            await session.append_commentary(content, delegation_id)
        except Exception:
            self._log.exception("Could not send commentary for delegation %s", delegation_id)

    async def _think(self, session: LiveSession, delegation_id: str, content: str) -> None:
        try:
            await session.append_thinking(content, delegation_id)
        except Exception:
            self._log.debug("Could not send thinking for delegation %s", delegation_id)

    # -------------------- Tool execution --------------------
    async def _run_call(self, session: LiveSession, delegation_id: str, call: dict) -> tuple[str, Optional[dict]]:
        """Run one tool call. Returns (output for the model, action for main.py)."""
        name = call["name"]
        arguments = call["arguments"]
        self._log.info("Backend tool call: %s %s", name, arguments)

        # Tell GPT-Live what is happening so it can hold the conversation while
        # a slow lookup runs, instead of waiting in silence.
        await self._think(session, delegation_id, f"Running {name} for the user's request.")

        if name == "go_to_sleep":
            self._analytics.report_event("Sleep")
            return "Ending the session.", {"action": "sleep"}

        if name == "start_wake_word_training":
            self._analytics.report_event("WakeWordTraining")
            return "Entering wake word training mode.", {"action": "train"}

        if name == "begin_wake_word_capture":
            speaker = _argument(arguments, "name") or ""
            return "Starting the recording.", {"action": "capture", "name": speaker}

        if name == "start_cooking":
            return await asyncio.to_thread(self._start_cooking, arguments)

        if name == "stop_cooking":
            return "Stopped the cooking session.", {"action": "stop_cooking"}

        output = await asyncio.to_thread(self._run_plugin, name, arguments)
        return output, None

    def _run_plugin(self, name: str, arguments: str) -> str:
        """Dispatch to the first plugin that claims the call (blocking)."""
        failure = None
        for tool in self._tools:
            try:
                output = tool.handle(name, arguments)
                if output:
                    self._log.info("Tool %s handled function call %s", tool.name, name)
                    return output
            except Exception as e:
                # Keep looking - handle() is supposed to return None for calls a
                # tool does not own, but a raise should not hide the tool that
                # does own it. Report the failure only if nothing else answers.
                self._log.exception("Error in tool %s: %s", tool.name, e)
                failure = f"The {tool.name} tool failed: {e}"
        if failure:
            return failure
        self._log.warning("No tool handled function call %s", name)
        return f"No tool is available for {name}."

    def _start_cooking(self, arguments: str) -> tuple[str, Optional[dict]]:
        """Validate and load a recipe so the session can reopen in cooking mode."""
        recipe_tool = self._recipe_tool()
        if recipe_tool is None:
            return "Recipe tool is not available.", None

        recipe_name = _argument(arguments, "recipe_name")
        if not recipe_name:
            return "Missing required argument: recipe_name", None

        recipes_dir = recipe_tool._recipes_dir or recipe_tool._resolve_recipes_dir()
        if not recipes_dir:
            return "Recipes folder is not configured or does not exist", None

        recipe_path = (Path(recipes_dir) / recipe_name).resolve()
        if not recipe_path.is_file() or recipe_path.suffix.lower() != ".md":
            return f"Recipe file not found: {recipe_name}", None

        try:
            recipe = recipe_path.read_text(encoding="utf-8")
        except Exception as e:
            self._log.exception("Failed to read recipe file %s: %s", recipe_name, e)
            return f"Failed to read recipe file {recipe_name}: {e}", None

        self._analytics.report_event("Cooking")
        self._log.info("Started cooking session with recipe %s", recipe_name)
        return (
            f"Started cooking {recipe_name}.",
            {"action": "cook", "recipe": recipe, "name": recipe_name},
        )


def _split_output(response: Any) -> tuple[list[dict], str]:
    """Pull function calls and assistant text out of a Responses result."""
    calls: list[dict] = []
    text = ""

    for item in getattr(response, "output", None) or []:
        kind = getattr(item, "type", None)
        if kind == "function_call":
            calls.append({
                "call_id": getattr(item, "call_id", "") or "",
                "name": getattr(item, "name", "") or "",
                "arguments": getattr(item, "arguments", "") or "{}",
            })
        elif kind == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "output_text":
                    text += getattr(part, "text", "") or ""

    return calls, text.strip()


def _argument(arguments: Any, key: str) -> Optional[str]:
    """Read one argument out of a tool call's JSON payload, forgivingly."""
    if isinstance(arguments, dict):
        value = arguments.get(key)
    else:
        try:
            value = (json.loads(arguments or "{}") or {}).get(key)
        except Exception:
            return None
    if value is None:
        return None
    return str(value).strip() or None
