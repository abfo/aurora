"""GPT-Live session transport for Aurora.

Aurora talks to GPT-Live over a single WebSocket: mic audio goes up as
``session.input_audio.append``, speech comes back as ``session.output_audio.delta``,
and JSON control events travel both ways on the same socket.

The voice model only listens and speaks. Every task is handled by Aurora's own
backend agent (see backend_agent.py), so the session is opened in client
delegation mode and carries no tool definitions at all - GPT-Live tells us it
wants something done with ``session.delegation.created`` and we answer with
``session.commentary.append``.

Everything protocol-shaped lives here: opening the session and waiting for
``session.started``, the send helpers (which enforce the 500-token cap on
appended content), and the graceful close that waits for ``session.closed`` so
the final usage is logged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import AsyncIterator, Optional

import websockets

from settings import settings

_LIVE_URL = "wss://api.openai.com/v1/live/sessions"

# GPT-Live uses one audio format for both directions, fixed at session.start.
# 24 kHz mono signed 16-bit little-endian is exactly what PyAudio already
# captures and plays, so nothing in Aurora has to resample.
LIVE_SAMPLERATE = 24000

_WS_PING_INTERVAL_SECONDS = 30
_WS_PING_TIMEOUT_SECONDS = 25

_SESSION_START_TIMEOUT_SECONDS = 15
_SESSION_CLOSE_TIMEOUT_SECONDS = 15

# Appended content is capped at 500 tokens per event. Counting tokens here would
# mean another dependency for no real gain, so clip on a conservative character
# budget instead - roughly 400 tokens of English.
_MAX_CONTENT_CHARS = 1600


def clip_content(text: str) -> str:
    """Trim appended content to fit GPT-Live's per-event limit.

    Cuts at a word boundary where there is one nearby, so Aurora never has to
    read half a word out loud.
    """
    text = (text or "").strip()
    if len(text) <= _MAX_CONTENT_CHARS:
        return text
    clipped = text[:_MAX_CONTENT_CHARS]
    cut = clipped.rfind(" ")
    if cut > _MAX_CONTENT_CHARS - 200:
        clipped = clipped[:cut]
    return clipped.rstrip() + "..."


def _event_id() -> str:
    """Fresh client_event_id, used to match acknowledgments and errors."""
    return str(uuid.uuid4())


class LiveSession:
    """An open GPT-Live session.

    The caller owns the receive loop (iterate :meth:`events`); this class only
    sends, and knows how to shut the session down cleanly. Send failures are
    raised, except for audio appends, which are dropped silently - a session
    that is going away shouldn't take the mic plumbing down with it.
    """

    def __init__(self, ws: websockets.ClientConnection, session_id: str, log: logging.Logger):
        self._ws = ws
        self._log = log
        self.session_id = session_id
        # True once we have asked to close, so teardown doesn't ask twice.
        self.closing = False
        # True once session.closed has been seen, meaning usage is final.
        self.closed = False
        # PCM must be appended in whole 16-bit samples. PyAudio hands us even
        # buffers, but carry any odd byte into the next append rather than
        # trusting that.
        self._pending = b""

    # -------------------- Receiving --------------------
    async def events(self) -> AsyncIterator[dict]:
        """Yield parsed server events until the socket ends."""
        async for message in self._ws:
            event = self._parse(message)
            if event is None:
                continue
            if event.get("type") == "session.closed":
                self.closed = True
            yield event

    def _parse(self, message) -> Optional[dict]:
        if isinstance(message, bytes):
            # GPT-Live sends audio inside JSON events; binary frames are not
            # part of the protocol, so anything here is unexpected.
            self._log.debug("Ignoring unexpected binary frame (%d bytes)", len(message))
            return None
        try:
            return json.loads(message)
        except Exception:
            self._log.warning("Could not parse server event: %.120s", message)
            return None

    # -------------------- Sending --------------------
    async def _send(self, event: dict) -> None:
        await self._ws.send(json.dumps(event))

    async def append_audio(self, frame: bytes) -> None:
        """Append a mic frame. Never raises; a dead session just drops audio."""
        if self.closing:
            return
        chunk = self._pending + frame
        complete = len(chunk) - len(chunk) % 2
        self._pending = chunk[complete:]
        if not complete:
            return
        try:
            await self._send({
                "type": "session.input_audio.append",
                "audio": base64.b64encode(chunk[:complete]).decode("utf-8"),
            })
        except Exception:
            pass

    async def append_commentary(self, content: str, delegation_id: str) -> None:
        """Hand GPT-Live a verified result to say out loud."""
        await self._send({
            "type": "session.commentary.append",
            "event_id": _event_id(),
            "delegation_id": delegation_id,
            "content": clip_content(content),
        })

    async def append_thinking(self, content: str, delegation_id: str) -> None:
        """Report background progress. Not spoken, but informs later answers."""
        await self._send({
            "type": "session.thinking.append",
            "event_id": _event_id(),
            "delegation_id": delegation_id,
            "content": clip_content(content),
        })

    async def append_instructions(self, content: str, delegation_id: Optional[str] = None) -> None:
        """Steer the live conversation. Can interrupt speech in progress."""
        await self._send({
            "type": "session.instructions.append",
            "event_id": _event_id(),
            "delegation_id": delegation_id,
            "content": clip_content(content),
        })

    # -------------------- Shutting down --------------------
    async def request_close(self) -> None:
        """Ask GPT-Live to end the session.

        Queued backend work is cancelled and further commands are rejected, but
        events keep arriving until ``session.closed``, so the caller should go on
        reading (or call :meth:`drain_until_closed`).
        """
        if self.closing or self.closed:
            return
        self.closing = True
        try:
            await self._send({"type": "session.close", "event_id": _event_id()})
        except Exception:
            self._log.debug("Session already gone when requesting close")

    async def drain_until_closed(self) -> None:
        """Read remaining events until ``session.closed``, and log final usage.

        Only safe once the caller's own receive loop has stopped: two consumers
        on one socket would race for events.
        """
        if self.closed:
            return
        try:
            await asyncio.wait_for(self._drain(), timeout=_SESSION_CLOSE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            self._log.warning("No session.closed within %ds; final usage unconfirmed",
                              _SESSION_CLOSE_TIMEOUT_SECONDS)
        except Exception as e:
            self._log.debug("Session ended without session.closed: %s", e)

    async def _drain(self) -> None:
        async for event in self.events():
            if event.get("type") == "session.closed":
                self._log.info("Session closed: reason=%s usage=%s",
                               event.get("reason"), (event.get("usage") or {}).get("seconds"))
                return
            if event.get("type") == "error":
                self._log.warning("GPT-Live error while closing: %s", event.get("error"))

    async def disconnect(self) -> None:
        """Release the socket. Call after :meth:`drain_until_closed`."""
        try:
            await self._ws.close()
        except Exception:
            pass


async def connect_live_session(instructions: str, log: logging.Logger) -> LiveSession:
    """Open a GPT-Live session and wait until it is ready for audio.

    ``instructions`` is the live prompt - persona, speaking style and the
    delegation policy. It is fixed for the lifetime of the session (as are the
    model, audio format and voice), which is why Aurora reconnects rather than
    reconfigures when it switches into cooking or training mode.
    """
    ws = await websockets.connect(
        _LIVE_URL,
        additional_headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        ping_interval=_WS_PING_INTERVAL_SECONDS,
        ping_timeout=_WS_PING_TIMEOUT_SECONDS,
    )

    start_event = {
        "type": "session.start",
        "event_id": _event_id(),
        "session": {
            "model": settings.live_model,
            "instructions": instructions,
            "audio": {
                "format": {"type": "audio/pcm", "rate": LIVE_SAMPLERATE},
                "output": {"voice": settings.agent_voice},
            },
            # Client delegation: Aurora's own backend agent does the work, so
            # no tools are declared here. See backend_agent.py.
            "delegation": {"type": "client"},
        },
    }

    try:
        await ws.send(json.dumps(start_event))
        session_id = await asyncio.wait_for(
            _await_started(ws, log), timeout=_SESSION_START_TIMEOUT_SECONDS
        )
    except Exception:
        await ws.close()
        raise

    log.info("GPT-Live session started: %s (model=%s, voice=%s)",
             session_id, settings.live_model, settings.agent_voice)
    return LiveSession(ws, session_id, log)


async def _await_started(ws: websockets.ClientConnection, log: logging.Logger) -> str:
    """Wait for ``session.started`` and return the session id."""
    async for message in ws:
        try:
            event = json.loads(message)
        except Exception:
            continue
        kind = event.get("type")
        if kind == "session.started":
            return ((event.get("session") or {}).get("id")) or ""
        if kind == "error":
            raise RuntimeError(f"GPT-Live rejected session.start: {event.get('error')}")
        log.debug("Pre-start event: %s", kind)
    raise RuntimeError("Connection closed before session.started")
