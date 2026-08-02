"""Plain-REST surface over the :class:`Runner`.

Two routes let a generic HTTP client (no A2A SDK, no AgentCard) drive a
served agent:

* ``POST /run`` runs the agent to completion and returns the result as
  JSON.
* ``POST /run_sse`` runs the agent and streams its run items as
  Server-Sent Events, terminated by a ``result`` event carrying the
  final summary.

Request body (JSON)::

    {
        "prompt": "...",  # required, non-empty string
        "max_turns": 10,  # optional; framework default when omitted
        "session": {  # optional; used only when a session
            "user_id": "...",  #   factory is wired into the app
            "session_id": "...",
        },
    }

Responses serialize Layer-1 / Layer-3 types only (see
:mod:`troopai.adk.serving.serializers`); the provider wire format
never crosses this boundary.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse
from starlette.routing import Route

from troopai.adk.run.runner import Runner
from troopai.adk.serving.serializers import (
    run_result_to_dict,
    stream_event_to_dict,
    streaming_result_to_dict,
)

if TYPE_CHECKING:
    from starlette.requests import Request

    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.types.session.store import SessionStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_BYTES = 1024 * 1024
"""Cost-conservative default cap (1 MiB) on the request body read (R3).

Bounds how many bytes a single ``POST /run`` or ``POST /run_sse`` request
may stream before the surface returns ``413 Payload Too Large``. Pass
``max_body_bytes=0`` to :func:`rest_routes` to disable the cap.
"""

SessionFactory = Callable[[str, str], Awaitable["SessionStore"]]
"""Builds a per-request session store from ``(user_id, session_id)``.

Async so it can wrap backends whose session lookup is a coroutine (e.g.
``SQLiteMultiSessions.get_or_create``). The app-name and storage backend
are bound by the caller that supplies the factory (e.g. the ``serve``
command), so the REST layer never hard-codes a session implementation.
"""


class _BadRequest(Exception):
    """Internal signal that a request failed validation or exceeded a limit.

    Carries the HTTP ``status_code`` the endpoint should return — 400 for a
    malformed body (the default), 413 when the body exceeds the size cap.
    """

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def _read_json_body(request: Request, max_body_bytes: int) -> Any:
    """Read the request body under a size cap and parse it as JSON.

    Reads the body incrementally so an over-large payload is rejected
    before it is fully buffered, rather than being read whole into memory.

    Args:
        request: The incoming HTTP request.
        max_body_bytes: Maximum bytes to accept. A declared
            ``Content-Length`` over this — or a streamed body that grows
            past it — raises a 413 signal. ``0`` disables the cap.

    Returns:
        The parsed JSON value.

    Raises:
        _BadRequest: The body exceeds the cap (status 413) or is not valid
            JSON (status 400).
    """
    if max_body_bytes > 0:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_body_bytes:
            raise _BadRequest(f"request body exceeds {max_body_bytes} bytes", status_code=413)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if max_body_bytes > 0 and len(body) > max_body_bytes:
            raise _BadRequest(f"request body exceeds {max_body_bytes} bytes", status_code=413)
    try:
        return json.loads(bytes(body))
    except (ValueError, UnicodeDecodeError) as exc:
        raise _BadRequest("request body must be valid JSON") from exc


async def _parse_prompt(request: Request, *, max_body_bytes: int) -> tuple[str, dict[str, Any]]:
    """Read and validate the JSON body, returning ``(prompt, body)``.

    Args:
        request: The incoming HTTP request.
        max_body_bytes: Maximum request-body size in bytes; ``0`` disables
            the cap.

    Returns:
        The validated prompt string and the full parsed body.

    Raises:
        _BadRequest: If the body exceeds the cap, is not valid JSON, or is
            not a JSON object with a non-empty string ``prompt``.
    """
    body = await _read_json_body(request, max_body_bytes)
    if not isinstance(body, dict):
        raise _BadRequest("request body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or len(prompt) == 0:
        raise _BadRequest("field 'prompt' must be a non-empty string")
    return prompt, body


def _resolve_max_turns(
    body: dict[str, Any],
    fallback: int | None,
    *,
    allow_client_max_turns_above_server_limit: bool = False,
) -> int | None:
    """Resolve the per-request turn budget, falling back to the server default.

    Args:
        body: The parsed request body.
        fallback: The app-level default (may be ``None``).
        allow_client_max_turns_above_server_limit: When ``True``, a client
            value above the app-level default is accepted. The default treats
            the app-level value as a ceiling.

    Returns:
        The positive turn budget, or ``None`` to defer to the framework.

    Raises:
        _BadRequest: If ``max_turns`` is present but not a positive integer.
    """
    raw = body.get("max_turns")
    if raw is None:
        return fallback
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise _BadRequest("field 'max_turns' must be a positive integer")
    if fallback is not None and raw > fallback and not allow_client_max_turns_above_server_limit:
        raise _BadRequest(f"field 'max_turns' must be at most the server limit ({fallback})")
    return raw


async def _open_session(body: dict[str, Any], factory: SessionFactory | None) -> SessionStore | None:
    """Build a session store from the request's ``session`` block.

    Args:
        body: The parsed request body.
        factory: The app's session factory, or ``None`` when sessions
            are not wired in.

    Returns:
        A session store, or ``None`` when no session was requested.

    Raises:
        _BadRequest: If a ``session`` block is malformed.
    """
    spec = body.get("session")
    if spec is None or factory is None:
        return None
    if not isinstance(spec, dict):
        raise _BadRequest("field 'session' must be a JSON object")
    user_id = spec.get("user_id")
    session_id = spec.get("session_id")
    if not isinstance(user_id, str) or not isinstance(session_id, str):
        raise _BadRequest("session requires string 'user_id' and 'session_id'")
    return await factory(user_id, session_id)


def _json_default(value: Any) -> Any:
    """Coerce a value :func:`json.dumps` cannot natively encode.

    Stream events forward developer-controlled payloads verbatim (e.g. a
    hook lifecycle ``payload`` whose values may be dataclasses or arbitrary
    objects). Coercing here means one exotic value degrades to a structured
    dict / string instead of raising ``TypeError`` and aborting the whole
    SSE stream before its terminal ``result`` event.

    Args:
        value: A value :func:`json.dumps` could not serialize natively.

    Returns:
        A JSON-serializable stand-in — a dict for dataclasses, else ``str``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return dataclasses.asdict(value)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _sse_dump(payload: object) -> str:
    """Serialize an SSE payload to JSON without ever raising.

    Uses the coercing :func:`_json_default` so developer-controlled values
    degrade gracefully; if serialization still fails (e.g. a circular
    reference), returns a small error object so a single bad event never
    aborts the stream before its terminal ``result`` frame.

    Args:
        payload: The already-projected JSON-able payload.

    Returns:
        A JSON string — the encoded payload, or an error marker on failure.
    """
    try:
        return json.dumps(payload, separators=(",", ":"), default=_json_default)
    except (TypeError, ValueError):
        logger.exception("Serializing an SSE payload failed; emitting an error marker instead.")
        return json.dumps({"error": "event serialization failed"}, separators=(",", ":"))


async def _event_source(
    streaming: RunResultStreaming,
    session: SessionStore | None,
) -> AsyncIterator[dict[str, str]]:
    """Yield Server-Sent Events for a streaming run, then a result event.

    Args:
        streaming: The streaming result whose events to forward.
        session: The per-request session to close when the stream ends.

    Yields:
        SSE frames as ``{"data": ...}`` dicts, ending with an ``event:
        result`` frame carrying the final summary. Serialization is
        defensive so a payload the JSON encoder cannot handle never aborts
        the stream before that terminal frame.
    """
    try:
        async for event in streaming.stream_events():
            payload = stream_event_to_dict(event)
            if payload is not None:
                yield {"data": _sse_dump(payload)}
        yield {"event": "result", "data": _sse_dump(streaming_result_to_dict(streaming))}
    finally:
        if session is not None:
            await session.close()


def rest_routes(
    agent: Agent[Any],
    *,
    max_turns: int | None = None,
    run_config: RunConfig | None = None,
    session_factory: SessionFactory | None = None,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    allow_client_max_turns_above_server_limit: bool = False,
) -> list[Route]:
    """Build the ``POST /run`` and ``POST /run_sse`` routes for *agent*.

    Args:
        agent: The agent each request runs.
        max_turns: App-level default agent-loop budget; ``None`` defers
            to the framework default. A request body may override it.
        run_config: Optional :class:`RunConfig` applied to every run.
        session_factory: Optional ``(user_id, session_id) -> SessionStore``
            used when a request carries a ``session`` block.
        max_body_bytes: Maximum request-body size in bytes; a larger body
            gets ``413 Payload Too Large``. Default 1 MiB; ``0`` disables
            the cap.
        allow_client_max_turns_above_server_limit: When ``False`` (default),
            ``max_turns`` is a server-enforced ceiling. Set ``True`` only
            when this surface deliberately allows clients to request larger
            turn budgets.

    Returns:
        The two REST :class:`starlette.routing.Route` objects.
    """

    async def run_endpoint(request: Request) -> JSONResponse:
        try:
            prompt, body = await _parse_prompt(request, max_body_bytes=max_body_bytes)
            turns = _resolve_max_turns(
                body,
                max_turns,
                allow_client_max_turns_above_server_limit=allow_client_max_turns_above_server_limit,
            )
            session = await _open_session(body, session_factory)
        except _BadRequest as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status_code)
        try:
            if turns is None:
                result = await Runner.arun(agent, prompt, session=session, run_config=run_config)
            else:
                result = await Runner.arun(agent, prompt, session=session, run_config=run_config, max_turns=turns)
        finally:
            if session is not None:
                await session.close()
        return JSONResponse(run_result_to_dict(result))

    async def run_sse_endpoint(request: Request) -> EventSourceResponse | JSONResponse:
        try:
            prompt, body = await _parse_prompt(request, max_body_bytes=max_body_bytes)
            turns = _resolve_max_turns(
                body,
                max_turns,
                allow_client_max_turns_above_server_limit=allow_client_max_turns_above_server_limit,
            )
            session = await _open_session(body, session_factory)
        except _BadRequest as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status_code)
        try:
            if turns is None:
                streaming = await Runner.arun(agent, prompt, session=session, run_config=run_config, stream=True)
            else:
                streaming = await Runner.arun(
                    agent, prompt, session=session, run_config=run_config, max_turns=turns, stream=True
                )
        except Exception:
            if session is not None:
                await session.close()
            raise
        return EventSourceResponse(_event_source(streaming, session))

    return [
        Route("/run", run_endpoint, methods=["POST"]),
        Route("/run_sse", run_sse_endpoint, methods=["POST"]),
    ]
