"""Tests for the plain-REST surface (``POST /run`` and ``POST /run_sse``)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("starlette")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

from starlette.applications import Starlette
from starlette.testclient import TestClient

from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.run.stream import HookEventKind, HookLifecycleEvent
from troopai.adk.serving import build_app
from troopai.adk.serving.rest import _event_source, _sse_dump, rest_routes


class FakeSession:
    """Minimal in-memory ``SessionStore`` double for the session seam."""

    def __init__(self, session_id: str) -> None:
        self._id = session_id
        self.closed = False

    @property
    def id(self) -> str:
        return self._id

    @property
    def settings(self) -> None:
        return None

    async def get(self, limit: int | None = None) -> list[Any]:
        return []

    async def add(self, events: list[Any]) -> None:
        return None

    async def save_state(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def test_run_returns_final_output(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, rest=True))
    resp = client.post("/run", json={"prompt": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_output"] == "hello from agent"
    assert body["requires_action"] is False
    assert body["last_agent"] == "support"
    assert isinstance(body["new_items"], list)


def test_run_missing_prompt_is_400(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, rest=True))
    resp = client.post("/run", json={})
    assert resp.status_code == 400
    assert "prompt" in resp.json()["error"]


def test_run_empty_prompt_is_400(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, rest=True))
    assert client.post("/run", json={"prompt": ""}).status_code == 400


def test_run_invalid_json_is_400(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, rest=True))
    resp = client.post("/run", content=b"not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_run_zero_max_turns_is_400(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, rest=True))
    assert client.post("/run", json={"prompt": "hi", "max_turns": 0}).status_code == 400


def test_run_rejects_max_turns_above_server_ceiling(scripted_agent: Agent[None]) -> None:
    client = TestClient(Starlette(routes=rest_routes(scripted_agent, max_turns=2)))

    resp = client.post("/run", json={"prompt": "hi", "max_turns": 3})

    assert resp.status_code == 400
    assert "max_turns" in resp.json()["error"]


def test_run_sse_streams_a_result_event(streaming_agent: Agent[None]) -> None:
    client = TestClient(build_app(streaming_agent, rest=True))
    resp = client.post("/run_sse", json={"prompt": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "result" in text
    assert "streamed reply" in text


def test_run_session_factory_invoked_and_closed(scripted_agent: Agent[None]) -> None:
    created: list[tuple[str, str]] = []
    made: list[FakeSession] = []

    async def factory(user_id: str, session_id: str) -> FakeSession:
        created.append((user_id, session_id))
        session = FakeSession(session_id)
        made.append(session)
        return session

    client = TestClient(build_app(scripted_agent, rest=True, session_factory=factory))
    resp = client.post("/run", json={"prompt": "hi", "session": {"user_id": "u1", "session_id": "s1"}})
    assert resp.status_code == 200
    assert created == [("u1", "s1")]
    assert made[0].closed is True


def test_run_sse_closes_session_when_stream_construction_fails(
    scripted_agent: Agent[None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[FakeSession] = []

    async def factory(_user_id: str, session_id: str) -> FakeSession:
        session = FakeSession(session_id)
        made.append(session)
        return session

    async def fail_arun(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("stream setup failed")

    monkeypatch.setattr(Runner, "arun", staticmethod(fail_arun))
    client = TestClient(
        Starlette(routes=rest_routes(scripted_agent, session_factory=factory)),
        raise_server_exceptions=False,
    )

    resp = client.post("/run_sse", json={"prompt": "hi", "session": {"user_id": "u1", "session_id": "s1"}})

    assert resp.status_code == 500
    assert made[0].closed is True


# ----------------------------------------------------------------------
# Regression: request-body size cap (413) — rest_routes exposes the knob;
# build_app/CLI wiring is a separate follow-up (unowned files).
# ----------------------------------------------------------------------


def test_run_oversize_body_is_413(scripted_agent: Agent[None]) -> None:
    # Regression: the body was read whole with no cap. A body over the cap
    # must be rejected with 413 rather than buffered into memory.
    client = TestClient(Starlette(routes=rest_routes(scripted_agent, max_body_bytes=16)))
    resp = client.post("/run", json={"prompt": "x" * 100})
    assert resp.status_code == 413


def test_run_body_within_cap_ok(scripted_agent: Agent[None]) -> None:
    client = TestClient(Starlette(routes=rest_routes(scripted_agent, max_body_bytes=4096)))
    resp = client.post("/run", json={"prompt": "hi"})
    assert resp.status_code == 200


# ----------------------------------------------------------------------
# Regression: SSE serialization must not abort on an unserializable payload
# ----------------------------------------------------------------------


def test_sse_dump_coerces_dataclass_payload() -> None:
    # Root cause: json.dumps of a verbatim dataclass payload raised
    # TypeError. The SSE dump must coerce it into structured JSON, not crash.
    @dataclasses.dataclass
    class _Payload:
        x: int

    parsed = json.loads(_sse_dump({"payload": {"tool_output": _Payload(3)}}))
    assert parsed["payload"]["tool_output"] == {"x": 3}


class _FakeStreaming:
    """Minimal ``RunResultStreaming`` double for the SSE serialization path."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.context = None
        self.final_output = "done"
        self.new_items: list[Any] = []
        self.current_agent = None

    async def stream_events(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event


async def test_sse_stream_survives_unserializable_hook_payload() -> None:
    # Regression: a hook payload carrying a dataclass aborted the SSE stream
    # with a TypeError before the terminal 'result' event. That terminal
    # event must still be emitted.
    @dataclasses.dataclass
    class _ToolOut:
        value: int

    bad_event = HookLifecycleEvent(
        kind=HookEventKind.TOOL_END,
        agent_name="support",
        payload={"tool_output": _ToolOut(7)},
    )
    streaming: Any = _FakeStreaming([bad_event])
    frames = [frame async for frame in _event_source(streaming, None)]
    # The terminal result frame is present despite the unserializable event.
    assert any(frame.get("event") == "result" for frame in frames)
    # The bad event was coerced into a data frame rather than crashing.
    data_frames = [f for f in frames if "data" in f and f.get("event") != "result"]
    assert len(data_frames) == 1
