"""Tests for ``troopai.adk.sandbox.session.sinks``."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import override
from unittest.mock import patch

import pytest

from troopai.adk.sandbox.session import (
    CallbackSink,
    ChainedSink,
    EventSink,
    HttpPostSink,
    JsonlOutboxSink,
    OpName,
    SandboxSessionFinishEvent,
    SandboxSessionStartEvent,
)

SESSION_ID = uuid.UUID("00000000-0000-0000-0000-000000000077")


def _start_event() -> SandboxSessionStartEvent:
    return SandboxSessionStartEvent.model_validate(
        {
            "session_id": SESSION_ID,
            "seq": 1,
            "op": OpName.RUN,
            "span_id": "span-1",
        }
    )


def _finish_event(*, ok: bool = True) -> SandboxSessionFinishEvent:
    return SandboxSessionFinishEvent.model_validate(
        {
            "session_id": SESSION_ID,
            "seq": 2,
            "op": OpName.RUN,
            "span_id": "span-1",
            "ok": ok,
            "duration_ms": 1.0,
        }
    )


class TestCallbackSink:
    async def test_invokes_sync_callable(self) -> None:
        captured: list[object] = []
        sink = CallbackSink(lambda e: captured.append(e))
        await sink.handle(_start_event())
        assert len(captured) == 1

    async def test_invokes_async_callable(self) -> None:
        captured: list[object] = []

        async def acb(event: object) -> None:
            captured.append(event)

        sink = CallbackSink(acb)
        await sink.handle(_start_event())
        assert len(captured) == 1

    async def test_callback_exception_propagates_by_default(self) -> None:
        def bad(_event: object) -> None:
            raise RuntimeError("boom")

        sink = CallbackSink(bad)
        with pytest.raises(RuntimeError, match="boom"):
            await sink.handle(_start_event())


class TestJsonlOutboxSink:
    async def test_writes_one_line_per_event(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        sink = JsonlOutboxSink(path)
        await sink.handle(_start_event())
        await sink.handle(_finish_event())
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        parsed_start = json.loads(lines[0])
        parsed_finish = json.loads(lines[1])
        assert parsed_start["phase"] == "start"
        assert parsed_finish["phase"] == "finish"

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "audit.jsonl"
        sink = JsonlOutboxSink(path)
        await sink.handle(_start_event())
        assert path.exists()

    async def test_concurrent_writes_serialize(self, tmp_path: Path) -> None:
        import asyncio

        path = tmp_path / "audit.jsonl"
        sink = JsonlOutboxSink(path)
        # Fire 10 concurrent writes; each line should land intact.
        await asyncio.gather(*(sink.handle(_start_event()) for _ in range(10)))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 10
        for line in lines:
            payload = json.loads(line)
            assert payload["phase"] == "start"


class TestHttpPostSink:
    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            HttpPostSink("")

    async def test_posts_serialized_event(self) -> None:
        captured: dict[str, object] = {}

        class _FakeResp:
            def read(self) -> bytes:
                return b""

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *_: object) -> None:
                return None

        def fake_urlopen(request: object, *, timeout: float) -> _FakeResp:
            captured["url"] = getattr(request, "full_url", None)
            captured["data"] = getattr(request, "data", None)
            captured["method"] = getattr(request, "method", None)
            captured["timeout"] = timeout
            return _FakeResp()

        with patch("troopai.adk.sandbox.session.sinks.urlopen", fake_urlopen):
            sink = HttpPostSink("https://example.invalid/audit")
            await sink.handle(_start_event())

        assert captured["url"] == "https://example.invalid/audit"
        assert captured["method"] == "POST"
        payload = json.loads(captured["data"])  # type: ignore[arg-type]  # bytes is JSON-decodable
        assert payload["phase"] == "start"

    async def test_wraps_transport_errors(self) -> None:
        with patch(
            "troopai.adk.sandbox.session.sinks.urlopen",
            side_effect=TimeoutError("slow"),
        ):
            sink = HttpPostSink("https://example.invalid/audit")
            with pytest.raises(RuntimeError, match="timed out after"):
                await sink.handle(_start_event())


class TestChainedSink:
    async def test_fanout_to_all_sinks(self) -> None:
        calls_a: list[object] = []
        calls_b: list[object] = []
        sink_a = CallbackSink(lambda e: calls_a.append(e))
        sink_b = CallbackSink(lambda e: calls_b.append(e))
        chain = ChainedSink([sink_a, sink_b])
        await chain.handle(_start_event())
        assert len(calls_a) == 1
        assert len(calls_b) == 1

    async def test_first_failure_aborts_when_on_error_raise(self) -> None:
        calls_b: list[object] = []

        def bad(_e: object) -> None:
            raise RuntimeError("boom")

        sink_a = CallbackSink(bad, on_error="raise")
        sink_b = CallbackSink(lambda e: calls_b.append(e))
        chain = ChainedSink([sink_a, sink_b])
        with pytest.raises(RuntimeError, match="boom"):
            await chain.handle(_start_event())
        # Second sink MUST NOT have been invoked.
        assert calls_b == []

    async def test_on_error_log_continues_chain(self) -> None:
        calls_b: list[object] = []

        def bad(_e: object) -> None:
            raise RuntimeError("boom")

        sink_a = CallbackSink(bad, on_error="log")
        sink_b = CallbackSink(lambda e: calls_b.append(e))
        chain = ChainedSink([sink_a, sink_b])
        await chain.handle(_start_event())
        assert len(calls_b) == 1

    async def test_on_error_log_logs_at_error_level(self) -> None:
        """Regression: ChainedSink used logger.warning but Instrumentation uses
        logger.error for on_error='log'. Align to ERROR to avoid misleading alert
        rules that only trigger on ERROR and above."""
        import logging

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        target_logger = logging.getLogger("troopai.adk.sandbox.session.sinks")
        handler = _Capture()
        target_logger.addHandler(handler)
        try:

            def bad(_e: object) -> None:
                raise RuntimeError("oops")

            sink = CallbackSink(bad, on_error="log")
            chain = ChainedSink([sink])
            await chain.handle(_start_event())
        finally:
            target_logger.removeHandler(handler)

        assert len(records) == 1
        assert records[0].levelno == logging.ERROR, f"Expected ERROR but got {records[0].levelname}"

    async def test_on_error_ignore_swallows_silently(self) -> None:
        calls_b: list[object] = []

        def bad(_e: object) -> None:
            raise RuntimeError("boom")

        sink_a = CallbackSink(bad, on_error="ignore")
        sink_b = CallbackSink(lambda e: calls_b.append(e))
        chain = ChainedSink([sink_a, sink_b])
        await chain.handle(_start_event())
        assert len(calls_b) == 1

    def test_sinks_view_is_read_only(self) -> None:
        sink_a = CallbackSink(lambda _e: None)
        chain = ChainedSink([sink_a])
        view = chain.sinks
        # Tuples don't have append; this asserts the property is a tuple.
        assert isinstance(view, tuple)


class TestEventSinkBaseClass:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            EventSink()  # type: ignore[abstract]  # explicit ABC instantiation test

    def test_subclass_must_implement_handle(self) -> None:
        class Incomplete(EventSink):
            mode = "sync"
            on_error = "raise"
            payload_policy = None

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]  # asserting ABC-instantiation TypeError

    def test_complete_subclass_can_be_instantiated(self) -> None:
        class Complete(EventSink):
            def __init__(self) -> None:
                self.mode = "sync"
                self.on_error = "raise"
                self.payload_policy = None

            @override
            async def handle(self, event: object) -> None:
                _ = event

        Complete()  # no raise

    def test_subclass_omitting_mode_on_error_uses_class_defaults(self) -> None:
        """Regression: mode/on_error/payload_policy had no class-level default.

        A subclass that only implements handle() but omits those attributes in
        __init__ was silently instantiable yet raised AttributeError at delivery.
        The fix adds class-level defaults so omitting them is safe.
        """

        class MinimalSink(EventSink):
            """Only implements handle — relies on base class defaults."""

            @override
            async def handle(self, event: object) -> None:
                _ = event

        sink = MinimalSink()
        # Access must not raise AttributeError.
        assert sink.mode == "sync"
        assert sink.on_error == "raise"
        assert sink.payload_policy is None
