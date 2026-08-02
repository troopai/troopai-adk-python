"""Tests for ``AuditSink`` (P39)."""

from __future__ import annotations

import logging

import pytest

from troopai.adk.sandbox.observability import (
    AuditSink,
    LoggingAuditSink,
    NullAuditSink,
    SandboxAuditEvent,
)


def _make_event(event_type: str = "exec") -> SandboxAuditEvent:
    return SandboxAuditEvent(
        event_type=event_type,  # type: ignore[arg-type]
        agent_name="alice",
        backend_id="unix_local",
        session_id="sess-1",
        command="ls",
        exit_code=0,
    )


class TestSandboxAuditEvent:
    def test_construction(self) -> None:
        e = _make_event()
        assert e.event_type == "exec"
        assert e.timestamp_iso.endswith("+00:00") or "T" in e.timestamp_iso

    def test_default_extras_empty(self) -> None:
        assert _make_event().extra == {}


class TestNullAuditSink:
    @pytest.mark.asyncio
    async def test_emit_no_op(self) -> None:
        sink = NullAuditSink()
        await sink.emit(_make_event())  # no raise


class TestLoggingAuditSink:
    @pytest.mark.asyncio
    async def test_logs_at_default_levels(self) -> None:
        captured: list[tuple[int, str]] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append((record.levelno, record.getMessage()))

        logger = logging.getLogger("test.audit")
        logger.setLevel(logging.DEBUG)
        handler = _Capture()
        logger.addHandler(handler)
        try:
            sink = LoggingAuditSink(logger)
            await sink.emit(_make_event("start"))
            await sink.emit(_make_event("exec"))
            await sink.emit(_make_event("violation"))
            await sink.emit(_make_event("error"))
            levels = [lvl for lvl, _ in captured]
            assert logging.INFO in levels  # start
            assert logging.DEBUG in levels  # exec
            assert logging.WARNING in levels  # violation
            assert logging.ERROR in levels  # error
        finally:
            logger.removeHandler(handler)

    @pytest.mark.asyncio
    async def test_custom_level_map(self) -> None:
        captured: list[int] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record.levelno)

        logger = logging.getLogger("test.audit2")
        logger.setLevel(logging.DEBUG)
        handler = _Capture()
        logger.addHandler(handler)
        try:
            sink = LoggingAuditSink(
                logger,
                level_map={"start": logging.WARNING},
            )
            await sink.emit(_make_event("start"))
            assert logging.WARNING in captured
        finally:
            logger.removeHandler(handler)


class TestSinkABCEnforcement:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AuditSink()  # type: ignore[abstract]
