"""Tests for ``troopai.adk.sandbox.session.manager`` (Instrumentation)."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from troopai.adk.sandbox.session import (
    CallbackSink,
    ChainedSink,
    EventPayloadPolicy,
    Instrumentation,
    OpName,
    SandboxSessionEvent,
    SandboxSessionFinishEvent,
    SandboxSessionStartEvent,
)

SESSION_ID = uuid.UUID("00000000-0000-0000-0000-0000000000aa")


def _as_finish(event: SandboxSessionEvent) -> SandboxSessionFinishEvent:
    """Narrow a captured union event to a finish event for attribute access."""
    assert isinstance(event, SandboxSessionFinishEvent)
    return event


def _start(op: OpName = OpName.RUN) -> SandboxSessionStartEvent:
    return SandboxSessionStartEvent.model_validate({"session_id": SESSION_ID, "seq": 1, "op": op, "span_id": "s1"})


def _finish(
    *,
    op: OpName = OpName.RUN,
    stdout_bytes: bytes | None = None,
    data: dict[str, object] | None = None,
) -> SandboxSessionFinishEvent:
    payload: dict[str, object] = {
        "session_id": SESSION_ID,
        "seq": 2,
        "op": op,
        "span_id": "s1",
        "ok": True,
        "duration_ms": 1.0,
    }
    if stdout_bytes is not None:
        payload["stdout_bytes"] = stdout_bytes
    if data is not None:
        payload["data"] = data
    return SandboxSessionFinishEvent.model_validate(payload)


class TestEmitDelivery:
    async def test_emits_to_all_sinks(self) -> None:
        a: list[object] = []
        b: list[object] = []
        instr = Instrumentation(sinks=[CallbackSink(lambda e: a.append(e)), CallbackSink(lambda e: b.append(e))])
        await instr.emit(_start())
        assert len(a) == 1
        assert len(b) == 1

    async def test_add_sink_takes_effect(self) -> None:
        seen: list[object] = []
        instr = Instrumentation()
        instr.add_sink(CallbackSink(lambda e: seen.append(e)))
        await instr.emit(_start())
        assert len(seen) == 1

    async def test_chained_sink_delivered_in_order(self) -> None:
        order: list[str] = []
        chain = ChainedSink(
            [
                CallbackSink(lambda _e: order.append("a")),
                CallbackSink(lambda _e: order.append("b")),
            ]
        )
        instr = Instrumentation(sinks=[chain])
        await instr.emit(_start())
        assert order == ["a", "b"]

    async def test_sinks_property_is_snapshot(self) -> None:
        sink = CallbackSink(lambda _e: None)
        instr = Instrumentation(sinks=[sink])
        snapshot = instr.sinks
        snapshot.clear()
        assert len(instr.sinks) == 1


class TestPayloadPolicyLayering:
    async def test_exec_output_redacted_by_default(self) -> None:
        captured: list[SandboxSessionEvent] = []
        instr = Instrumentation(sinks=[CallbackSink(lambda e: captured.append(e))])
        await instr.emit(_finish(stdout_bytes=b"secret"))
        fin = _as_finish(captured[0])
        assert fin.stdout is None
        assert fin.stdout_bytes is None

    async def test_per_op_policy_enables_exec_output(self) -> None:
        captured: list[SandboxSessionEvent] = []
        instr = Instrumentation(
            sinks=[CallbackSink(lambda e: captured.append(e))],
            payload_policy_by_op={OpName.RUN: EventPayloadPolicy(include_exec_output=True)},
        )
        await instr.emit(_finish(op=OpName.RUN, stdout_bytes=b"hello stdout"))
        assert _as_finish(captured[0]).stdout == "hello stdout"

    async def test_per_sink_policy_overrides_only_set_fields(self) -> None:
        captured: list[SandboxSessionEvent] = []
        # Per-sink policy enables exec output but leaves max_stdout_chars
        # at the default — the merge must not reset it to a different value.
        sink = CallbackSink(
            lambda e: captured.append(e),
            payload_policy=EventPayloadPolicy(include_exec_output=True),
        )
        instr = Instrumentation(sinks=[sink])
        await instr.emit(_finish(stdout_bytes=b"x" * 10))
        assert _as_finish(captured[0]).stdout == "x" * 10

    async def test_stdout_truncated_to_policy_max(self) -> None:
        captured: list[SandboxSessionEvent] = []
        sink = CallbackSink(
            lambda e: captured.append(e),
            payload_policy=EventPayloadPolicy(include_exec_output=True, max_stdout_chars=5),
        )
        instr = Instrumentation(sinks=[sink])
        await instr.emit(_finish(stdout_bytes=b"abcdefghij"))
        assert len(_as_finish(captured[0]).stdout or "") <= 6  # 5 chars + ellipsis

    async def test_write_len_redacted_when_disabled(self) -> None:
        captured: list[SandboxSessionEvent] = []
        sink = CallbackSink(
            lambda e: captured.append(e),
            payload_policy=EventPayloadPolicy(include_write_len=False),
        )
        instr = Instrumentation(sinks=[sink])
        await instr.emit(_finish(data={"bytes": 4096, "path": "/x"}))
        fin = _as_finish(captured[0])
        assert "bytes" not in fin.data
        assert fin.data["path"] == "/x"

    async def test_policy_clone_isolated_per_sink(self) -> None:
        a: list[SandboxSessionEvent] = []
        b: list[SandboxSessionEvent] = []
        sink_a = CallbackSink(
            lambda e: a.append(e),
            payload_policy=EventPayloadPolicy(include_exec_output=True),
        )
        sink_b = CallbackSink(lambda e: b.append(e))  # default: redacted
        instr = Instrumentation(sinks=[sink_a, sink_b])
        await instr.emit(_finish(stdout_bytes=b"data"))
        assert _as_finish(a[0]).stdout == "data"
        assert _as_finish(b[0]).stdout is None


class TestDeliveryModesAndErrors:
    async def test_sync_on_error_raise_propagates(self) -> None:
        def boom(_e: object) -> None:
            raise RuntimeError("sink failed")

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="sync", on_error="raise")])
        with pytest.raises(RuntimeError, match="sandbox event sink failed"):
            await instr.emit(_start())

    async def test_sync_on_error_log_swallows(self) -> None:
        def boom(_e: object) -> None:
            raise RuntimeError("sink failed")

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="sync", on_error="log")])
        await instr.emit(_start())  # no raise

    async def test_async_mode_runs_in_background_and_flush_awaits(self) -> None:
        done = asyncio.Event()

        async def slow(_e: object) -> None:
            await asyncio.sleep(0.01)
            done.set()

        instr = Instrumentation(sinks=[CallbackSink(slow, mode="async", on_error="log")])
        await instr.emit(_start())
        assert not done.is_set()  # background, not yet awaited
        await instr.flush()
        assert done.is_set()

    async def test_best_effort_swallows_failure(self) -> None:
        def boom(_e: object) -> None:
            raise RuntimeError("sink failed")

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="best_effort", on_error="raise")])
        await instr.emit(_start())  # best_effort forces no-raise
        await instr.flush()

    async def test_unknown_mode_raises_value_error(self) -> None:
        sink = CallbackSink(lambda _e: None)
        sink.mode = "telepathy"  # type: ignore[assignment]  # EventSink.mode is Literal[...]; out-of-Literal value drives the exhaustiveness fallback
        instr = Instrumentation(sinks=[sink])
        with pytest.raises(ValueError, match="unknown sink.mode"):
            await instr.emit(_start())

    async def test_flush_noop_when_no_background_tasks(self) -> None:
        instr = Instrumentation(sinks=[CallbackSink(lambda _e: None)])
        await instr.emit(_start())
        await instr.flush()  # no raise, no hang


class TestNeverDowngradeContract:
    async def test_programmer_bug_reraised_despite_on_error_log_sync(self) -> None:
        # TypeError is a NEVER_DOWNGRADE_EXC: on_error="log" MUST NOT bury it.
        def buggy(_e: object) -> None:
            raise TypeError("bug in custom sink")

        instr = Instrumentation(sinks=[CallbackSink(buggy, mode="sync", on_error="log")])
        with pytest.raises(TypeError, match="bug in custom sink"):
            await instr.emit(_start())

    async def test_programmer_bug_reraised_despite_best_effort(self) -> None:
        # best_effort forces no-raise for transport errors, but an
        # AttributeError is a real bug and must surface via flush().
        def buggy(_e: object) -> None:
            raise AttributeError("typo'd attr in sink")

        instr = Instrumentation(sinks=[CallbackSink(buggy, mode="best_effort", on_error="ignore")])
        await instr.emit(_start())
        # The re-raised AttributeError escapes the background task; flush
        # gathers it and logs ERROR (does not itself raise).
        await instr.flush()

    async def test_flush_surfaces_unhandled_background_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        def buggy(_e: object) -> None:
            raise NameError("undefined name in sink")

        instr = Instrumentation(sinks=[CallbackSink(buggy, mode="best_effort", on_error="ignore")])
        await instr.emit(_start())
        with caplog.at_level("ERROR", logger="troopai.adk.sandbox.session.manager"):
            await instr.flush()
        assert any("NOT" in rec.getMessage() and "on_error policy" in rec.getMessage() for rec in caplog.records)

    async def test_transport_error_still_downgraded(self) -> None:
        # RuntimeError is NOT a programmer-bug type — on_error="log"
        # still swallows it (the normal transport-failure path).
        def transient(_e: object) -> None:
            raise RuntimeError("transient transport blip")

        instr = Instrumentation(sinks=[CallbackSink(transient, mode="sync", on_error="log")])
        await instr.emit(_start())  # no raise


class TestErrorChainingAndBreadcrumbs:
    async def test_raise_policy_chains_original_cause(self) -> None:
        original = RuntimeError("the real failure")

        def boom(_e: object) -> None:
            raise original

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="sync", on_error="raise")])
        with pytest.raises(RuntimeError, match="sandbox event sink failed") as exc_info:
            await instr.emit(_start())
        assert exc_info.value.__cause__ is original

    async def test_async_raise_path_also_chains(self) -> None:
        original = RuntimeError("async real failure")

        async def boom(_e: object) -> None:
            raise original

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="async", on_error="raise")])
        with pytest.raises(RuntimeError, match="sandbox event sink failed") as exc_info:
            await instr.emit(_start())
        assert exc_info.value.__cause__ is original

    async def test_ignore_policy_leaves_debug_breadcrumb(self, caplog: pytest.LogCaptureFixture) -> None:
        def boom(_e: object) -> None:
            raise RuntimeError("dropped silently-but-traceably")

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="sync", on_error="ignore")])
        with caplog.at_level("DEBUG", logger="troopai.adk.sandbox.session.manager"):
            await instr.emit(_start())
        assert any("dropped (on_error=ignore" in rec.getMessage() for rec in caplog.records)

    async def test_log_policy_includes_event_identity(self, caplog: pytest.LogCaptureFixture) -> None:
        def boom(_e: object) -> None:
            raise RuntimeError("logged failure")

        instr = Instrumentation(sinks=[CallbackSink(boom, mode="sync", on_error="log")])
        with caplog.at_level("ERROR", logger="troopai.adk.sandbox.session.manager"):
            await instr.emit(_start())
        record = next(r for r in caplog.records if "event sink failed" in r.getMessage())
        assert "session=" in record.getMessage()
        assert "seq=" in record.getMessage()
        assert "op=" in record.getMessage()


class TestFrozenPayloadPolicy:
    def test_post_construction_mutation_raises(self) -> None:
        policy = EventPayloadPolicy()
        with pytest.raises((TypeError, ValueError)):
            policy.include_exec_output = True  # type: ignore[misc]  # frozen model — mutation must raise, not silently no-op

    def test_model_copy_update_still_works(self) -> None:
        policy = EventPayloadPolicy()
        tightened = policy.model_copy(update={"include_exec_output": True})
        assert tightened.include_exec_output is True
        assert policy.include_exec_output is False  # original untouched
