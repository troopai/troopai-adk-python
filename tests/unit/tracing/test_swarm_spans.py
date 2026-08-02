"""Unit tests for SwarmSpanData / SwarmTurnSpanData payload contracts."""

from __future__ import annotations

from unittest.mock import patch

from troopai.adk.tracing.spans import NoOpSpan, swarm_span, swarm_turn_span
from troopai.adk.types.tracing.span_data import (
    CustomSpanData,
    SwarmSpanData,
    SwarmTurnSpanData,
)


class TestSwarmSpanData:
    def test_minimal_construction(self) -> None:
        data = SwarmSpanData(swarm_id="abc-123")
        assert data.swarm_id == "abc-123"
        assert data.entry is None
        assert data.status is None
        assert data.turns_total is None
        assert data.type == "swarm"

    def test_full_construction_round_trips_via_export(self) -> None:
        data = SwarmSpanData(
            swarm_id="abc-123",
            entry="approver",
            status="completed",
            turns_total=4,
        )
        exported = data.export()
        assert exported == {
            "type": "swarm",
            "swarm_id": "abc-123",
            "entry": "approver",
            "status": "completed",
            "turns_total": 4,
        }


class TestSwarmTurnSpanData:
    def test_minimal_construction(self) -> None:
        data = SwarmTurnSpanData(
            swarm_id="abc-123",
            index=1,
            member="approver",
        )
        assert data.swarm_id == "abc-123"
        assert data.index == 1
        assert data.member == "approver"
        assert data.status is None
        assert data.duration_ms is None
        assert data.resume_attempt is None
        assert data.type == "swarm_turn"

    def test_full_construction_round_trips_via_export(self) -> None:
        data = SwarmTurnSpanData(
            swarm_id="abc-123",
            index=3,
            member="approver",
            status="interrupted",
            duration_ms=147,
            resume_attempt=2,
        )
        exported = data.export()
        assert exported == {
            "type": "swarm_turn",
            "swarm_id": "abc-123",
            "index": 3,
            "member": "approver",
            "status": "interrupted",
            "duration_ms": 147,
            "resume_attempt": 2,
        }


class TestSwarmSpanFactory:
    def test_disabled_returns_noop_span(self) -> None:
        span = swarm_span(swarm_id="abc-123", entry="approver", disabled=True)
        assert isinstance(span, NoOpSpan)

    def test_enabled_routes_through_tracer_with_swarm_typed_payload(self) -> None:
        captured: list[CustomSpanData] = []

        class _FakeTracer:
            def custom_span(self, data: CustomSpanData) -> NoOpSpan[CustomSpanData]:
                captured.append(data)
                return NoOpSpan(data)

        with patch("troopai.adk.tracing.spans.get_tracer", return_value=_FakeTracer()):
            swarm_span(swarm_id="abc-123", entry="approver")

        assert len(captured) == 1
        assert captured[0].name == "swarm.abc-123"
        assert captured[0].data["type"] == "swarm"
        assert captured[0].data["swarm_id"] == "abc-123"
        assert captured[0].data["entry"] == "approver"


class TestSwarmTurnSpanFactory:
    def test_disabled_returns_noop_span(self) -> None:
        span = swarm_turn_span(
            swarm_id="abc-123",
            index=1,
            member="approver",
            disabled=True,
        )
        assert isinstance(span, NoOpSpan)

    def test_enabled_routes_through_tracer_with_swarm_turn_typed_payload(self) -> None:
        captured: list[CustomSpanData] = []

        class _FakeTracer:
            def custom_span(self, data: CustomSpanData) -> NoOpSpan[CustomSpanData]:
                captured.append(data)
                return NoOpSpan(data)

        with patch("troopai.adk.tracing.spans.get_tracer", return_value=_FakeTracer()):
            swarm_turn_span(swarm_id="abc-123", index=3, member="approver")

        assert len(captured) == 1
        assert captured[0].name == "swarm.turn.3"
        assert captured[0].data["type"] == "swarm_turn"
        assert captured[0].data["swarm_id"] == "abc-123"
        assert captured[0].data["index"] == 3
        assert captured[0].data["member"] == "approver"
