"""Unit tests for :class:`FlowCheckpoint` JSON round-trip + structural validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from troopai.adk.flows import (
    FlowCheckpoint,
    FlowDeferredStep,
    FlowTriggerEvent,
)


class TestRoundTrip:
    def test_minimal_round_trip(self) -> None:
        cp = FlowCheckpoint(
            flow_id="flow-abc",
            completed_steps=("a", "b"),
            pending_steps=("c",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"x": 1}',
        )
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        assert rehydrated.flow_id == "flow-abc"
        assert rehydrated.completed_steps == ("a", "b")
        assert rehydrated.pending_steps == ("c",)
        assert rehydrated.state_data == '{"x": 1}'
        assert rehydrated.deferred_steps == ()

    def test_round_trip_with_gates(self) -> None:
        cp = FlowCheckpoint(
            flow_id="flow-xyz",
            completed_steps=("a",),
            pending_steps=("b",),
            and_gate_arrivals={"gate1": ("a", "b")},
            consumed_gates=("gate2",),
            state_data="{}",
        )
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        assert rehydrated.and_gate_arrivals == {"gate1": ("a", "b")}
        assert rehydrated.consumed_gates == ("gate2",)

    def test_round_trip_with_deferred_steps(self) -> None:
        # Fixed timestamp so the round-trip is byte-exact.
        ts = datetime(2026, 1, 15, 12, 30, 0, tzinfo=UTC)
        deferred = FlowDeferredStep(
            step_name="sensitive_step",
            triggers=(FlowTriggerEvent(name="upstream", source_step="upstream", kind="step_completion"),),
            request_time=ts,
        )
        cp = FlowCheckpoint(
            flow_id="flow-defer",
            completed_steps=("upstream",),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            deferred_steps=(deferred,),
        )
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        assert len(rehydrated.deferred_steps) == 1
        out = rehydrated.deferred_steps[0]
        assert out.step_name == "sensitive_step"
        assert out.request_time == ts
        assert len(out.triggers) == 1
        assert out.triggers[0].name == "upstream"
        assert out.triggers[0].source_step == "upstream"
        assert out.triggers[0].kind == "step_completion"


class TestStructuralValidation:
    @pytest.mark.parametrize(
        "missing_field",
        [
            "flow_id",
            "completed_steps",
            "pending_steps",
            "and_gate_arrivals",
            "consumed_gates",
            "state_data",
        ],
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        full_payload: dict[str, object] = {
            "flow_id": "x",
            "completed_steps": [],
            "pending_steps": [],
            "and_gate_arrivals": {},
            "consumed_gates": [],
            "state_data": "{}",
        }
        del full_payload[missing_field]
        with pytest.raises(ValueError, match="missing required field"):
            FlowCheckpoint.from_json(json.dumps(full_payload))

    def test_corrupted_trigger_kind_raises(self) -> None:
        payload = json.dumps(
            {
                "flow_id": "x",
                "completed_steps": [],
                "pending_steps": [],
                "and_gate_arrivals": {},
                "consumed_gates": [],
                "state_data": "{}",
                "deferred_steps": [
                    {
                        "step_name": "s",
                        "triggers": [{"name": "n", "source_step": "n", "kind": "garbage"}],
                        "request_time": "2026-01-15T12:30:00+00:00",
                    },
                ],
            }
        )
        with pytest.raises(ValueError, match="FlowTriggerEvent.kind"):
            FlowCheckpoint.from_json(payload)


class TestCapture:
    def test_capture_sorts_for_stability(self) -> None:
        cp = FlowCheckpoint.capture(
            flow_id="x",
            completed_steps=("a", "b"),
            pending_steps=("c",),
            and_gate_arrivals={"g": {"b", "a"}},
            consumed_gates={"g2", "g1"},
            state_data="{}",
        )
        # Sorted by sorted() — deterministic across runs.
        assert cp.and_gate_arrivals == {"g": ("a", "b")}
        assert cp.consumed_gates == ("g1", "g2")

    def test_capture_passes_deferred_steps_through(self) -> None:
        deferred = FlowDeferredStep(
            step_name="s",
            triggers=(FlowTriggerEvent(name="t", source_step="t", kind="step_completion"),),
        )
        cp = FlowCheckpoint.capture(
            flow_id="x",
            completed_steps=(),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=set(),
            state_data="{}",
            deferred_steps=(deferred,),
        )
        assert cp.deferred_steps == (deferred,)


# ── Finding 9: pending_step_triggers round-trip ──────────────────────────────


class TestPendingStepTriggersRoundTrip:
    """Finding 9: pending_step_triggers must survive a to_json/from_json round-trip.

    Non-deferred pending steps that were scheduled by sibling completions in
    the same batch as a deferral have their trigger events stored in
    ``pending_triggers`` at checkpoint time.  Without serialising them, the
    resumed executor's ``_build_step_context`` pops an empty tuple from
    ``pending_triggers``, silently corrupting ``ctx.triggers`` for any gate
    callable that branches on it.
    """

    def test_pending_step_triggers_round_trips(self) -> None:
        """pending_step_triggers survives to_json / from_json."""
        trigger = FlowTriggerEvent(
            name="sibling_done",
            source_step="sibling_step",
            kind="step_completion",
        )
        cp = FlowCheckpoint(
            flow_id="flow-pst",
            completed_steps=("sibling_step",),
            pending_steps=("pending_step",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            pending_step_triggers={"pending_step": (trigger,)},
        )
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        assert "pending_step" in rehydrated.pending_step_triggers
        restored = rehydrated.pending_step_triggers["pending_step"]
        assert len(restored) == 1
        assert restored[0].name == "sibling_done"
        assert restored[0].source_step == "sibling_step"
        assert restored[0].kind == "step_completion"

    def test_pending_step_triggers_empty_by_default(self) -> None:
        """A checkpoint with no pending_step_triggers field deserializes to {}."""
        payload = json.dumps(
            {
                "flow_id": "x",
                "completed_steps": [],
                "pending_steps": ["c"],
                "and_gate_arrivals": {},
                "consumed_gates": [],
                "state_data": "{}",
            }
        )
        cp = FlowCheckpoint.from_json(payload)
        # Older checkpoints that lack the field must default gracefully.
        assert cp.pending_step_triggers == {}

    def test_capture_accepts_pending_step_triggers(self) -> None:
        """FlowCheckpoint.capture passes pending_step_triggers through correctly."""
        trigger = FlowTriggerEvent(name="up", source_step="up", kind="step_completion")
        cp = FlowCheckpoint.capture(
            flow_id="x",
            completed_steps=("up",),
            pending_steps=("down",),
            and_gate_arrivals={},
            consumed_gates=set(),
            state_data="{}",
            pending_step_triggers={"down": [trigger]},
        )
        assert "down" in cp.pending_step_triggers
        assert len(cp.pending_step_triggers["down"]) == 1
        assert cp.pending_step_triggers["down"][0].name == "up"

    def test_seed_executor_restores_non_deferred_triggers(self) -> None:
        """_seed_executor_from_checkpoint must restore non-deferred pending triggers.

        The executor's ``pending_triggers`` for non-deferred pending steps
        must be seeded from ``checkpoint.pending_step_triggers`` so that
        ``_build_step_context`` returns the correct ``ctx.triggers`` tuple on
        resume rather than an empty tuple.

        This test targets just the pending_triggers restoration logic by
        verifying the executor's pending_triggers dict is updated.  We bypass
        the table.replace() call by pre-patching the import so a simple
        namespace stands in.
        """
        from dataclasses import dataclass

        from troopai.adk.run.runner import _seed_executor_from_checkpoint

        trigger = FlowTriggerEvent(name="src", source_step="src_step", kind="step_completion")
        cp = FlowCheckpoint(
            flow_id="flow-seed",
            completed_steps=("src_step",),
            pending_steps=("target_step",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            pending_step_triggers={"target_step": (trigger,)},
        )

        # Build a minimal fake executor.  The table must be a real dataclass
        # so dataclasses.replace() works inside _seed_executor_from_checkpoint.
        @dataclass
        class _FakeTable:
            starts: tuple[str, ...]

        class _FakeExecutor:
            def __init__(self) -> None:
                self.pending_triggers: dict[str, list[object]] = {}
                self.completed_steps: list[str] = []
                self.step_count: int = 0
                self.and_arrivals: dict[str, set[str]] = {}
                self.consumed_gates: set[str] = set()
                self.table = _FakeTable(starts=())

        executor = _FakeExecutor()
        _seed_executor_from_checkpoint(executor, cp)

        # The non-deferred pending step's trigger must be in pending_triggers.
        assert "target_step" in executor.pending_triggers, (
            "_seed_executor_from_checkpoint must restore non-deferred pending "
            "triggers from checkpoint.pending_step_triggers"
        )
        restored = executor.pending_triggers["target_step"]
        assert len(restored) == 1
        assert restored[0].name == "src"  # type: ignore[attr-defined]
