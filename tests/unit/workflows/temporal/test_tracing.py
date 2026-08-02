"""Tests for :mod:`troopai.adk.workflows.temporal.tracing`.

Covers:
- ``should_emit_span`` suppresses emission during Temporal workflow replay.
- ``should_emit_span`` emits when workflow is not replaying.
- ``should_emit_span`` emits when called outside a workflow context.
- ``deterministic_timestamp`` uses ``workflow.now()`` inside a workflow.
- ``deterministic_timestamp`` uses ``time.time()`` outside a workflow.
- ``deterministic_uuid`` returns a string inside a workflow.
- ``deterministic_uuid`` returns a string outside a workflow.
- ``deterministic_uuid`` uses ``workflow.uuid4()`` inside a workflow.
- ``deterministic_uuid`` uses ``uuid.uuid4()`` outside a workflow.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("temporalio")

import temporalio as _temporalio_pkg
import temporalio.workflow  # side-effect import: loads submodule so patch.object finds it as a package attribute

from troopai.adk.workflows.temporal.tracing import (
    deterministic_timestamp,
    deterministic_uuid,
    should_emit_span,
)


def _mock_workflow(in_workflow: bool, is_replaying: bool = False) -> MagicMock:
    """Build a ``MagicMock`` mimicking the ``temporalio.workflow`` module."""
    mock_wf = MagicMock()
    mock_wf.in_workflow.return_value = in_workflow
    mock_wf.unsafe = MagicMock()
    mock_wf.unsafe.is_replaying.return_value = is_replaying
    return mock_wf


class TestShouldEmitSpan:
    def test_should_emit_span_suppresses_during_replay(self) -> None:
        """Returns ``False`` when the workflow is replaying."""
        mock_wf = _mock_workflow(in_workflow=True, is_replaying=True)

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = should_emit_span()

        assert result is False

    def test_should_emit_span_emits_outside_replay(self) -> None:
        """Returns ``True`` when inside a workflow that is not replaying."""
        mock_wf = _mock_workflow(in_workflow=True, is_replaying=False)

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = should_emit_span()

        assert result is True

    def test_should_emit_span_emits_outside_workflow(self) -> None:
        """Returns ``True`` when called outside any Temporal workflow context."""
        mock_wf = _mock_workflow(in_workflow=False)

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = should_emit_span()

        assert result is True


class TestDeterministicTimestamp:
    def test_deterministic_timestamp_uses_workflow_now(self) -> None:
        """Uses ``workflow.now()`` when inside a Temporal workflow."""
        fixed_dt = datetime.datetime(2025, 1, 15, 12, 0, 0, tzinfo=datetime.UTC)
        mock_wf = _mock_workflow(in_workflow=True)
        mock_wf.now.return_value = fixed_dt

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = deterministic_timestamp()

        assert result == fixed_dt.timestamp()
        mock_wf.now.assert_called_once()

    def test_deterministic_timestamp_uses_time_outside(self) -> None:
        """Uses ``time.time()`` when outside a Temporal workflow."""
        mock_wf = _mock_workflow(in_workflow=False)

        with (
            patch.object(_temporalio_pkg, "workflow", mock_wf),
            patch("troopai.adk.workflows.temporal.tracing.time") as mock_time,
        ):
            mock_time.time.return_value = 1_700_000_000.0
            result = deterministic_timestamp()

        assert result == 1_700_000_000.0
        mock_time.time.assert_called_once()

    def test_deterministic_timestamp_is_float(self) -> None:
        """Returns a float when outside a workflow."""
        mock_wf = _mock_workflow(in_workflow=False)

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = deterministic_timestamp()

        assert isinstance(result, float)


class TestDeterministicUuid:
    def test_deterministic_uuid_uses_workflow_uuid4(self) -> None:
        """Uses ``workflow.uuid4()`` when inside a Temporal workflow."""
        fake_uuid = "12345678-1234-5678-1234-567812345678"
        mock_wf = _mock_workflow(in_workflow=True)
        mock_wf.uuid4.return_value = fake_uuid

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = deterministic_uuid()

        assert result == fake_uuid
        mock_wf.uuid4.assert_called_once()

    def test_deterministic_uuid_is_string_inside_workflow(self) -> None:
        """Returns a string when called inside a Temporal workflow."""
        mock_wf = _mock_workflow(in_workflow=True)
        mock_wf.uuid4.return_value = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = deterministic_uuid()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_uuid_is_string_outside_workflow(self) -> None:
        """Returns a string when called outside any Temporal workflow."""
        mock_wf = _mock_workflow(in_workflow=False)

        with patch.object(_temporalio_pkg, "workflow", mock_wf):
            result = deterministic_uuid()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_uuid_uses_system_uuid_outside_workflow(self) -> None:
        """Uses :func:`uuid.uuid4` when outside a Temporal workflow."""
        mock_wf = _mock_workflow(in_workflow=False)

        with (
            patch.object(_temporalio_pkg, "workflow", mock_wf),
            patch("troopai.adk.workflows.temporal.tracing.uuid") as mock_uuid_mod,
        ):
            mock_uuid_mod.uuid4.return_value = "system-uuid-value"
            result = deterministic_uuid()

        assert result == "system-uuid-value"
        mock_uuid_mod.uuid4.assert_called_once()
