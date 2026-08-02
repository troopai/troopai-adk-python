"""Tests for :class:`~troopai.adk.workflows.restate.service.TroopAIRestateService`
and :class:`~troopai.adk.workflows.restate.service.RestateHumanReply`.

Covers:
- ``RestateHumanReply`` is a frozen dataclass with the expected fields.
- ``TroopAIRestateService`` exposes a ``wait_for_human_reply`` method.

All tests skip when ``restate`` is not installed.
"""

from __future__ import annotations

import dataclasses

import pytest

restate = pytest.importorskip("restate")
# isort: split
from troopai.adk.workflows.restate.service import RestateHumanReply, TroopAIRestateService

# ---------------------------------------------------------------------------
# Tests: RestateHumanReply
# ---------------------------------------------------------------------------


class TestRestateHumanReplyDataclass:
    def test_restate_human_reply_dataclass(self) -> None:
        """``RestateHumanReply`` stores node_id, value, and metadata correctly."""
        reply = RestateHumanReply(
            node_id="node-42",
            value="approved",
            metadata={"reviewer": "alice"},
        )

        assert reply.node_id == "node-42"
        assert reply.value == "approved"
        assert reply.metadata == {"reviewer": "alice"}

    def test_restate_human_reply_is_frozen(self) -> None:
        """``RestateHumanReply`` is a frozen dataclass (assignment raises)."""
        reply = RestateHumanReply(node_id="n1", value="ok")

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            reply.value = "mutated"  # type: ignore[misc]

    def test_restate_human_reply_default_metadata(self) -> None:
        """``RestateHumanReply`` defaults ``metadata`` to an empty dict."""
        reply = RestateHumanReply(node_id="n2", value="yes")

        assert reply.metadata == {}

    def test_restate_human_reply_is_dataclass(self) -> None:
        """``RestateHumanReply`` is recognized as a dataclass by the stdlib."""
        assert dataclasses.is_dataclass(RestateHumanReply)

    def test_restate_human_reply_fields(self) -> None:
        """``RestateHumanReply`` exposes the three expected dataclass fields."""
        field_names = {f.name for f in dataclasses.fields(RestateHumanReply)}
        assert field_names == {"node_id", "value", "metadata"}


# ---------------------------------------------------------------------------
# Tests: TroopAIRestateService
# ---------------------------------------------------------------------------


class TestTroopAIRestateServiceExists:
    def test_troopai_restate_service_exists(self) -> None:
        """``TroopAIRestateService`` is importable and has ``wait_for_human_reply``."""
        assert hasattr(TroopAIRestateService, "wait_for_human_reply")

    def test_wait_for_human_reply_is_callable(self) -> None:
        """``wait_for_human_reply`` is callable on the class."""
        assert callable(TroopAIRestateService.wait_for_human_reply)

    async def test_wait_for_human_reply_resolves_promise(self) -> None:
        """``wait_for_human_reply`` awaits ctx.promise(name).value() and builds a reply."""
        from unittest.mock import AsyncMock, MagicMock

        promise_mock = MagicMock()
        promise_mock.value = AsyncMock(
            return_value={
                "node_id": "step-7",
                "value": "proceed",
                "metadata": {"ts": "2026-01-01"},
            }
        )

        ctx = MagicMock()
        ctx.promise = MagicMock(return_value=promise_mock)

        service = TroopAIRestateService()
        reply = await service.wait_for_human_reply(ctx, promise_name="approval")

        ctx.promise.assert_called_once_with("approval")
        promise_mock.value.assert_awaited_once()
        assert isinstance(reply, RestateHumanReply)
        assert reply.node_id == "step-7"
        assert reply.value == "proceed"
        assert reply.metadata == {"ts": "2026-01-01"}

    async def test_wait_for_human_reply_default_promise_name(self) -> None:
        """``wait_for_human_reply`` uses ``'human_reply'`` as the default promise name."""
        from unittest.mock import AsyncMock, MagicMock

        promise_mock = MagicMock()
        promise_mock.value = AsyncMock(return_value={"node_id": "n", "value": "v"})

        ctx = MagicMock()
        ctx.promise = MagicMock(return_value=promise_mock)

        service = TroopAIRestateService()
        await service.wait_for_human_reply(ctx)

        ctx.promise.assert_called_once_with("human_reply")
