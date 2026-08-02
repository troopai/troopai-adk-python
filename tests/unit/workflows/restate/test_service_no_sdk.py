"""TroopAIRestateService contract tests that run WITHOUT the restate SDK.

The WorkflowContext guard in ``wait_for_human_reply`` is pure Python (operates
on ``ctx: Any``), so these tests must NOT ``importorskip('restate')`` — the
module imports fine without the SDK.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.workflows.restate.service import TroopAIRestateService


class TestWaitForHumanReplyRequiresWorkflowContext:
    """Durable promises (``ctx.promise``) are a workflow-only primitive.

    Regression: the method awaited ``ctx.promise(...)`` while the docstring
    advertised a plain service ``restate.Context``. A service Context has no
    ``promise``, so a caller following the old docs hit an obscure
    AttributeError deep inside the await. The guard now fails fast with an
    actionable message.
    """

    async def test_missing_promise_raises_type_error(self) -> None:
        """A plain service Context (no ``.promise``) is rejected with TypeError."""
        service = TroopAIRestateService()

        class _ServiceContext:
            """Stand-in service Context: no durable-promise support."""

        with pytest.raises(TypeError, match="WorkflowContext"):
            await service.wait_for_human_reply(_ServiceContext())

    async def test_workflow_context_with_promise_is_accepted(self) -> None:
        """A context exposing ``.promise`` (a WorkflowContext) still works."""
        promise = MagicMock()
        promise.value = AsyncMock(return_value={"node_id": "n", "value": "v"})
        ctx = MagicMock()
        ctx.promise = MagicMock(return_value=promise)

        service = TroopAIRestateService()
        reply = await service.wait_for_human_reply(ctx)

        assert reply.node_id == "n"
        assert reply.value == "v"
        ctx.promise.assert_called_once_with("human_reply")
