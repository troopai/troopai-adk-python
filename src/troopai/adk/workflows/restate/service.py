"""TroopAI Restate service helpers and human-in-the-loop support.

Provides :class:`TroopAIRestateService` — a mixin/base class with helpers
for common Restate service patterns — and :class:`RestateHumanReply` — a
frozen dataclass carrying a human-provided reply resolved via a Restate
promise.

The human-in-the-loop pattern maps Temporal signals to Restate promises:

- Temporal: ``workflow.wait_for_signal("human_reply")``
- Restate: ``ctx.promise("human_reply").value()``

References:
    Restate Python SDK promises (durable promises):
    https://docs.restate.dev/develop/python/durable-execution#promises
    Restate human-in-the-loop pattern:
    https://docs.restate.dev/patterns-recipes/human-in-the-loop
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RestateHumanReply:
    """A human-provided reply received via a Restate durable promise.

    Constructed by :meth:`TroopAIRestateService.wait_for_human_reply` from
    the value resolved by ``ctx.promise(promise_name).value()``.

    Attributes:
        node_id: Identifier of the graph node or workflow step awaiting
            the reply.
        value: The human-provided reply text or encoded value.
        metadata: Optional key-value metadata attached to the reply
            (e.g. timestamps, author info, approval tokens).
    """

    node_id: str
    """Identifier of the graph node or workflow step awaiting the reply."""

    value: str
    """The human-provided reply text or encoded value."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional key-value metadata attached to the reply."""


class TroopAIRestateService:
    """Mixin providing common Restate service helpers for TroopAI workflows.

    A concrete service can hold this mixin to gain human-in-the-loop
    primitives built on Restate durable promises.

    Durable promises (``ctx.promise``) are a **workflow-only** primitive:
    the ``restate.WorkflowContext`` / ``restate.WorkflowSharedContext`` expose
    ``promise()``; the plain ``restate.Context`` of a service handler does
    not.  :meth:`wait_for_human_reply` must therefore run inside a
    ``restate.Workflow`` handler that receives a ``WorkflowContext``.

    Example::

        import restate

        helpers = TroopAIRestateService()
        agent = restate.Workflow("agent")


        @agent.main()
        async def run(ctx: restate.WorkflowContext, req: dict) -> dict:
            # ctx is a WorkflowContext, so ctx.promise is available.
            reply = await helpers.wait_for_human_reply(ctx)
            return {"reply": reply.value}

    References:
        Restate durable promises:
        https://docs.restate.dev/develop/python/durable-execution#promises
    """

    async def wait_for_human_reply(
        self,
        ctx: Any,
        promise_name: str = "human_reply",
    ) -> RestateHumanReply:
        """Wait for a human reply delivered via a Restate durable promise.

        Blocks (durably) until the promise named *promise_name* is resolved
        by an external call (e.g. a webhook or admin API endpoint that
        calls ``ctx.resolve_promise()``).  Restate journals the resolved
        value so that replays return immediately without blocking again.

        Args:
            ctx: The active Restate ``WorkflowContext``.  Durable promises
                (``ctx.promise``) are a workflow-only primitive — a plain
                service ``Context`` does not expose them, so this must be
                called from a ``@restate.workflow`` handler.
            promise_name: Name of the durable promise to wait on.
                Defaults to ``"human_reply"``.

        Returns:
            A :class:`RestateHumanReply` constructed from the promise payload.
            The payload must be a dict with at least ``"node_id"`` and
            ``"value"`` keys.

        Raises:
            TypeError: When *ctx* does not expose ``promise`` — i.e. it is a
                plain service ``Context`` rather than a ``WorkflowContext``.

        References:
            Restate durable promises:
            https://docs.restate.dev/develop/python/durable-execution#promises
        """
        if not hasattr(ctx, "promise"):
            raise TypeError(
                "wait_for_human_reply requires a Restate WorkflowContext: durable "
                "promises (ctx.promise) are a workflow-only primitive and are not "
                "available on a plain service Context. Declare the handler on a "
                "@restate.workflow class."
            )
        logger.info(
            "TroopAIRestateService.wait_for_human_reply: awaiting promise %r",
            promise_name,
        )
        raw: dict[str, Any] = await ctx.promise(promise_name).value()
        logger.debug(
            "TroopAIRestateService.wait_for_human_reply: promise %r resolved, node_id=%r",
            promise_name,
            raw.get("node_id"),
        )
        return RestateHumanReply(
            node_id=raw.get("node_id", ""),
            value=raw.get("value", ""),
            metadata=raw.get("metadata", {}),
        )
