"""``A2AServer`` — frozen config object pairing a local Agent with an AgentCard.

``A2AServer`` is intentionally **not** a running server. It is a frozen
``@dataclass`` that pairs an :class:`troopai.adk.agents.Agent` with a
manually-authored :class:`a2a.types.AgentCard`. The companion factory
:func:`build_starlette_app` consumes the config and returns a Starlette
ASGI app the developer's own ASGI runtime (``uvicorn``, ``hypercorn``,
``granian``, …) serves.

This split is deliberate. The ADK does not own:

* The ASGI runtime — choosing between uvicorn / hypercorn / granian is a
  deployment concern, not a framework concern.
* The process lifecycle — graceful shutdown, signal handling, worker
  count, port binding all live with the operator.
* The reverse-proxy / TLS termination — these are infra, not framework.

The split mirrors the framework's config/execution separation:
``A2AServer`` is the config; uvicorn (or your choice) is the execution.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

try:
    # AgentCard is the public discovery contract — every A2A server
    # MUST publish one. Authored manually by the developer (Microsoft
    # pattern) so every field in the card is intentional.
    from a2a.types import AgentCard
except ImportError as ie:
    if ie.name is not None and ie.name != "a2a" and not ie.name.startswith("a2a."):
        # A transitive dependency of an installed a2a-sdk failed — surface
        # the real error instead of mislabeling it "extra not installed".
        raise
    raise ImportError(
        "Please install the 'a2a' extra to use A2A protocol support. Run: pip install 'troopai-adk-python[a2a]'",
        # name="a2a" lets optional-extra guards (e.g. the graphs adapter's
        # A2A fallthrough) recognize the missing extra and degrade gracefully.
        name="a2a",
    ) from ie

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext

    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig


# Default agent loop turn budget when the developer doesn't override.
# Bound is reasonable for a single A2A task — a multi-step research or
# tool-using agent typically needs <10 turns; runaway loops should
# trip MaxTurnsExceeded before a human notices.
_DEFAULT_MAX_TURNS: int = 10


@dataclasses.dataclass(frozen=True, kw_only=True)
class A2AServer:
    """Frozen config object exposing a local :class:`Agent` over A2A.

    Attributes:
        agent: The local :class:`Agent` to expose. The agent's tools,
            guardrails, and instructions remain entirely local — the
            A2A boundary translates incoming task requests into
            ``Runner.arun(agent, prompt)`` calls and translates the
            resulting :class:`RunResult` back into A2A artifacts.
        agent_card: Manually-authored :class:`a2a.types.AgentCard`.
            Includes the agent's identity (``name``, ``description``,
            ``url``, ``version``), capabilities (``streaming``,
            ``push_notifications``), authentication requirements, and
            skills list. The card is published at
            ``/.well-known/agent-card.json`` and is the discovery
            contract every A2A client reads before sending a task.
        task_store: Optional :class:`a2a.server.tasks.TaskStore`. When
            ``None`` (default), :func:`build_starlette_app` constructs
            an :class:`InMemoryTaskStore`. **Production callers MUST
            supply a persistent store** (e.g.
            :class:`a2a.server.tasks.DatabaseTaskStore`) — in-memory
            storage loses tasks on server restart, which breaks the
            :class:`A2AContinuationToken` resume contract for any
            background task that outlives the process.
        executor_task_store: Optional framework
            :class:`~troopai.adk.a2a.task_store.TaskStore` for
            :class:`~troopai.adk.a2a.executor.A2AExecutor` persistence.
            When set, the executor writes each task's initial and terminal
            snapshot here, and a
            :class:`~troopai.adk.a2a.task_store.SQLiteTaskStore`'s
            ``recover_on_startup`` can be called before the server accepts
            requests to mark unfinished tasks from a prior process as
            FAILED. ``None`` (default) gives the executor an
            :class:`~troopai.adk.a2a.task_store.InMemoryTaskStore` — no
            restart recovery. This is distinct from ``task_store``, which
            is the a2a-sdk store handed to ``DefaultRequestHandler`` for
            wire-protocol task lookups.
        max_turns: Maximum local-agent loop turns per A2A task. Maps
            directly to ``Runner.arun(max_turns=...)``. Default 10.
        run_config: Optional :class:`RunConfig` override. When ``None``,
            the executor lets ``Runner.arun`` use framework defaults.
            Use this to plumb usage limits, history processors, or
            tracing metadata that should apply to every task served by
            this endpoint.
        rpc_url: HTTP path the JSON-RPC dispatcher mounts at. Default
            ``"/"`` per the A2A spec's recommended convention. Override
            only if you need to host the A2A endpoint alongside other
            routes on the same Starlette app and want a sub-path.
        compat_earlier_protocol: When ``True``, pass
            ``enable_v0_3_compat=True`` to the a2a-sdk's
            ``create_jsonrpc_routes`` factory, activating the SDK's
            dispatch shim for clients that speak the earlier v0.3 wire
            protocol. Default ``False`` leaves only the primary
            protocol route active. This is an opt-in flag; it is the
            caller's responsibility to ensure the installed a2a-sdk
            exposes ``enable_v0_3_compat``.
        extended_agent_card: Optional second :class:`a2a.types.AgentCard`
            passed as ``extended_agent_card`` to
            :class:`a2a.server.request_handlers.DefaultRequestHandler`.
            When set, authenticated clients that request the extended
            card receive this richer card (e.g. with additional skills
            or capability hints not shown to anonymous callers). ``None``
            (default) disables the extended-card endpoint.
        extended_card_modifier: Optional async callback passed as
            ``extended_card_modifier`` to
            :class:`a2a.server.request_handlers.DefaultRequestHandler`.
            The callback receives the extended :class:`AgentCard` and
            the :class:`a2a.server.context.ServerCallContext` and must
            return a (possibly transformed) :class:`AgentCard`. Useful
            for stamping per-caller metadata or scrubbing fields based
            on request context. ``None`` (default) disables the
            modifier; the extended card is returned verbatim.
        card_modifier: Optional async callback passed as
            ``card_modifier`` to the a2a-sdk's
            ``create_agent_card_routes`` factory.  The callback receives
            the public :class:`AgentCard` and must return a (possibly
            transformed) :class:`AgentCard`.  Useful for dynamically
            patching the public card (e.g. injecting a live
            ``url`` field at request time).  ``None`` (default) serves
            the card verbatim.
    """

    agent: Agent[Any]
    """The local Agent to expose over A2A."""

    agent_card: AgentCard
    """Manually-authored AgentCard published at the well-known URL."""

    task_store: Any | None = None
    """Optional persistent TaskStore — None falls back to InMemoryTaskStore."""

    executor_task_store: Any | None = None
    """Optional framework TaskStore for A2AExecutor persistence.

    When set, A2AExecutor persists task snapshots here and a
    SQLiteTaskStore's recover_on_startup can run before the server
    accepts requests. None (default) gives the executor an
    InMemoryTaskStore — no restart recovery. Distinct from
    ``task_store`` (the a2a-sdk store for DefaultRequestHandler).
    """

    max_turns: int = _DEFAULT_MAX_TURNS
    """Per-task agent loop budget passed to Runner.arun(max_turns=...)."""

    run_config: RunConfig | None = None
    """Optional RunConfig override applied to every task served."""

    rpc_url: str = "/"
    """HTTP path the JSON-RPC dispatcher mounts at."""

    compat_earlier_protocol: bool = False
    """Opt-in dispatch shim for clients on the earlier A2A v0.3 wire protocol.

    Maps to ``enable_v0_3_compat`` on the a2a-sdk's ``create_jsonrpc_routes``
    factory.  ``False`` (default) leaves only the primary protocol route active.
    """

    extended_agent_card: AgentCard | None = None
    """Richer card served to authenticated callers; None disables the extended endpoint.

    Passed as ``extended_agent_card`` to DefaultRequestHandler.
    """

    extended_card_modifier: Callable[[AgentCard, ServerCallContext], Awaitable[AgentCard]] | None = None
    """Async transform applied to the extended card per request.

    Passed as ``extended_card_modifier`` to DefaultRequestHandler.
    None (default) serves the extended card verbatim.
    """

    card_modifier: Callable[[AgentCard], Awaitable[AgentCard]] | None = None
    """Async transform applied to the public card per request.

    Passed as ``card_modifier`` to create_agent_card_routes.
    None (default) serves the public card verbatim.
    """
