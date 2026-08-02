"""Middleware configuration object for an :class:`~troopai.adk.agents.agent.Agent`.

Holds typed lists of middleware grouped by execution layer. Four slots
are wired today:

- ``tools`` — function-scope middleware around ``tool.on_invoke``.
- ``agents`` — agent-turn middleware around each per-agent block in
  the run loop. Applies on streaming AND non-streaming runs.
- ``llms`` — non-streaming LLM-call middleware around each
  ``LLM.acomplete()`` invocation.
- ``stream_llms`` — streaming LLM-call middleware around each
  ``LLM.acomplete(stream=True)`` invocation. Sibling Protocol with a
  non-overlapping return type (``AsyncIterator[LLMStreamEvent]``
  rather than ``LLMResponse``); see
  :class:`~troopai.adk.llms.llm_stream_middleware.LLMStreamMiddleware`.

All four layers share the same plumbing-only contract — see the
Protocol docstrings under ``troopai.adk.tools.tool_middleware``,
``troopai.adk.run.agent_middleware``,
``troopai.adk.llms.llm_middleware``, and
``troopai.adk.llms.llm_stream_middleware`` for the
forbidden-vs-allowed rules.

Routing each layer through a separately-typed slot — instead of a
polymorphic ``list[Any]`` — lets mypy and pyright catch any attempt
to register a tool middleware where an agent or LLM middleware is
expected.

Lives next to ``agents/agent_guardrails.py`` because both are
agent-config types: a developer reading an Agent definition finds
every config surface in ``agents/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only imports — `from __future__ import annotations` keeps
    # `list[...Middleware]` strings at runtime, so this branch is never
    # evaluated outside type checking.
    from troopai.adk.llms.llm_middleware import LLMMiddleware
    from troopai.adk.llms.llm_stream_middleware import LLMStreamMiddleware
    from troopai.adk.run.agent_middleware import AgentMiddleware
    from troopai.adk.tools.tool_middleware import ToolMiddleware


@dataclass
class Middleware:
    """Per-layer middleware lists registered on an Agent.

    Each slot is typed against its own Protocol so the type checker
    rejects layer mixing.

    Attributes:
        tools: Tool-execution middleware applied around every
            ``tool.on_invoke`` call. Composed outer-to-inner: the first
            entry runs first (outermost), the last entry runs last
            (innermost, just before the tool's own invoker). See
            :class:`~troopai.adk.tools.tool_middleware.ToolMiddleware` for
            the forbidden-vs-allowed contract.
        agents: Agent-turn middleware applied around each per-agent
            block in the loop (re-fires on every handoff / swarm
            transition). Same outer-to-inner composition. Applies on
            streaming and non-streaming runs alike. See
            :class:`~troopai.adk.run.agent_middleware.AgentMiddleware`.
        llms: Non-streaming LLM-call middleware applied around each
            ``LLM.acomplete()`` call the runner issues for this
            agent. Same outer-to-inner composition. See
            :class:`~troopai.adk.llms.llm_middleware.LLMMiddleware`.
        stream_llms: Streaming LLM-call middleware applied around each
            ``LLM.acomplete(stream=True)`` call. Sibling Protocol with
            a non-overlapping return type (``AsyncIterator[LLMStreamEvent]``).
            Use :func:`~troopai.adk.llms.llm_stream_middleware.make_logging_middlewares`
            to register one paired logger across both LLM paths.
    """

    tools: list[ToolMiddleware] = field(default_factory=list)
    """Tool-execution middleware. See class docstring for the contract."""

    agents: list[AgentMiddleware] = field(default_factory=list)
    """Agent-turn middleware. Wraps each per-agent block in the run loop."""

    llms: list[LLMMiddleware] = field(default_factory=list)
    """Non-streaming LLM-call middleware. Wraps each ``LLM.acomplete()`` call.

    Streaming calls go through ``stream_llms`` instead — registering
    a non-streaming middleware here while running ``Runner.arun(stream=True)``
    triggers a one-time ``logger.warning`` per streaming call so users
    notice the path mismatch.
    """

    stream_llms: list[LLMStreamMiddleware] = field(default_factory=list)
    """Streaming LLM-call middleware. Wraps each ``LLM.acomplete(stream=True)`` call.

    Sibling Protocol of ``LLMMiddleware`` with a non-overlapping
    return type (``AsyncIterator[LLMStreamEvent]``). The two slots
    exist because Protocol satisfaction depends on ``__call__``'s
    return-type identity — a single class cannot structurally
    satisfy both. See
    :func:`~troopai.adk.llms.llm_stream_middleware.make_logging_middlewares`
    for the paired-registration helper.
    """
