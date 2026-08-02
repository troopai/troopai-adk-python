"""Swarm — production-ready multi-agent iterative collaboration.

A Swarm is a configuration object describing a roster of agents that
take turns on a shared problem until an explicit termination signal
is raised. Unlike ``Handoff`` (one-shot transfer-of-control) and
``Agent.as_tool()`` (delegate-and-resume), a Swarm supports cycles:
agent A can hand off to B, B can hand off to A, until an
``ExplicitDoneTermination`` triggers or a hard guard trips.

Canonical definition — fluent builder (mirrors ``Graph.new``)::

    swarm = (
        Swarm.new("code-review", description="author → reviewer → security")
        .members(author, reviewer, security)
        .entry("author")
        .llm_handoff()
        .terminate_on(ExplicitDoneTermination() | MaxTurnsTermination(12))
        .with_config(SwarmConfig(max_total_tokens=50_000))
        .compile()
    )
    result = await Runner.arun_swarm(swarm, "Refactor this module.")

Direct construction also works; defaults fill the rest
(``LLMHandoffPolicy`` + ``DEFAULT_TERMINATION``)::

    swarm = Swarm(members=(author, reviewer), entry="author")

Key design principles (see ``docs/swarms/swarms.md`` for the user
guide):

- **Swarm = config, Runner = execution.** No ``Swarm.run()``. The
  driver lives in ``troopai.adk.run.swarm_loop``.
- **No hidden behavior.** No auto-injected system prompts, no
  preambles. Opt-in via ``prompt_with_swarm_instructions()``.
- **No provider-specific wire types.** Shared context uses Layer 1
  (``LLMInputContentItem``) or Layer 3 (``RunItem``) only.
- **Explicit termination.** Never terminate by absence of a tool
  call. The ``swarm_done`` tool must be called explicitly.
- **Pluggable routing.** ``SwarmPolicy`` ABC supports LLM-handoff,
  round-robin, and structured-intent-based routing.

This package re-exports the swarm data types, the builder, policies,
termination conditions, state, shared-context strategy, hooks,
interrupt/resume primitives, checkpointer protocols, the opt-in prompt
helper, and the stream-event vocabulary. The driver lives in
``troopai.adk.run.swarm_loop`` (non-streaming) and
``troopai.adk.run.swarm_loop_streamed`` (event-emitting variant);
runner / builder integration lives in ``troopai.adk.run.runner``.
"""

from __future__ import annotations

from troopai.adk.swarms.builder import SwarmBuilder
from troopai.adk.swarms.checkpointer import (
    SwarmCheckpoint,
    SwarmCheckpointer,
    SwarmHookRegistry,
)
from troopai.adk.swarms.config import SharedContextConfig, SwarmConfig
from troopai.adk.swarms.events import (
    SwarmDoneEvent,
    SwarmEvent,
    SwarmHandoffEvent,
    SwarmStartEvent,
    SwarmTurnEndEvent,
    SwarmTurnInterruptEvent,
    SwarmTurnStartEvent,
)
from troopai.adk.swarms.hooks import HookRegistry, SwarmHooks
from troopai.adk.swarms.interrupt import SwarmResume, request_human_input_in_swarm
from troopai.adk.swarms.policy import (
    CustomPolicy,
    LLMHandoffPolicy,
    RoundRobinPolicy,
    StructuredRoutingPolicy,
    SwarmExtraToolsFn,
    SwarmPolicy,
    SwarmSelector,
)
from troopai.adk.swarms.result import SwarmRunResult, SwarmRunResultStreaming
from troopai.adk.swarms.shared_context import prepare_turn_input
from troopai.adk.swarms.shared_context_strategy import SharedContextStrategy
from troopai.adk.swarms.state import (
    SwarmState,
    SwarmStateDict,
)
from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.swarms.swarm import DEFAULT_MAX_TURNS, DEFAULT_TERMINATION, Swarm
from troopai.adk.swarms.swarm_prompt import (
    RECOMMENDED_SWARM_PROMPT_PREFIX,
    prompt_with_swarm_instructions,
)
from troopai.adk.swarms.termination import (
    AndTermination,
    ExplicitDoneTermination,
    HandoffToTermination,
    MaxTurnsTermination,
    OrTermination,
    TerminationCondition,
    TextMentionTermination,
    TokenBudgetTermination,
)
from troopai.adk.swarms.yield_signal import (
    SWARM_DONE_TOOL_NAME,
    SwarmDone,
    SwarmHandoff,
    SwarmYieldSignal,
)

__all__ = [
    # Sentinels / constants
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TERMINATION",
    "RECOMMENDED_SWARM_PROMPT_PREFIX",
    "SWARM_DONE_TOOL_NAME",
    "AndTermination",
    "CustomPolicy",
    "ExplicitDoneTermination",
    "HandoffToTermination",
    # Hooks + registries
    "HookRegistry",
    "LLMHandoffPolicy",
    "MaxTurnsTermination",
    "OrTermination",
    "RoundRobinPolicy",
    "SharedContextConfig",
    "SharedContextStrategy",
    "StopReason",
    "StructuredRoutingPolicy",
    # Core config object + builder
    "Swarm",
    "SwarmBuilder",
    # Checkpointer protocols
    "SwarmCheckpoint",
    "SwarmCheckpointer",
    "SwarmConfig",
    "SwarmDone",
    "SwarmDoneEvent",
    "SwarmEvent",
    "SwarmExtraToolsFn",
    "SwarmHandoff",
    "SwarmHandoffEvent",
    "SwarmHookRegistry",
    "SwarmHooks",
    # Policies
    "SwarmPolicy",
    # Interrupt / resume
    "SwarmResume",
    "SwarmRunResult",
    "SwarmRunResultStreaming",
    "SwarmSelector",
    "SwarmStartEvent",
    # State + results
    "SwarmState",
    "SwarmStateDict",
    "SwarmTurnEndEvent",
    "SwarmTurnInterruptEvent",
    "SwarmTurnStartEvent",
    # Yield signals
    "SwarmYieldSignal",
    # Termination
    "TerminationCondition",
    "TextMentionTermination",
    "TokenBudgetTermination",
    # Shared context + opt-in prompt
    "prepare_turn_input",
    "prompt_with_swarm_instructions",
    "request_human_input_in_swarm",
]
