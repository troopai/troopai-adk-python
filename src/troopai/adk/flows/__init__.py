"""Flow primitive — decorator-driven multi-step orchestration over typed shared state.

A :class:`Flow` is a class-based, declarative orchestration that composes
:class:`~troopai.adk.agents.agent.Agent`, :class:`~troopai.adk.swarms.swarm.Swarm`,
:class:`~troopai.adk.graphs.graph.Graph`, and :class:`~troopai.adk.tasks.task.Task`
calls as ordered steps, with typed shared state, event-driven listeners,
and state-based routers. Fills the gap between the existing
:class:`~troopai.adk.graphs.graph.Graph` (DAG with message-threading) and
:class:`~troopai.adk.tasks.task_pipeline.TaskPipeline` (sequential with no
typed shared state).

Canonical minimal example — a two-step Flow over a Pydantic state::

    from pydantic import BaseModel

    from troopai.adk import Runner
    from troopai.adk.flows import Flow, flow_listen, flow_start


    class ResearchState(BaseModel):
        topic: str = ""
        summary: str = ""


    class ResearchFlow(Flow[ResearchState]):
        state_factory = ResearchState

        @flow_start
        async def kickoff(self) -> None:
            self.state.topic = "climate"

        @flow_listen(kickoff)
        async def summarize(self) -> None:
            self.state.summary = f"Summary of {self.state.topic}."


    flow = ResearchFlow()  # or ResearchFlow(initial_state=ResearchState(topic="ml"))
    result = await Runner.arun_flow(flow)

**Anti-hidden-behavior contract**: every wire is declared by an explicit
decorator on a method; step methods take only ``self``; ``self.state`` is
the developer's mutable typed object; persistence is explicit via
:class:`~troopai.adk.flows.checkpoint.FlowCheckpoint`. The framework
NEVER auto-injects arguments, auto-persists state, auto-routes on bare
string returns, or auto-instantiates state from the generic parameter.

**Combinators are operator-only**: use ``method_a | method_b`` /
``method_a & method_b``. There are no ``or_()`` / ``and_()`` helper
functions in this ADK — those CrewAI helpers are intentionally omitted
in favor of the fluent operator API.

The name ``Flow`` (rather than ``Workflow``) reserves the latter name
for the future Temporal-style durable execution layer, which composes
*over* this orchestration topology.

See ``docs/flows/flows.md`` for usage and ``examples/flows/`` for runnable
examples.
"""

from __future__ import annotations

from troopai.adk.flows.agent_bridge import arun_flow_agent
from troopai.adk.flows.approval_policy import FlowApprovalPolicy
from troopai.adk.flows.checkpoint import FlowCheckpoint
from troopai.adk.flows.combinators import And, Or
from troopai.adk.flows.config import FlowConfig, FlowErrorPolicy
from troopai.adk.flows.decorators import FlowTriggerSpec, flow_listen, flow_router, flow_start
from troopai.adk.flows.deferred import (
    FlowApprovalDecision,
    FlowApprovalStatus,
    FlowDeferralKind,
    FlowDeferredStep,
)
from troopai.adk.flows.definition import (
    FlowDefinition,
    GateInfo,
    StepInfo,
    build_flow_definition,
)
from troopai.adk.flows.events import (
    FlowEndEvent,
    FlowEvent,
    FlowRouteEvaluatedEvent,
    FlowStartEvent,
    FlowStepDeferredEvent,
    FlowStepEndEvent,
    FlowStepErrorEvent,
    FlowStepRejectedEvent,
    FlowStepSkippedEvent,
    FlowStepStartEvent,
)
from troopai.adk.flows.exceptions import (
    FlowAgentDeferred,
    FlowCheckpointNotFoundError,
    FlowDefinitionError,
    FlowMaxStepsExceeded,
    FlowStepError,
)
from troopai.adk.flows.executable import FlowExecutable
from troopai.adk.flows.flow import Flow, FlowMeta, collect_step_descriptions
from troopai.adk.flows.flow_wrappers import FlowRole, FlowStep
from troopai.adk.flows.registry import (
    FlowStepRegistry,
    FlowTransitionTable,
    GateSpec,
    TriggerSpec,
    build_transition_table,
)
from troopai.adk.flows.result import FlowRunResult, FlowRunResultStreaming, FlowRunStatus
from troopai.adk.flows.sqlite_worker_backend import SqliteFlowWorkerBackend
from troopai.adk.flows.step_cache_policy import FlowCacheKeyFn, FlowStepCachePolicy
from troopai.adk.flows.step_context import FlowStepContext, FlowStepGate
from troopai.adk.flows.step_guardrails import (
    FlowStepGuardrailFn,
    FlowStepGuardrails,
    FlowStepGuardrailVerdict,
)
from troopai.adk.flows.step_rate_limit import (
    FlowStepRateLimit,
    FlowStepRateLimitBehavior,
)
from troopai.adk.flows.triggers import FLOW_ERROR_TRIGGER, FlowTriggerEvent, FlowTriggerKind
from troopai.adk.flows.worker_backend import (
    FlowBatchClaim,
    FlowWorkerBackend,
    InMemoryFlowWorkerBackend,
)

__all__ = [
    # Alphabetically sorted (RUF022). Themes, for orientation:
    # core (Flow, FlowStep), decorators (flow_*), combinators (Or, And),
    # config & result, events, HITL & deferral, step governance,
    # triggers, distributed execution, exceptions, definition/registry.
    "FLOW_ERROR_TRIGGER",
    "And",
    "Flow",
    "FlowAgentDeferred",
    "FlowApprovalDecision",
    "FlowApprovalPolicy",
    "FlowApprovalStatus",
    "FlowBatchClaim",
    "FlowCacheKeyFn",
    "FlowCheckpoint",
    "FlowCheckpointNotFoundError",
    "FlowConfig",
    "FlowDeferralKind",
    "FlowDeferredStep",
    "FlowDefinition",
    "FlowDefinitionError",
    "FlowEndEvent",
    "FlowErrorPolicy",
    "FlowEvent",
    "FlowExecutable",
    "FlowMaxStepsExceeded",
    "FlowMeta",
    "FlowRole",
    "FlowRouteEvaluatedEvent",
    "FlowRunResult",
    "FlowRunResultStreaming",
    "FlowRunStatus",
    "FlowStartEvent",
    "FlowStep",
    "FlowStepCachePolicy",
    "FlowStepContext",
    "FlowStepDeferredEvent",
    "FlowStepEndEvent",
    "FlowStepError",
    "FlowStepErrorEvent",
    "FlowStepGate",
    "FlowStepGuardrailFn",
    "FlowStepGuardrailVerdict",
    "FlowStepGuardrails",
    "FlowStepRateLimit",
    "FlowStepRateLimitBehavior",
    "FlowStepRegistry",
    "FlowStepRejectedEvent",
    "FlowStepSkippedEvent",
    "FlowStepStartEvent",
    "FlowTransitionTable",
    "FlowTriggerEvent",
    "FlowTriggerKind",
    "FlowTriggerSpec",
    "FlowWorkerBackend",
    "GateInfo",
    "GateSpec",
    "InMemoryFlowWorkerBackend",
    "Or",
    "SqliteFlowWorkerBackend",
    "StepInfo",
    "TriggerSpec",
    "arun_flow_agent",
    "build_flow_definition",
    "build_transition_table",
    "collect_step_descriptions",
    "flow_listen",
    "flow_router",
    "flow_start",
]
