"""TroopAI Graph — composable multi-agent orchestration primitive.

A :class:`Graph` is a directed graph of :class:`GraphNode`\\ s executed
under BSP (Bulk Synchronous Parallel) supersteps. Nodes may host an
:class:`~troopai.adk.agents.agent.Agent`, a
:class:`~troopai.adk.swarms.swarm.Swarm`, another :class:`Graph`, or a
plain Python ``Callable`` — all uniformly, via the
:class:`~troopai.adk.orchestration.executable.Executable` seam and the
thin adapters in :mod:`troopai.adk.graphs.adapters`.

The public surface is intentionally flat so end-users write
``from troopai.adk.graphs import Graph, Merge, GraphConfig`` without
knowing the internal file layout.

See :doc:`docs/graphs/graphs.md` for the end-to-end tutorial and
:mod:`examples.graphs` for runnable demos.
"""

from __future__ import annotations

from troopai.adk.graphs.adapters import (
    AgentExecutable,
    CallableExecutable,
    CallableNodeFn,
    SwarmExecutable,
    to_executable,
)
from troopai.adk.graphs.builder import GraphBuilder
from troopai.adk.graphs.checkpointer import (
    Checkpointer,
    GraphCheckpoint,
)
from troopai.adk.graphs.checkpointers import InMemoryCheckpointer, SQLiteCheckpointer, TieredCheckpointer
from troopai.adk.graphs.config import (
    GraphConfig,
    NodeInputStrategy,
    NodeRetryPolicy,
)
from troopai.adk.graphs.events import (
    GRAPH_END,
    GRAPH_START,
    NODE_END,
    NODE_ERROR,
    NODE_INTERRUPT,
    NODE_START,
    NODE_STREAM,
    SUPERSTEP_END,
    SUPERSTEP_START,
    GraphEndEvent,
    GraphStartEvent,
    GraphStreamEvent,
    NodeEndEvent,
    NodeErrorEvent,
    NodeInterruptEvent,
    NodeStartEvent,
    NodeStreamEvent,
    SuperstepEndEvent,
    SuperstepStartEvent,
)
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import GraphHooks, HookProvider, HookRegistry
from troopai.adk.graphs.interrupt import (
    NESTED_AGENT_TOOL_APPROVAL_KIND,
    NESTED_GRAPH_INTERRUPT_KIND,
    GraphResume,
    GraphResumeError,
    Interrupt,
    InterruptException,
    NestedAgentApproval,
    NestedAgentDecision,
    NestedAgentInterrupt,
    NestedAgentRejection,
    NestedAgentReply,
    NestedAgentResumeError,
    NestedAgentSerializationError,
    NestedGraphInterrupt,
    request_human_input,
)
from troopai.adk.graphs.join import JoinBarrier, JoinSemantics
from troopai.adk.graphs.merge import DEFAULT_MERGE, Merge, MergeFn
from troopai.adk.graphs.node import EdgeCondition, GraphEdge, GraphNode
from troopai.adk.graphs.node_input import prepare_node_input
from troopai.adk.graphs.result import (
    GraphRunResult,
    GraphRunResultStreaming,
    GraphRunStatus,
    StructuredInterrupts,
)
from troopai.adk.graphs.state import GraphState

# Composition seam — re-exported for convenience so custom node code
# written against the graph module never needs to know about
# ``orchestration`` as a separate package.
from troopai.adk.orchestration.executable import (
    Executable,
    ExecutableInput,
    NodeResult,
)

__all__ = [
    "DEFAULT_MERGE",
    "GRAPH_END",
    "GRAPH_START",
    "NESTED_AGENT_TOOL_APPROVAL_KIND",
    "NESTED_GRAPH_INTERRUPT_KIND",
    "NODE_END",
    "NODE_ERROR",
    "NODE_INTERRUPT",
    "NODE_START",
    "NODE_STREAM",
    "SUPERSTEP_END",
    "SUPERSTEP_START",
    # Adapters (Agent / Swarm / Callable → Executable)
    "AgentExecutable",
    "CallableExecutable",
    "CallableNodeFn",
    # Checkpointer protocol + default impl
    "Checkpointer",
    "EdgeCondition",
    # Composition seam
    "Executable",
    "ExecutableInput",
    # Core primitive
    "Graph",
    "GraphBuilder",
    "GraphCheckpoint",
    "GraphConfig",
    "GraphEdge",
    "GraphEndEvent",
    # Hooks
    "GraphHooks",
    "GraphNode",
    # Interrupt types, human resume payload, and HITL helper
    "GraphResume",
    "GraphResumeError",
    "GraphRunResult",
    "GraphRunResultStreaming",
    "GraphRunStatus",
    "GraphStartEvent",
    # State + result
    "GraphState",
    # Events (streaming)
    "GraphStreamEvent",
    "HookProvider",
    "HookRegistry",
    "InMemoryCheckpointer",
    "Interrupt",
    "InterruptException",
    # Join semantics
    "JoinBarrier",
    "JoinSemantics",
    # Merge strategies
    "Merge",
    "MergeFn",
    "NestedAgentApproval",
    "NestedAgentDecision",
    "NestedAgentInterrupt",
    "NestedAgentRejection",
    "NestedAgentReply",
    "NestedAgentResumeError",
    "NestedAgentSerializationError",
    "NestedGraphInterrupt",
    "NodeEndEvent",
    "NodeErrorEvent",
    # Config knobs
    "NodeInputStrategy",
    "NodeInterruptEvent",
    "NodeResult",
    "NodeRetryPolicy",
    "NodeStartEvent",
    "NodeStreamEvent",
    "SQLiteCheckpointer",
    "StructuredInterrupts",
    "SuperstepEndEvent",
    "SuperstepStartEvent",
    "SwarmExecutable",
    "TieredCheckpointer",
    # Input preparation helpers
    "prepare_node_input",
    "request_human_input",
    "to_executable",
]
