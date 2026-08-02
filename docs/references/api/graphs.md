(references/api/graphs)=

# Graphs

State-machine orchestration: a directed graph of nodes executed in
supersteps, with checkpointing, interrupts, and streaming events.

## Core

```{eval-rst}
.. autoclass:: troopai.adk.graphs.Graph
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphBuilder
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphConfig
   :members:
   :show-inheritance:
```

## Nodes and edges

```{eval-rst}
.. autoclass:: troopai.adk.graphs.GraphNode
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphEdge
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.graphs.EdgeCondition

.. autoclass:: troopai.adk.graphs.NodeInputStrategy
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NodeRetryPolicy
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.graphs.prepare_node_input
```

## State and results

```{eval-rst}
.. autoclass:: troopai.adk.graphs.GraphState
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphRunResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphRunResultStreaming
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphRunStatus
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.StructuredInterrupts
   :members:
   :show-inheritance:
```

## Composition seam and adapters

```{eval-rst}
.. autoclass:: troopai.adk.graphs.Executable
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.ExecutableInput
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NodeResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.AgentExecutable
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.SwarmExecutable
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.CallableExecutable
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.graphs.CallableNodeFn

.. autofunction:: troopai.adk.graphs.to_executable
```

## Merge and join

```{eval-rst}
.. autoclass:: troopai.adk.graphs.Merge
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.graphs.MergeFn

.. autofunction:: troopai.adk.graphs.DEFAULT_MERGE

.. autoclass:: troopai.adk.graphs.JoinBarrier
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.JoinSemantics
   :members:
   :show-inheritance:
```

## Checkpointers

```{eval-rst}
.. autoclass:: troopai.adk.graphs.Checkpointer
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphCheckpoint
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.InMemoryCheckpointer
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.SQLiteCheckpointer
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.TieredCheckpointer
   :members:
   :show-inheritance:
```

## Hooks

```{eval-rst}
.. autoclass:: troopai.adk.graphs.GraphHooks
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.HookProvider
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.HookRegistry
   :members:
   :show-inheritance:
```

## Interrupts and resume

```{eval-rst}
.. autoclass:: troopai.adk.graphs.Interrupt
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.InterruptException
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphResume
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphResumeError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NestedGraphInterrupt
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NestedAgentInterrupt
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NestedAgentApproval
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NestedAgentRejection
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NestedAgentReply
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.graphs.NestedAgentDecision

.. autoclass:: troopai.adk.graphs.NestedAgentResumeError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NestedAgentSerializationError
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.graphs.request_human_input

.. autodata:: troopai.adk.graphs.NESTED_AGENT_TOOL_APPROVAL_KIND

.. autodata:: troopai.adk.graphs.NESTED_GRAPH_INTERRUPT_KIND
```

## Events

```{eval-rst}
.. autoclass:: troopai.adk.graphs.GraphStreamEvent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.GraphEndEvent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NodeStartEvent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NodeEndEvent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NodeErrorEvent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.NodeStreamEvent
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.graphs.SuperstepStartEvent
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.graphs.GRAPH_START

.. autodata:: troopai.adk.graphs.GRAPH_END

.. autodata:: troopai.adk.graphs.NODE_START

.. autodata:: troopai.adk.graphs.NODE_END

.. autodata:: troopai.adk.graphs.NODE_ERROR

.. autodata:: troopai.adk.graphs.NODE_INTERRUPT

.. autodata:: troopai.adk.graphs.NODE_STREAM

.. autodata:: troopai.adk.graphs.SUPERSTEP_START

.. autodata:: troopai.adk.graphs.SUPERSTEP_END
```

Three further `GraphStreamEvent` subclasses are documented in prose
because their source docstrings do not render through autodoc:

- `GraphStartEvent` — emitted once at the top of a graph run, before
  the first superstep. Keys: `type` (always `GRAPH_START`),
  `graph_path`, `graph_id`, `description`, `entry_node`,
  `terminal_nodes`.
- `SuperstepEndEvent` — emitted after a superstep completes. Keys:
  `type` (always `SUPERSTEP_END`), `graph_path`, `superstep`,
  `fired_nodes`, `errored_nodes`.
- `NodeInterruptEvent` — a node raised `InterruptException` and the
  run is suspending; carries the pending `Interrupt` so consumers can
  prompt the human and resume via `GraphResume`. Keys: `type` (always
  `NODE_INTERRUPT`), `graph_path`, `node_id`, `interrupt`.

Graphs are executed via `Runner.arun_graph`. The end-to-end walkthrough
lives in the [Graphs guide](../../graphs/graphs.md).
