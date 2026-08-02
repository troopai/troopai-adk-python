(references/api/swarms)=

# Swarms

Multi-agent iterative collaboration: a roster of agents taking turns on a
shared problem until an explicit termination signal fires.

## Core

```{eval-rst}
.. autoclass:: troopai.adk.swarms.Swarm
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmBuilder
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmConfig
   :members:
   :show-inheritance:
```

## Policies

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SwarmPolicy
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.LLMHandoffPolicy
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.RoundRobinPolicy
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.StructuredRoutingPolicy
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.CustomPolicy
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.swarms.SwarmSelector

.. autodata:: troopai.adk.swarms.SwarmExtraToolsFn
```

## Termination

```{eval-rst}
.. autoclass:: troopai.adk.swarms.TerminationCondition
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.ExplicitDoneTermination
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.MaxTurnsTermination
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.TokenBudgetTermination
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.TextMentionTermination
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.HandoffToTermination
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.AndTermination
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.OrTermination
   :members:
   :show-inheritance:
```

## State and results

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SwarmState
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmStateDict
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmRunResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmRunResultStreaming
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.StopReason
   :members:
   :show-inheritance:
```

## Events

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SwarmStartEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.swarms.SwarmTurnStartEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.swarms.SwarmTurnEndEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.swarms.SwarmTurnInterruptEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.swarms.SwarmHandoffEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.swarms.SwarmDoneEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autodata:: troopai.adk.swarms.SwarmEvent
```

## Yield signals

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SwarmDone
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmHandoff
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.swarms.SwarmYieldSignal
```

## Hooks and checkpoints

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SwarmHooks
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.HookRegistry
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmHookRegistry
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmCheckpoint
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SwarmCheckpointer
   :members:
   :show-inheritance:
```

## Interrupt and resume

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SwarmResume
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.swarms.request_human_input_in_swarm
```

## Shared context

```{eval-rst}
.. autoclass:: troopai.adk.swarms.SharedContextConfig
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.swarms.SharedContextStrategy
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.swarms.prepare_turn_input

.. autofunction:: troopai.adk.swarms.prompt_with_swarm_instructions
```

## Constants

```{eval-rst}
.. autodata:: troopai.adk.swarms.DEFAULT_MAX_TURNS

.. autodata:: troopai.adk.swarms.DEFAULT_TERMINATION

.. autodata:: troopai.adk.swarms.RECOMMENDED_SWARM_PROMPT_PREFIX

.. autodata:: troopai.adk.swarms.SWARM_DONE_TOOL_NAME
```

Swarms are executed via `Runner.arun_swarm`. The end-to-end walkthrough
lives in the [Swarms guide](../../swarms/swarms.md).
