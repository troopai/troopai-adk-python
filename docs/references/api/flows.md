(references/api/flows)=

# Flows

Decorator-driven multi-step orchestration over typed shared state, with
event-driven listeners and state-based routers.

## Core

```{eval-rst}
.. autoclass:: troopai.adk.flows.Flow
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowMeta
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowStep
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowRole
```

## Decorators

```{eval-rst}
.. autofunction:: troopai.adk.flows.flow_start

.. autofunction:: troopai.adk.flows.flow_listen

.. autofunction:: troopai.adk.flows.flow_router

.. autodata:: troopai.adk.flows.FlowTriggerSpec
```

## Combinators

```{eval-rst}
.. autoclass:: troopai.adk.flows.Or
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.And
   :members:
   :show-inheritance:
```

## Config and results

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowConfig
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowErrorPolicy

.. autoclass:: troopai.adk.flows.FlowRunResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowRunResultStreaming
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowRunStatus
```

## Triggers

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowTriggerEvent
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowTriggerKind

.. autodata:: troopai.adk.flows.FLOW_ERROR_TRIGGER
```

## Events

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowStartEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowEndEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowStepStartEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowStepEndEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowStepErrorEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowStepSkippedEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowStepDeferredEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowStepRejectedEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autoclass:: troopai.adk.flows.FlowRouteEvaluatedEvent
   :members:
   :show-inheritance:
   :exclude-members: type

.. autodata:: troopai.adk.flows.FlowEvent
```

## Approvals and deferral

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowApprovalPolicy
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowApprovalDecision
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowApprovalStatus

.. autodata:: troopai.adk.flows.FlowDeferralKind

.. autoclass:: troopai.adk.flows.FlowDeferredStep
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowAgentDeferred
   :members:
   :show-inheritance:
```

## Step governance

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowStepContext
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowStepGate

.. autoclass:: troopai.adk.flows.FlowStepGuardrails
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowStepGuardrailFn

.. autoclass:: troopai.adk.flows.FlowStepGuardrailVerdict
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowStepCachePolicy
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowCacheKeyFn

.. autoclass:: troopai.adk.flows.FlowStepRateLimit
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.FlowStepRateLimitBehavior
```

## Persistence and distributed execution

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowCheckpoint
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowWorkerBackend
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowBatchClaim
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.InMemoryFlowWorkerBackend
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.SqliteFlowWorkerBackend
   :members:
   :show-inheritance:
```

## Definition and registry

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowDefinition
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.StepInfo
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.GateInfo
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowStepRegistry
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowTransitionTable
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.GateSpec
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.flows.TriggerSpec

.. autofunction:: troopai.adk.flows.build_flow_definition

.. autofunction:: troopai.adk.flows.build_transition_table

.. autofunction:: troopai.adk.flows.collect_step_descriptions
```

## Agent bridge

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowExecutable
   :members:
   :show-inheritance:

.. autofunction:: troopai.adk.flows.arun_flow_agent
```

## Exceptions

```{eval-rst}
.. autoclass:: troopai.adk.flows.FlowDefinitionError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowStepError
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowMaxStepsExceeded
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.flows.FlowCheckpointNotFoundError
   :members:
   :show-inheritance:
```

Flows are executed via `Runner.arun_flow`. The end-to-end walkthrough
lives in the [Flows guide](../../flows/flows.md).
