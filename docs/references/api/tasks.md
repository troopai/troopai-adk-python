(references/api/tasks)=

# Tasks

Declarative units of work: an agent, a description, and per-call overrides
packaged into named, documented work units executed by the Runner.

## Core

```{eval-rst}
.. autoclass:: troopai.adk.tasks.Task
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.tasks.TaskDependency
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.tasks.TaskInputFilter
```

## Pipelines

```{eval-rst}
.. autoclass:: troopai.adk.tasks.TaskPipeline
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.tasks.TaskPipelineResult
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.tasks.TaskPipelineState
   :members:
   :show-inheritance:
```

## Task groups

```{eval-rst}
.. autoclass:: troopai.adk.tasks.TaskGroup
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.tasks.TaskGroupResult
   :members:
   :show-inheritance:

.. autodata:: troopai.adk.tasks.ErrorPolicy
```

## Input and output

```{eval-rst}
.. autoclass:: troopai.adk.tasks.TaskInputData
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.tasks.TaskOutput
   :members:
   :show-inheritance:
```

## Exceptions

```{eval-rst}
.. autoclass:: troopai.adk.tasks.TaskPipelineDefinitionError
   :members:
   :show-inheritance:
```

Tasks run via `Runner.arun_task`, `Runner.arun_task_pipeline`, and
`Runner.arun_task_group`. Usage lives in the
[Tasks guide](../../tasks/tasks.md).
