"""Task abstraction — declarative units of work executed by the Runner.

Canonical minimal example — two tasks in a sequential pipeline::

    from troopai.adk import Agent, Runner, Task, TaskPipeline

    researcher = Agent(name="researcher", system_prompt="Research the topic.")
    writer = Agent(name="writer", system_prompt="Write clear prose.")

    gather = Task(description="Gather facts about quantum computing.", agent=researcher)
    summarize = Task(description="Summarize the gathered facts.", agent=writer)

    pipeline = TaskPipeline(tasks=(gather, summarize))
    result = await Runner.arun_task_pipeline(pipeline)

A :class:`Task` packages an agent, a description, and per-call
overrides (guardrails, output schema, budgets) into a single
declarative value. :class:`TaskPipeline` composes multiple tasks
sequentially with explicit (never implicit) context chaining. The
Runner executes both via :meth:`Runner.arun_task` /
:meth:`Runner.arun_task_pipeline`.

This module is purely additive — every existing ``Runner.arun(...)``
call continues to work unchanged. Task is a higher-level convenience
for developers who want named, documented work units with explicit
budgets and guardrails, without inheriting CrewAI's hidden behavior
(no auto-manager-agent, no auto-context-aggregation, no string-
guardrails that silently spawn LLM Agents).

See ``docs/tasks/tasks.md`` for usage and ``examples/tasks/`` for
runnable examples.
"""

from __future__ import annotations

from troopai.adk.tasks.task import Task, TaskDependency, TaskInputFilter
from troopai.adk.tasks.task_group import ErrorPolicy, TaskGroup, TaskGroupResult
from troopai.adk.tasks.task_input_data import TaskInputData
from troopai.adk.tasks.task_output import TaskOutput
from troopai.adk.tasks.task_pipeline import TaskPipeline, TaskPipelineResult
from troopai.adk.tasks.task_pipeline_state import TaskPipelineState
from troopai.adk.tasks.topology import TaskPipelineDefinitionError

__all__ = [
    # Alphabetically sorted (RUF022). Themes, for orientation:
    # core (Task, TaskDependency), composition (TaskPipeline, TaskGroup),
    # input/output (TaskInputData, TaskOutput), errors, persistence.
    "ErrorPolicy",
    "Task",
    "TaskDependency",
    "TaskGroup",
    "TaskGroupResult",
    "TaskInputData",
    "TaskInputFilter",
    "TaskOutput",
    "TaskPipeline",
    "TaskPipelineDefinitionError",
    "TaskPipelineResult",
    "TaskPipelineState",
]
