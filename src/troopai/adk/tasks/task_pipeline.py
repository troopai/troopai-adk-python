"""TaskPipeline + TaskPipelineResult — sequential composition of Tasks.

A :class:`TaskPipeline` is a frozen, declarative ordering of
:class:`Task` instances. The Runner executes them in sequence; each
task's :attr:`Task.description` IS its user prompt, with no runtime
override. The pipeline aggregates per-task usage into a single
:class:`TaskPipelineResult.context` and supports conditional skip via
:attr:`Task.skip_if`.

Key design property: **the pipeline never transforms prompts at
runtime.** If task B needs task A's output, the developer wires that
explicitly: run task A via :meth:`Runner.arun_task`, then construct
task B with a description that embeds A's result, then run task B.
This matches CrewAI's `task.description = user prompt` model and
keeps the mental model unambiguous — the description you read in code
is exactly the prompt the agent sees.

The pipeline is the right abstraction when:

- You want N tasks to run in order with conditional :attr:`Task.skip_if`
  skips.
- You want cumulative LLM usage across the whole sequence in one
  :class:`TaskPipelineResult.context`.
- You want a single audit-channel object summarising N runs.
- The tasks do not need to forward intermediate outputs to each
  other's prompts (or you've already inlined those forwardings into
  each task's description at construction time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from troopai.adk.tasks.task_output import _output_preview

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tasks.task import Task
    from troopai.adk.tasks.task_output import TaskOutput


@dataclass(frozen=True, kw_only=True)
class TaskPipeline[TContext]:
    """Sequential composition of :class:`Task` instances.

    Each task's :attr:`Task.description` is its user prompt verbatim;
    the pipeline does NOT transform prompts at runtime. To forward an
    upstream output into a downstream task's prompt, call
    :meth:`Runner.arun_task` for the upstream task first, then
    construct the downstream :class:`Task` with a description that
    includes the prior output, then call :meth:`Runner.arun_task`
    again. The pipeline abstraction is reserved for sequential runs
    with conditional skip + usage aggregation; explicit data-flow
    chaining is the developer's responsibility.

    Attributes:
        tasks: Ordered tuple of tasks to execute. All tasks MUST share
            the same ``TContext`` — invariant generics enforce this at
            the type level. Empty tuples are rejected.

    Raises:
        ValueError: When :attr:`tasks` is empty.

    Example:
        ::

            from troopai.adk import Agent, Task, TaskPipeline, Runner

            classify = Task(description="Detect language.", agent=classifier)
            translate = Task(
                description="Translate to English.",
                agent=translator,
                skip_if=lambda prior: str(prior[-1].final_output).lower().startswith("english"),
            )
            review = Task(description="Comment in one sentence.", agent=reviewer)

            pipeline = TaskPipeline(tasks=(classify, translate, review))
            result = await Runner.arun_task_pipeline(pipeline)
    """

    tasks: tuple[Task[TContext], ...]
    """Ordered tuple of tasks to execute."""

    @override
    def __repr__(self) -> str:
        """One-line pipeline summary: task count + DAG flag.

        ``dag=True`` appears when any task declares
        :attr:`Task.depends_on` — the signal that execution runs in
        topological order instead of declaration order.
        """
        parts: list[str] = [f"tasks={len(self.tasks)}"]
        if any(t.depends_on for t in self.tasks):
            parts.append("dag=True")
        return f"TaskPipeline({', '.join(parts)})"

    def __post_init__(self) -> None:
        """Validate :class:`TaskPipeline` construction.

        Triggers full DAG validation (duplicate / unknown / missing
        IDs, cycles) when any task declares :attr:`Task.depends_on`.
        Pipelines without ``depends_on`` skip the DAG check and keep
        the existing declaration-order semantics — cost-conservative.

        Raises:
            ValueError: When :attr:`tasks` is empty.
            TaskPipelineDefinitionError: When the pipeline declares an
                invalid DAG. Subclass of
                :class:`troopai.adk.exceptions.exceptions.UserError`.
        """
        if len(self.tasks) == 0:
            raise ValueError("TaskPipeline.tasks must contain at least one Task")
        # Trigger topological validation eagerly so misconfigured
        # pipelines fail at construction (where the developer can
        # see the traceback), not deep in a run.
        if any(t.depends_on is not None and len(t.depends_on) > 0 for t in self.tasks):
            from troopai.adk.tasks.topology import topological_levels

            _ = topological_levels(self.tasks)

    def topological_levels(self) -> tuple[tuple[Task[TContext], ...], ...]:
        """Return the pipeline's tasks grouped by topological depth.

        Tasks at level 0 have no dependencies; tasks at level N depend
        only on tasks at levels 0..N-1. Within a level, tasks are
        deterministically sorted by ``task_id`` for stable output.
        Pipelines with no ``depends_on`` return a single level
        containing the tasks in declaration order.

        Returns:
            Tuple of levels; each level is a tuple of tasks safe to
            run concurrently.
        """
        from troopai.adk.tasks.topology import topological_levels

        return topological_levels(self.tasks)


@dataclass(frozen=True, kw_only=True)
class TaskPipelineResult[TContext]:
    """Result of a :class:`TaskPipeline` execution.

    Attributes:
        task_outputs: One :class:`TaskOutput` per task in
            :attr:`TaskPipeline.tasks`, in the same order. Skipped
            tasks appear with ``skipped=True`` — slots are NEVER
            silently dropped, so positional indexing matches the
            input pipeline.
        final_output: The :attr:`TaskOutput.final_output` of the LAST
            non-skipped task, or ``None`` when every task was skipped
            or the pipeline halted on error before producing output.
        context: The :class:`RunContext` shared across the whole
            pipeline. Its ``usage`` reflects cumulative LLM usage
            across every task that actually ran — the pipeline harness
            sums each completed task's :attr:`TaskOutput.usage` into this
            instance after the task returns.
    """

    task_outputs: tuple[TaskOutput, ...]
    """One TaskOutput per task in TaskPipeline.tasks, in order."""

    final_output: Any = None
    """final_output of the last non-skipped task; ``None`` otherwise."""

    context: RunContext[TContext] | None = None
    """Shared RunContext — ``.usage`` is cumulative across all tasks."""

    @override
    def __repr__(self) -> str:
        """One-line pipeline-result summary: slot counts + output preview.

        ``skipped`` / ``errors`` are derived from
        :attr:`task_outputs` — the same slots a human would scan to
        gauge pipeline health.
        """
        parts: list[str] = [f"tasks={len(self.task_outputs)}"]
        parts.append(f"skipped={sum(1 for o in self.task_outputs if o.skipped)}")
        parts.append(f"errors={sum(1 for o in self.task_outputs if o.error is not None)}")
        if self.final_output is not None:
            parts.append(f"final_output={_output_preview(self.final_output)}")
        return f"TaskPipelineResult({', '.join(parts)})"
