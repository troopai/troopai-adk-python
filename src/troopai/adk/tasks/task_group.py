"""TaskGroup + TaskGroupResult — parallel composition of Tasks.

A :class:`TaskGroup` is the parallel counterpart to :class:`TaskPipeline`.
It bundles N tasks that run **concurrently** under
:func:`asyncio.gather` semantics, optionally bounded by a semaphore,
and aggregates per-task usage into a single
:class:`TaskGroupResult.context`.

Use a :class:`TaskGroup` when:

- You want N independent tasks to run in parallel (different prompts,
  different agents, or the same agent against different inputs).
- You want cumulative LLM usage across the whole fan-out in one
  :class:`TaskGroupResult.context`.
- You want a single audit-channel object summarising N concurrent runs
  with input-order-preserved result indexing.

A :class:`TaskGroup` is the wrong abstraction when tasks depend on each
other's outputs — use :class:`TaskPipeline` for sequential chaining or
:class:`Graph` for arbitrary dependency DAGs.

Concurrency contract (developer-visible):

- ``on_task_start`` / ``on_task_end`` hooks fire **concurrently** when
  invoked from :meth:`Runner.arun_task_group`. Hooks holding shared
  mutable state MUST lock — the framework does not synchronise.
- **Shared :class:`Session` is not recommended.** When N parallel
  tasks share a single Session, writes race on the SQLite connection.
  ``aiosqlite`` queues at the connection level, but per-task
  multi-statement sequences (INSERT events + UPDATE timestamp + COMMIT)
  can interleave with other tasks' writes, producing a non-coherent
  audit log. For multi-task groups give each task its own
  :class:`Session` (or write to a downstream aggregator after each
  task completes).
- A cancelled task's in-flight provider HTTP call may complete after
  the cancel is observed locally — meaning a Session write can land
  AFTER the result tuple reports the slot as cancelled. Treat the
  cancel as a control-plane signal, not a data-plane delete.
- ``error_policy="halt_on_first"`` cancels still-running siblings via
  ``asyncio.Task.cancel()``. The cancel is best-effort: in-flight
  provider HTTP calls finish unless the LLM client itself honours
  cancellation. Cancelled tasks appear in :attr:`TaskGroupResult.task_outputs`
  with an explanatory ``error`` field.

Cost-conservative defaults (every field opts the developer into a
specific cost; nothing is implicit):

- :attr:`TaskGroup.error_policy` defaults to ``"collect_all"`` —
  every task runs to completion regardless of sibling failures.
- :attr:`TaskGroup.max_concurrent` defaults to ``None`` — concurrency
  matches the task count. Set to a small integer to throttle against
  provider rate limits. Cost trade-off: ``None`` means N parallel
  provider calls (faster but more bursty); a small value spreads
  calls over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tasks.task import Task
    from troopai.adk.tasks.task_output import TaskOutput


ErrorPolicy = Literal["halt_on_first", "collect_all"]
"""Behaviour when one task in the group fails.

``"collect_all"`` (default) — every task runs to completion; failures
surface as :class:`TaskOutput` slots with the ``error`` field set.

``"halt_on_first"`` — on the first task failure, the Runner cancels
all still-running siblings via :meth:`asyncio.Task.cancel`. The
cancelled tasks appear in the result tuple with
``error="cancelled by halt_on_first policy"`` so positional indexing
remains stable.
"""


@dataclass(frozen=True, kw_only=True)
class TaskGroup[TContext]:
    """Parallel composition of :class:`Task` instances.

    All tasks in the group MUST share the same ``TContext``
    parameter — invariant generics enforce this at the type level.
    The Runner executes them concurrently under
    :func:`asyncio.gather`, optionally bounded by an
    :class:`asyncio.Semaphore` when :attr:`max_concurrent` is set.

    Attributes:
        tasks: Ordered tuple of tasks to execute. Order determines
            positional indexing in
            :attr:`TaskGroupResult.task_outputs`; execution order is
            asyncio-scheduler-dependent and is NOT guaranteed.
            Empty tuples are rejected.
        error_policy: How to handle the first task failure. Default
            ``"collect_all"`` runs every task to completion;
            ``"halt_on_first"`` cancels still-running siblings on the
            first failure. See module docstring for the cancellation
            contract.
        max_concurrent: Maximum number of tasks running concurrently.
            ``None`` (the default) means no throttle —
            ``len(tasks)`` tasks run in parallel. Set to a positive
            integer to bound concurrency (useful against provider
            rate limits or memory pressure). MUST be positive when
            set.
        metadata: Open-ended developer metadata. Carried forward on
            ``TaskGroupResult.metadata``. Use for request-correlation
            IDs, tags, custom trace attributes.

    Raises:
        ValueError: When :attr:`tasks` is empty, or when
            :attr:`max_concurrent` is set but non-positive.

    Example:
        ::

            import logging

            from troopai.adk import Agent, Task, Runner
            from troopai.adk.tasks import TaskGroup

            logger = logging.getLogger(__name__)

            researcher = Agent(name="r", system_prompt="Research the topic.")
            group = TaskGroup(
                tasks=(
                    Task(description="Quantum computing.", agent=researcher),
                    Task(description="Machine learning.", agent=researcher),
                    Task(description="Cryptography.", agent=researcher),
                ),
                max_concurrent=2,
            )
            result = await Runner.arun_task_group(group)
            for output in result.task_outputs:
                logger.info("%s -> %s", output.task_name, output.final_output)
    """

    tasks: tuple[Task[TContext], ...]
    """Ordered tuple of tasks to execute concurrently."""

    error_policy: ErrorPolicy = "collect_all"
    """Behaviour on first task failure."""

    max_concurrent: int | None = None
    """Concurrency ceiling; ``None`` means no throttle."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Open-ended developer metadata."""

    @override
    def __repr__(self) -> str:
        """One-line group summary: task count, policy, throttle.

        ``max_concurrent`` appears only when set — ``None`` (no
        throttle) is the default and adds no information.
        """
        parts: list[str] = [f"tasks={len(self.tasks)}", f"error_policy={self.error_policy!r}"]
        if self.max_concurrent is not None:
            parts.append(f"max_concurrent={self.max_concurrent}")
        return f"TaskGroup({', '.join(parts)})"

    def __post_init__(self) -> None:
        """Validate :class:`TaskGroup` construction.

        Raises:
            ValueError: When :attr:`tasks` is empty, or when
                :attr:`max_concurrent` is set but non-positive.
        """
        if len(self.tasks) == 0:
            raise ValueError("TaskGroup.tasks must contain at least one Task")
        if self.max_concurrent is not None and self.max_concurrent <= 0:
            raise ValueError(
                f"TaskGroup.max_concurrent must be positive when set, got {self.max_concurrent}",
            )


@dataclass(frozen=True, kw_only=True)
class TaskGroupResult[TContext]:
    """Result of a :class:`TaskGroup` execution.

    Attributes:
        task_outputs: One :class:`TaskOutput` per task in
            :attr:`TaskGroup.tasks`, in the **same input order**.
            Execution order is asyncio-scheduler-dependent, but the
            result tuple preserves the position each task occupied in
            :attr:`TaskGroup.tasks`. Failed tasks appear with
            ``error`` set; cancelled (by ``halt_on_first``) tasks
            appear with an explanatory ``error`` field.
        context: The :class:`RunContext` for the whole group. Its
            ``usage`` field aggregates :class:`LLMUsage` across every
            task that produced one (skipped / cancelled-before-LLM
            tasks contribute zero usage). The aggregation order is
            non-deterministic, but
            :meth:`LLMUsage.__add__` is commutative so the resulting
            totals are stable.
        metadata: Verbatim copy of :attr:`TaskGroup.metadata`.
            Carried through for downstream correlation.
    """

    task_outputs: tuple[TaskOutput, ...]
    """One TaskOutput per task in TaskGroup.tasks, in input order."""

    context: RunContext[TContext] | None = None
    """RunContext for the group; ``.usage`` is cumulative across tasks."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Verbatim copy of TaskGroup.metadata."""

    @override
    def __repr__(self) -> str:
        """One-line group-result summary: slot count + error count."""
        parts: list[str] = [
            f"tasks={len(self.task_outputs)}",
            f"errors={sum(1 for o in self.task_outputs if o.error is not None)}",
        ]
        return f"TaskGroupResult({', '.join(parts)})"
