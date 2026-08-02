"""Topological-level resolver for :class:`TaskPipeline` DAG ordering.

Kahn's algorithm grouped by depth: every task at level 0 has zero
declared dependencies; every task at level N depends only on tasks
present at levels 0..N-1. :meth:`Runner.arun_task_pipeline` consumes
the result and runs each level's tasks concurrently via
``asyncio.gather``.

Module-private by omission from :mod:`troopai.adk.tasks`'s ``__all__``
— consumers go through :meth:`TaskPipeline.topological_levels`. The
file itself is not underscore-prefixed (module-private status is
communicated by omission from the package's ``__all__``, not by file
naming).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from troopai.adk.exceptions.exceptions import UserError

if TYPE_CHECKING:
    from troopai.adk.tasks.task import Task


class TaskPipelineDefinitionError(UserError):
    """Raised when a :class:`TaskPipeline` declares an invalid DAG.

    Subclass of :class:`UserError` — these are developer-side bugs in
    pipeline definition (unknown dependency ids, duplicate ids,
    cycles), not runtime failures.
    """


def normalised_depends_on(task: Task) -> tuple[str, ...]:
    """Resolve ``task.depends_on`` entries to task_id strings.

    ``depends_on`` is ``None`` (no deps) or a :class:`Sequence` whose
    entries are :class:`Task` instances, ``task_id`` strings, or
    :class:`TaskDependency` wrappers. Task instances (whether
    standalone or wrapped) MUST carry an explicit ``task_id``;
    otherwise we raise :class:`TaskPipelineDefinitionError` since we
    can't name them stably for the resolver.

    Args:
        task: The task whose ``depends_on`` sequence is to be
            normalised.

    Returns:
        A tuple of ``task_id`` strings extracted from the
        ``depends_on`` sequence, in declaration order. Empty tuple
        when ``depends_on`` is ``None``.

    Raises:
        TaskPipelineDefinitionError: When a :class:`Task` entry (bare
            or wrapped in :class:`TaskDependency`) has no ``task_id``
            set.
    """
    from troopai.adk.tasks.task import Task, TaskDependency

    if task.depends_on is None:
        return ()
    ids: list[str] = []
    for entry in task.depends_on:
        if isinstance(entry, TaskDependency):
            inner = entry.task
            if isinstance(inner, Task):
                if inner.task_id is None:
                    raise TaskPipelineDefinitionError(
                        f"Task {task.task_id!r} depends_on a TaskDependency wrapping a Task with no task_id. "
                        f"Set task_id on the referenced task.",
                    )
                ids.append(inner.task_id)
            else:
                ids.append(inner)
        elif isinstance(entry, Task):
            if entry.task_id is None:
                raise TaskPipelineDefinitionError(
                    f"Task {task.task_id!r} depends_on a Task instance with no task_id. "
                    f"Set task_id on the referenced task.",
                )
            ids.append(entry.task_id)
        else:
            ids.append(entry)
    return tuple(ids)


def topological_levels(tasks: Sequence[Task]) -> tuple[tuple[Task, ...], ...]:
    """Group ``tasks`` by topological depth using Kahn's algorithm.

    The result is a tuple of levels; each level is a tuple of tasks
    safe to run concurrently. Level 0 holds every task with zero
    declared dependencies, level N holds every task whose declared
    dependencies are all present in levels 0..N-1. Within each level,
    tasks are sorted by ``task_id`` for deterministic ordering — the
    same input pipeline always produces the same topology, which makes
    diffs (and the rendered output) stable across runs.

    Args:
        tasks: The pipeline's tasks in declaration order.

    Returns:
        A tuple of levels. Each level is a tuple of tasks whose
        dependencies are fully satisfied by previous levels.

    Raises:
        TaskPipelineDefinitionError: When the pipeline declares
            duplicate task IDs, references unknown IDs, leaves a
            dependency unsatisfied because a referenced task is
            missing a stable ``task_id``, or contains a cycle.
    """
    _validate_ids(tasks)
    by_id = {t.task_id: t for t in tasks if t.task_id is not None}
    deps_per_task = {tid: normalised_depends_on(t) for tid, t in by_id.items()}
    _validate_references(by_id, deps_per_task)
    indegree = {tid: len(deps_per_task[tid]) for tid in by_id}

    if not any(len(deps_per_task[tid]) > 0 for tid in by_id):
        return (tuple(tasks),)

    levels: list[tuple[Task, ...]] = []
    remaining = dict(indegree)

    while len(remaining) > 0:
        current_level_ids = sorted([tid for tid, deg in remaining.items() if deg == 0])
        if len(current_level_ids) == 0:
            raise TaskPipelineDefinitionError(
                f"TaskPipeline has a cycle involving task IDs: {sorted(remaining)}",
            )
        current_level = tuple(by_id[tid] for tid in current_level_ids)
        levels.append(current_level)
        for tid in current_level_ids:
            del remaining[tid]
        for tid in list(remaining.keys()):
            remaining[tid] = sum(1 for d in deps_per_task[tid] if d in remaining)

    return tuple(levels)


def _validate_ids(tasks: Sequence[Task]) -> None:
    """Reject duplicate task IDs and missing IDs in DAG pipelines.

    When ANY task in the pipeline declares ``depends_on``, ALL tasks must
    carry an explicit ``task_id``. A no-id task in a DAG pipeline cannot
    be addressed by ``completed_task_ids`` on resume, so it is rejected
    at construction instead of being placed silently.
    """
    is_dag = any(t.depends_on is not None and len(t.depends_on) > 0 for t in tasks)
    seen: dict[str, int] = {}
    for idx, task in enumerate(tasks):
        if task.task_id is not None:
            existing = seen.get(task.task_id)
            if existing is not None:
                raise TaskPipelineDefinitionError(
                    f"TaskPipeline has duplicate task_id={task.task_id!r} at indexes {existing} and {idx}.",
                )
            seen[task.task_id] = idx
        elif task.depends_on is not None and len(task.depends_on) > 0:
            raise TaskPipelineDefinitionError(
                f"Task at index {idx} declares depends_on but has no task_id. "
                f"Tasks that participate in a depends_on DAG must set a stable task_id.",
            )
        elif is_dag:
            raise TaskPipelineDefinitionError(
                f"Task at index {idx} has no task_id but this pipeline uses depends_on. "
                f"When any task declares depends_on, ALL tasks must carry an explicit task_id "
                f"so they can be addressed on resume and referenced in completed_task_ids.",
            )


def _validate_references(
    by_id: dict[str, Task],
    deps_per_task: dict[str, tuple[str, ...]],
) -> None:
    """Reject any depends_on entry that isn't a known task_id."""
    for tid, deps in deps_per_task.items():
        unknown = [d for d in deps if d not in by_id]
        if len(unknown) > 0:
            raise TaskPipelineDefinitionError(
                f"Task task_id={tid!r} declares depends_on={deps!r} "
                f"but {unknown!r} is not a task_id present in this pipeline.",
            )
