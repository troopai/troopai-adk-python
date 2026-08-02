"""Task — declarative unit of work for an :class:`Agent`.

A :class:`Task` pairs an agent with a description and optional per-call
overrides (output schema, guardrails, budgets, conditional skip).
Tasks are configuration objects; execution lives on
:meth:`Runner.arun_task` / :meth:`Runner.arun_task_pipeline` — agents are
configuration, runners execute.

Deliberately rejected CrewAI patterns:

- No ``expected_output`` field — developers put output expectations
  inside ``description`` or via :attr:`output_schema`. The framework
  NEVER mutates the LLM prompt behind the developer's back.
- No string-guardrails — guardrails MUST be explicit
  :class:`AgentInputGuardrail` / :class:`AgentOutputGuardrail`
  instances. The framework NEVER silently spawns an LLM guardrail
  agent.
- :attr:`skip_if` is a single callable on Task; no separate
  ``ConditionalTask`` class, no hidden ``get_skipped_task_output()``.
- Pipelines NEVER transform prompts at runtime. ``Task.description``
  is fed verbatim as the user prompt; if a downstream task needs an
  upstream task's output in its prompt, the developer wires that by
  running the upstream task first and constructing the downstream
  task with a description that embeds the prior result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

from troopai.adk.agents.agent_guardrails import AgentGuardrails
from troopai.adk.tasks.task_input_data import TaskInputData
from troopai.adk.tasks.task_output import _output_preview

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.tasks.task_output import TaskOutput
    from troopai.adk.types.tokens.llm_usage import LLMUsageLimits

type TaskInputFilter = Callable[[TaskInputData], TaskInputData]
"""Callable shaping one upstream task's contribution to a downstream task.

Receives a :class:`TaskInputData` describing one upstream task's
completion (``task_id``, ``output``, ``items``), and returns a new
:class:`TaskInputData` (via :meth:`TaskInputData.clone`) with
``forwarded`` set to the subset of :class:`RunItem` instances that
should flow into the downstream task's input.

Wire shape: the runner concatenates ``forwarded`` items across all
upstreams (in ``depends_on`` declaration order), converts each item
to a Layer-1 ``LLMInputContentItem`` via :meth:`RunItem.to_param`,
and prepends the resulting message list BEFORE the message(s)
derived from ``Task.description``. The downstream agent's user
prompt becomes a single ``list[LLMInputContentItem]`` containing the
forwarded messages followed by the description messages.

The filter is attached per-upstream via :class:`TaskDependency`. A
bare ``Task`` or ``task_id`` string in ``Task.depends_on`` declares
pure ordering with no forwarding. Wrap an upstream in
:class:`TaskDependency` when you want a filter; mix bare and wrapped
entries freely in the same ``depends_on`` list.
"""


@dataclass(frozen=True, kw_only=True)
class TaskDependency:
    """One upstream task with optional per-edge forwarding policy.

    Used as an entry in :attr:`Task.depends_on`. A bare :class:`Task`
    instance (or ``task_id`` string) entry declares pure ordering —
    wait for the upstream to complete, do not read its output. A
    :class:`TaskDependency` entry declares ordering AND optional
    forwarding via :attr:`input_filter` — different upstreams of the
    same downstream task can carry different filters.

    Attributes:
        task: Upstream task reference — :class:`Task` instance or
            ``task_id`` string. Task instances must have an explicit
            ``task_id`` set so the resolver can name them stably.
        input_filter: Optional :data:`TaskInputFilter`. When set, the
            runner builds a :class:`TaskInputData` after the upstream
            completes, calls this filter, and concatenates the
            resulting ``forwarded`` items into the downstream task's
            user prompt. When ``None``, this dependency is pure
            ordering — the upstream's output is not forwarded.
    """

    task: Task[Any] | str
    """Upstream task reference."""

    input_filter: TaskInputFilter | None = None
    """Optional filter transforming upstream output into downstream input."""

    @override
    def __repr__(self) -> str:
        """One-line dependency summary: upstream identity + filter name.

        The upstream renders as its ``task_id`` when set, else its
        ``name``, else a capped description preview — never the full
        :class:`Task` repr, which would bury the edge in noise.
        """
        parts: list[str] = []
        task = self.task
        if isinstance(task, str):
            parts.append(f"task={task!r}")
        elif task.task_id is not None:
            parts.append(f"task={task.task_id!r}")
        elif task.name is not None:
            parts.append(f"task={task.name!r}")
        else:
            parts.append(f"task={_output_preview(task.description)}")
        if self.input_filter is not None:
            filter_name = getattr(self.input_filter, "__name__", type(self.input_filter).__name__)
            parts.append(f"input_filter={filter_name}")
        return f"TaskDependency({', '.join(parts)})"


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Task[TContext]:
    """Declarative unit of work for an :class:`Agent`.

    Attributes:
        description: Objective fed to the agent as the user prompt
            verbatim. The framework does NOT transform prompts at
            runtime — what you put here is exactly what the agent
            sees. In a :class:`TaskPipeline`, each task's description
            is its own prompt; the pipeline does not chain outputs
            into downstream prompts. To forward an upstream result,
            run the upstream task first and construct the downstream
            task with a description that embeds the prior output.
        agent: The execution target. May be an :class:`Agent`, a
            :class:`Swarm` (iterative multi-agent collaboration), or
            a :class:`Graph` (state-machine orchestration). The
            Runner dispatches to the matching ``arun_*`` entry point
            via an :func:`isinstance` chain.

            The field is called ``agent`` because Agents are the
            canonical execution target; Swarm and Graph are unioned
            in for symmetric dispatch. Read it as "the thing that
            executes this task" — the type annotation makes the
            union explicit.

            Hooks under non-Agent targets:

            * :class:`Swarm` — ``hooks`` propagate normally through
              :meth:`Runner.arun_swarm`.
            * :class:`Graph` — user-supplied ``RunHooks`` are NOT
              propagated into the graph (the graph layer uses
              :class:`GraphHooks` instead). ``on_task_start`` /
              ``on_task_end`` still fire from :meth:`Runner.arun_task`,
              but per-node hooks must be attached to the
              :class:`Graph` directly.
        name: Optional human-readable display name surfaced in the
            verbose Task panel, ``on_task_start`` / ``on_task_end``
            hooks, and tracing metadata. Defaults to a truncated form
            of :attr:`description` (≤ 80 chars).
        task_id: Optional stable identity. When ``None``, the Runner
            generates a full ``str(uuid.uuid4())`` (36-char canonical
            UUID with hyphens) per invocation. Set explicitly to
            correlate spans across retries or replays. The verbose
            Task panel truncates to the first 8 characters for
            display, but the full UUID propagates through hooks,
            tracing, and :class:`TaskOutput.task_id`. When any task
            in a :class:`TaskPipeline` declares :attr:`depends_on`,
            every referenced task MUST have an explicit ``task_id``
            so the dependency resolver can name them — the validator
            raises a clear error pointing to any task that's missing
            one.
        depends_on: Optional tuple of upstream task references that
            MUST complete before this task fires. Each entry is a
            :class:`Task` instance (the framework reads its ``task_id``),
            a ``task_id`` string, or a :class:`TaskDependency` wrapper
            (ordering plus an optional per-upstream ``input_filter``).
            When non-empty, the
            owning :class:`TaskPipeline` switches from
            sequential-by-declaration order to topological DAG
            execution: tasks at the same depth run concurrently via
            ``asyncio.gather`` and downstream tasks wait until all
            their upstream dependencies finish. ``None`` (the default)
            keeps the existing declaration-order semantics —
            cost-conservative, no behaviour change vs. pre-DAG runs.
            Validation in :meth:`TaskPipeline.__post_init__` rejects
            unknown IDs, duplicate IDs, and cycles.
        output_schema: Optional per-task override for the agent's
            structured-output schema. The Runner constructs a
            transient ``dataclasses.replace(agent, output_schema=…)``
            for this call; the original agent definition is untouched.
            **Only supported when :attr:`agent` is an :class:`Agent`.**
            Setting ``output_schema`` together with a :class:`Swarm`
            or :class:`Graph` target raises :class:`ValueError` at
            Task construction — the override has no meaningful
            target for those types.
        guardrails: Per-task :class:`AgentGuardrails` config holding
            ``input`` and ``output`` guardrail lists. Mirrors
            :attr:`Agent.guardrails` exactly so the same authoring
            patterns apply at the task scope. The Runner appends task
            guardrails AFTER any ``RunConfig`` guardrails when building
            the transient run config — run-scope guardrails run first,
            task-scope guardrails second. Duplicates are NOT
            de-duplicated; the same callable appearing twice runs
            twice.
        max_turns: Optional per-task ceiling on the agent loop. When
            ``None``, the Runner falls back to its ``DEFAULT_MAX_TURNS``.
            Cost-conservative: explicit opt-in, never silently widened.
        usage_limits: Optional per-task LLM usage budget. When ``None``,
            the caller's ``RunConfig.usage_limits`` flows through
            unchanged.
        skip_if: Optional predicate for pipeline-conditional execution.
            Called by :meth:`Runner.arun_task_pipeline` with the immutable
            tuple of prior :class:`TaskOutput` results (in order;
            skipped tasks remain in the tuple with ``skipped=True``).
            Returning ``True`` causes the Runner to insert a
            ``TaskOutput(skipped=True, …)`` slot and skip the agent
            call. The predicate MUST be a pure function of its inputs
            — mutable closure state is the caller's responsibility and
            is NOT validated by the framework. Ignored by
            :meth:`Runner.arun_task` (single-task path).
        metadata: Open-ended developer metadata. Surfaced verbatim on
            :attr:`TaskOutput.metadata`. Use for request-correlation
            IDs, tags, custom trace attributes.

    Raises:
        ValueError: When :attr:`description` is empty or whitespace,
            or when :attr:`max_turns` is non-positive.

    Example:
        ::

            from troopai.adk import Agent, Task, Runner

            summariser = Agent(name="Summariser", system_prompt="…")
            task = Task(
                description="Summarise the notes below.",
                agent=summariser,
            )
            output = await Runner.arun_task(task)
    """

    description: UserPrompt
    """Objective fed to the agent as the user prompt.

    Accepts either a plain ``str`` (the common case) or a
    ``list[LLMInputContentItem]`` for full control over the message
    structure. The ``str`` form is presented to the LLM as a single
    user message. The list form is passed through to
    :meth:`Runner.arun` verbatim — useful when a :class:`TaskDependency`
    ``input_filter`` prepends upstream messages, or when the developer
    wants to inject system / assistant priming alongside the user turn.
    """

    agent: Agent[TContext] | Swarm[TContext] | Graph[TContext]
    """Execution target — Agent, Swarm, or Graph."""

    name: str | None = None
    """Optional display name (defaults to truncated description)."""

    task_id: str | None = None
    """Optional stable identity (defaults to fresh ``str(uuid.uuid4())``)."""

    depends_on: Sequence[Task[Any] | str | TaskDependency] | None = None
    """Upstream task references — bare or wrapped per-edge.

    Each entry is one of:

    - A :class:`Task` instance — pure ordering, no forwarding.
    - A ``task_id`` string — pure ordering, no forwarding.
    - A :class:`TaskDependency` wrapper — ordering plus an optional
      :attr:`TaskDependency.input_filter` shaping that upstream's
      contribution to this task's input.

    ``None`` (the default) ⇒ pipeline runs in declaration order. Non-
    empty ⇒ owning :class:`TaskPipeline` runs in topological order
    with same-depth tasks concurrent. Any :class:`Sequence` is
    accepted (list or tuple); mix bare and wrapped entries freely.
    """

    output_schema: type | AgentOutputSchemaBase | None = None
    """Optional per-call output schema override."""

    guardrails: AgentGuardrails = field(default_factory=AgentGuardrails)
    """Per-task input/output guardrails appended after RunConfig guardrails.

    A single :class:`AgentGuardrails` config holding ``input`` and
    ``output`` lists — mirrors :attr:`Agent.guardrails` exactly.
    """

    max_turns: int | None = None
    """Optional per-task ceiling on the agent loop."""

    usage_limits: LLMUsageLimits | None = None
    """Optional per-task LLM usage budget."""

    skip_if: Callable[[Sequence[TaskOutput]], bool] | None = None
    """Optional pipeline-skip predicate. Pure function of prior outputs."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Open-ended developer metadata surfaced on TaskOutput."""

    def __post_init__(self) -> None:
        """Validate :class:`Task` construction.

        Raises:
            ValueError: When ``description`` is empty/whitespace, when
                ``max_turns`` is set but non-positive, or when
                ``output_schema`` is set together with a non-Agent
                target.
        """
        from troopai.adk.agents.agent import Agent

        if isinstance(self.description, str):
            if len(self.description.strip()) == 0:
                raise ValueError("Task.description must be a non-empty string")
        elif len(self.description) == 0:
            raise ValueError(
                "Task.description (list form) must contain at least one LLMInputContentItem",
            )
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError(f"Task.max_turns must be positive when set, got {self.max_turns}")
        if self.output_schema is not None and not isinstance(self.agent, Agent):
            raise ValueError(
                "Task.output_schema is only supported when Task.agent is an Agent. "
                "Swarm and Graph targets manage their own output shape.",
            )

    @override
    def __repr__(self) -> str:
        """One-line task summary for humans.

        The full dataclass repr dumps the agent's system prompt, tools,
        and guardrails — unreadable in a REPL or log line. This shows
        what a human checks first: the task ``name`` (or a capped
        description preview when unnamed), the execution target's
        ``name`` (falling back to its class name when the target has
        none, e.g. an unnamed :class:`Swarm` or a :class:`Graph`), and
        the dependency count. ``skip_if`` surfaces only when set.
        """
        parts: list[str] = []
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        else:
            parts.append(f"description={_output_preview(self.description)}")
        target_name = getattr(self.agent, "name", None)
        if target_name is not None:
            parts.append(f"agent={target_name!r}")
        else:
            parts.append(f"agent={type(self.agent).__name__}")
        if self.depends_on:
            parts.append(f"depends_on={len(self.depends_on)}")
        if self.skip_if is not None:
            parts.append("skip_if=True")
        return f"Task({', '.join(parts)})"
