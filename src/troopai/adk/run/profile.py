"""Immutable runner profiles and target-specific runner handles.

``RunnerProfile`` captures reusable run defaults. Target runners bind those
defaults to one executable primitive and delegate to the existing
``Runner`` classmethods; they do not introduce a second execution path.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar, overload, override

from troopai.adk.agents.agent_guardrails import AgentGuardrails
from troopai.adk.run.config import DEFAULT_MAX_TURNS, RunConfig
from troopai.adk.types.tokens.llm_usage import LLMUsageLimits

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.agents.agent_guardrails import AgentInputGuardrail, AgentOutputGuardrail
    from troopai.adk.audit import AuditSink
    from troopai.adk.budgets import CostLedger, TenantBudget
    from troopai.adk.context.context_config import ContextManagementConfig
    from troopai.adk.flows.checkpoint import FlowCheckpoint
    from troopai.adk.flows.config import FlowConfig
    from troopai.adk.flows.flow import Flow
    from troopai.adk.flows.result import FlowRunResult, FlowRunResultStreaming
    from troopai.adk.flows.worker_backend import FlowWorkerBackend
    from troopai.adk.graphs.checkpointer import Checkpointer
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.hooks import GraphHooks, HookProvider
    from troopai.adk.graphs.interrupt import GraphResume
    from troopai.adk.graphs.result import GraphRunResult, GraphRunResultStreaming
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.llms.llm_usage import LLMUsageLimits as LLMUsageLimitsAlias
    from troopai.adk.llms.routing import LLMRouter
    from troopai.adk.memory.memory_config import MemoryConfig
    from troopai.adk.run.config import ErrorHandler, HistoryProcessor
    from troopai.adk.run.state import RunState
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.sandbox.config import SandboxRunConfig
    from troopai.adk.swarms.checkpointer import SwarmCheckpointer
    from troopai.adk.swarms.interrupt import SwarmResume
    from troopai.adk.swarms.result import SwarmRunResult, SwarmRunResultStreaming
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.swarms.termination import TerminationCondition
    from troopai.adk.tasks.task import Task
    from troopai.adk.tasks.task_group import TaskGroup, TaskGroupResult
    from troopai.adk.tasks.task_output import TaskOutput
    from troopai.adk.tasks.task_pipeline import TaskPipeline, TaskPipelineResult
    from troopai.adk.tasks.task_pipeline_state import TaskPipelineState
    from troopai.adk.types.run import RunResult
    from troopai.adk.types.session import SessionStore
    from troopai.adk.verbose.config import VerboseConfig

TContext = TypeVar("TContext")


def _copy_run_config(config: RunConfig) -> RunConfig:
    """Return a fresh ``RunConfig`` copy safe for profile reuse.

    ``RunConfig`` is intentionally mutable. Profiles and target runners are
    frozen, so every exposed config copy and every execution call receives a
    fresh config object. Mutable containers are copied at the top level; owned
    service objects such as ledgers and audit sinks are shared by reference.
    """
    return config.snapshot()


@dataclass(frozen=True, init=False, slots=True)
class RunnerProfile:
    """Reusable immutable defaults for ``Runner`` execution.

    A profile is target-agnostic: it stores a :class:`RunConfig` and optional
    user context. Bind a target via :meth:`agent`, :meth:`swarm`,
    :meth:`graph`, :meth:`task`, :meth:`pipeline`, :meth:`task_group`, or
    :meth:`flow` to get an executable runner.
    """

    _config: RunConfig = field(repr=False)
    _context: Any = field(default=None, repr=False)

    def __init__(
        self,
        run_config: RunConfig | None = None,
        *,
        context: Any = None,
    ) -> None:
        config = run_config if run_config is not None else RunConfig()
        object.__setattr__(self, "_config", _copy_run_config(config))
        object.__setattr__(self, "_context", context)

    @property
    def run_config(self) -> RunConfig:
        """A fresh copy of the profile's run configuration."""
        return _copy_run_config(self._config)

    @property
    def context_value(self) -> Any:
        """The user context value applied to target runners by default."""
        return self._context

    def with_config(self, config: RunConfig) -> RunnerProfile:
        """Replace this profile's base ``RunConfig``."""
        return RunnerProfile(config, context=self._context)

    def context(self, context: Any) -> RunnerProfile:
        """Return a profile with a different user context."""
        return RunnerProfile(self._config, context=context)

    def model(self, model: str) -> RunnerProfile:
        """Set the default model for targets that use ``RunConfig``."""
        config = self.run_config
        config.model = model
        return self.with_config(config)

    def verbose(
        self,
        config: VerboseConfig | None = None,
        *,
        enabled: bool = True,
    ) -> RunnerProfile:
        """Enable, configure, or disable verbose run output."""
        updated = self.run_config
        if not enabled:
            updated.verbose = None
            return self.with_config(updated)
        if config is not None:
            updated.verbose = config
            return self.with_config(updated)
        from troopai.adk.verbose.config import VerboseConfig

        updated.verbose = VerboseConfig()
        return self.with_config(updated)

    def limits(
        self,
        limits: LLMUsageLimits | None = None,
        *,
        requests: int | None = None,
        tool_calls: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tokens: int | None = None,
    ) -> RunnerProfile:
        """Set token/request limits for targets that use ``RunConfig``."""
        if limits is not None and any(
            value is not None for value in (requests, tool_calls, input_tokens, output_tokens, tokens)
        ):
            raise ValueError("Pass either limits= or individual limit fields, not both.")
        if limits is None:
            limits = LLMUsageLimits()
            if requests is not None:
                limits.request_limit = requests
            if tool_calls is not None:
                limits.tool_calls_limit = tool_calls
            if input_tokens is not None:
                limits.input_tokens_limit = input_tokens
            if output_tokens is not None:
                limits.output_tokens_limit = output_tokens
            if tokens is not None:
                limits.total_tokens_limit = tokens
        config = self.run_config
        config.usage_limits = limits
        return self.with_config(config)

    def context_management(self, config: ContextManagementConfig) -> RunnerProfile:
        """Set context management for targets that use ``RunConfig``."""
        updated = self.run_config
        updated.context_management = config
        return self.with_config(updated)

    def history_processors(self, processors: list[HistoryProcessor]) -> RunnerProfile:
        """Set Layer 3 history processors."""
        updated = self.run_config
        updated.history_processors = list(processors)
        return self.with_config(updated)

    def max_total_turns(self, turns: int | None) -> RunnerProfile:
        """Set the cross-agent cumulative turn limit."""
        updated = self.run_config
        updated.max_total_turns = turns
        return self.with_config(updated)

    def fail_on_tool_error(self, enabled: bool = True) -> RunnerProfile:
        """Set whether tool errors raise instead of returning to the LLM."""
        updated = self.run_config
        updated.fail_on_tool_error = enabled
        return self.with_config(updated)

    def max_tool_calls_per_turn(self, limit: int) -> RunnerProfile:
        """Set the per-turn tool-call ceiling."""
        updated = self.run_config
        updated.max_tool_calls_per_turn = limit
        return self.with_config(updated)

    def tracing(
        self,
        *,
        enabled: bool = True,
        metadata: Mapping[str, Any] | None = None,
        metrics: bool | None = None,
    ) -> RunnerProfile:
        """Configure tracing and optional metric emission."""
        updated = self.run_config
        updated.tracing_enabled = enabled
        if metadata is not None:
            updated.tracing_metadata = dict(metadata)
        if metrics is not None:
            updated.metrics_enabled = metrics
        return self.with_config(updated)

    def metrics(self, *, enabled: bool = True) -> RunnerProfile:
        """Enable or disable metric instruments."""
        updated = self.run_config
        updated.metrics_enabled = enabled
        return self.with_config(updated)

    def tenant(self, tenant_id: str | None) -> RunnerProfile:
        """Set the opaque tenant id threaded through execution."""
        updated = self.run_config
        updated.tenant_id = tenant_id
        return self.with_config(updated)

    def tenant_budget(
        self,
        budget: TenantBudget,
        *,
        ledger: CostLedger | None = None,
        fail_open: bool | None = None,
    ) -> RunnerProfile:
        """Set tenant budget controls."""
        updated = self.run_config
        updated.tenant_budget = budget
        if ledger is not None:
            updated.cost_ledger = ledger
        if fail_open is not None:
            updated.ledger_fail_open = fail_open
        return self.with_config(updated)

    def cost_ledger(self, ledger: CostLedger) -> RunnerProfile:
        """Set the cross-run cost ledger."""
        updated = self.run_config
        updated.cost_ledger = ledger
        return self.with_config(updated)

    def tenant_tool_allowlist(
        self,
        allowlist: Mapping[str, set[str]],
        *,
        default_deny: bool | None = None,
        soft_deny: bool | None = None,
    ) -> RunnerProfile:
        """Set per-tenant tool allowlists."""
        updated = self.run_config
        updated.tenant_tool_allowlist = {tenant: set(tools) for tenant, tools in allowlist.items()}
        if default_deny is not None:
            updated.tenant_allowlist_default_deny = default_deny
        if soft_deny is not None:
            updated.tenant_allowlist_soft_deny = soft_deny
        return self.with_config(updated)

    def audit(self, sink: AuditSink | None, *, strict: bool | None = None) -> RunnerProfile:
        """Set the audit sink and optional strict failure mode."""
        updated = self.run_config
        updated.audit_sink = sink
        if strict is not None:
            updated.audit_strict = strict
        return self.with_config(updated)

    def router(self, router: LLMRouter | None) -> RunnerProfile:
        """Set the optional model router."""
        updated = self.run_config
        updated.router = router
        return self.with_config(updated)

    def guardrails(
        self,
        *,
        input: list[AgentInputGuardrail] | None = None,
        output: list[AgentOutputGuardrail] | None = None,
    ) -> RunnerProfile:
        """Set run-scope input and/or output guardrails."""
        updated = self.run_config
        updated.guardrails = AgentGuardrails(
            input=list(input) if input is not None else list(updated.guardrails.input),
            output=list(output) if output is not None else list(updated.guardrails.output),
        )
        return self.with_config(updated)

    def sandbox(self, config: SandboxRunConfig | None) -> RunnerProfile:
        """Set per-run sandbox configuration."""
        updated = self.run_config
        updated.sandbox = config
        return self.with_config(updated)

    def max_parallel_tools(self, limit: int | None) -> RunnerProfile:
        """Set the per-turn parallel tool concurrency limit."""
        updated = self.run_config
        updated.max_parallel_tools = limit
        return self.with_config(updated)

    def error_handlers(self, handlers: dict[type[Exception], ErrorHandler] | None) -> RunnerProfile:
        """Set exception recovery handlers."""
        updated = self.run_config
        updated.error_handlers = dict(handlers) if handlers is not None else None
        return self.with_config(updated)

    def include_hook_events(self, enabled: bool = True) -> RunnerProfile:
        """Emit hook lifecycle events during streaming runs."""
        updated = self.run_config
        updated.include_hook_events = enabled
        return self.with_config(updated)

    def agent(self, agent: Agent[TContext]) -> AgentRunner[TContext]:
        """Bind this profile to an agent target."""
        return AgentRunner(agent=agent, profile=self)

    def swarm(self, swarm: Swarm[TContext]) -> SwarmRunner[TContext]:
        """Bind this profile to a swarm target."""
        return SwarmRunner(swarm=swarm, profile=self)

    def graph(self, graph: Graph[Any]) -> GraphRunner:
        """Bind this profile to a graph target."""
        return GraphRunner(graph=graph, profile=self)

    def task(self, task: Task[TContext]) -> TaskRunner[TContext]:
        """Bind this profile to a task target."""
        return TaskRunner(task=task, profile=self)

    def pipeline(self, pipeline: TaskPipeline[TContext]) -> TaskPipelineRunner[TContext]:
        """Bind this profile to a task pipeline target."""
        return TaskPipelineRunner(pipeline=pipeline, profile=self)

    def task_pipeline(self, pipeline: TaskPipeline[TContext]) -> TaskPipelineRunner[TContext]:
        """Alias for :meth:`pipeline`."""
        return self.pipeline(pipeline)

    def task_group(self, group: TaskGroup[TContext]) -> TaskGroupRunner[TContext]:
        """Bind this profile to a task group target."""
        return TaskGroupRunner(group=group, profile=self)

    def flow(self, flow: Flow[Any]) -> FlowRunner:
        """Bind this profile to a flow target."""
        return FlowRunner(flow=flow, profile=self)


class _ProfileContextConfigured:
    """Mixin for runners that inherit only profile context."""

    profile: RunnerProfile

    def _with_profile(self: Self, profile: RunnerProfile) -> Self:
        raise NotImplementedError

    @property
    def context_value(self) -> Any:
        """The context value inherited from the profile."""
        return self.profile.context_value

    def context(self: Self, context: Any) -> Self:
        """Return this runner with a different profile context."""
        return self._with_profile(self.profile.context(context))


class _ProfileConfigured(_ProfileContextConfigured):
    """Mixin for target runners backed by a full ``RunnerProfile``."""

    @property
    def run_config(self) -> RunConfig:
        """A fresh config copy for inspection or execution."""
        return self.profile.run_config

    def with_config(self: Self, config: RunConfig) -> Self:
        """Return this runner with a different profile config."""
        return self._with_profile(self.profile.with_config(config))

    def model(self: Self, model: str) -> Self:
        """Set the default model for this runner."""
        return self._with_profile(self.profile.model(model))

    def verbose(
        self: Self,
        config: VerboseConfig | None = None,
        *,
        enabled: bool = True,
    ) -> Self:
        """Enable, configure, or disable verbose run output."""
        return self._with_profile(self.profile.verbose(config, enabled=enabled))

    def limits(
        self: Self,
        limits: LLMUsageLimitsAlias | None = None,
        *,
        requests: int | None = None,
        tool_calls: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tokens: int | None = None,
    ) -> Self:
        """Set token/request limits for this runner."""
        return self._with_profile(
            self.profile.limits(
                limits,
                requests=requests,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tokens=tokens,
            ),
        )

    def max_total_turns(self: Self, turns: int | None) -> Self:
        """Set the cross-agent cumulative turn limit."""
        return self._with_profile(self.profile.max_total_turns(turns))


@dataclass(frozen=True, slots=True)
class AgentRunner(_ProfileConfigured, Generic[TContext]):
    """Executable handle for running one agent with a ``RunnerProfile``."""

    agent: Agent[TContext]
    """Agent configuration to execute."""

    profile: RunnerProfile
    """Reusable run defaults."""

    turns: int | None = None
    """Per-agent turn limit override."""

    run_hooks: RunHooks[TContext] | None = None
    """Run-level hooks for this agent runner."""

    session_store: SessionStore | None = None
    """Optional conversation session."""

    memory_config: MemoryConfig | None = None
    """Optional memory configuration."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> AgentRunner[TContext]:
        return replace(self, profile=profile)

    def max_turns(self, turns: int) -> AgentRunner[TContext]:
        """Set the per-agent turn limit."""
        return replace(self, turns=turns)

    def hooks(self, hooks: RunHooks[TContext]) -> AgentRunner[TContext]:
        """Set run-level hooks."""
        return replace(self, run_hooks=hooks)

    def session(self, session: SessionStore) -> AgentRunner[TContext]:
        """Set the conversation session."""
        return replace(self, session_store=session)

    def memory(self, memory: MemoryConfig) -> AgentRunner[TContext]:
        """Set memory configuration."""
        return replace(self, memory_config=memory)

    @overload
    def run(
        self,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[False] = False,
    ) -> RunResult: ...

    @overload
    def run(
        self,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[True],
    ) -> RunResultStreaming: ...

    def run(
        self,
        user_prompt: UserPrompt | RunState,
        *,
        stream: bool = False,
    ) -> RunResult | RunResultStreaming:
        """Execute this agent synchronously."""
        from troopai.adk.run.runner import Runner

        if stream:
            return Runner.run(
                self.agent,
                user_prompt,
                stream=True,
                context=self.context_value,
                hooks=self.run_hooks,
                max_turns=self.turns if self.turns is not None else DEFAULT_MAX_TURNS,
                run_config=self.run_config,
                session=self.session_store,
                memory=self.memory_config,
            )
        return Runner.run(
            self.agent,
            user_prompt,
            stream=False,
            context=self.context_value,
            hooks=self.run_hooks,
            max_turns=self.turns if self.turns is not None else DEFAULT_MAX_TURNS,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )

    @overload
    async def arun(
        self,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[False] = False,
    ) -> RunResult: ...

    @overload
    async def arun(
        self,
        user_prompt: UserPrompt | RunState,
        *,
        stream: Literal[True],
    ) -> RunResultStreaming: ...

    async def arun(
        self,
        user_prompt: UserPrompt | RunState,
        *,
        stream: bool = False,
    ) -> RunResult | RunResultStreaming:
        """Execute this agent asynchronously."""
        from troopai.adk.run.runner import Runner

        if stream:
            return await Runner.arun(
                self.agent,
                user_prompt,
                stream=True,
                context=self.context_value,
                hooks=self.run_hooks,
                max_turns=self.turns if self.turns is not None else DEFAULT_MAX_TURNS,
                run_config=self.run_config,
                session=self.session_store,
                memory=self.memory_config,
            )
        return await Runner.arun(
            self.agent,
            user_prompt,
            stream=False,
            context=self.context_value,
            hooks=self.run_hooks,
            max_turns=self.turns if self.turns is not None else DEFAULT_MAX_TURNS,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )


@dataclass(frozen=True, slots=True)
class SwarmRunner(_ProfileConfigured, Generic[TContext]):
    """Executable handle for running one swarm with a ``RunnerProfile``."""

    swarm: Swarm[TContext]
    """Swarm configuration to execute."""

    profile: RunnerProfile
    """Reusable run defaults."""

    run_hooks: RunHooks[TContext] | None = None
    """Run-level hooks for this swarm runner."""

    session_store: SessionStore | None = None
    """Optional conversation session."""

    checkpointer_value: SwarmCheckpointer | None = None
    """Optional swarm checkpointer."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> SwarmRunner[TContext]:
        return replace(self, profile=profile)

    def hooks(self, hooks: RunHooks[TContext]) -> SwarmRunner[TContext]:
        """Set run-level hooks."""
        return replace(self, run_hooks=hooks)

    def session(self, session: SessionStore) -> SwarmRunner[TContext]:
        """Set the conversation session."""
        return replace(self, session_store=session)

    def checkpointer(self, checkpointer: SwarmCheckpointer) -> SwarmRunner[TContext]:
        """Attach a swarm checkpointer."""
        return replace(self, checkpointer_value=checkpointer)

    def termination(self, termination: TerminationCondition) -> SwarmRunner[TContext]:
        """Override the swarm termination condition on a copy."""
        return replace(self, swarm=replace(self.swarm, termination=termination))

    def run(self, user_prompt: UserPrompt) -> SwarmRunResult:
        """Execute this swarm synchronously."""
        from troopai.adk.run.runner import Runner

        return Runner.run_swarm(
            self.swarm,
            user_prompt,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            checkpointer=self.checkpointer_value,
        )

    async def arun(self, user_prompt: UserPrompt) -> SwarmRunResult:
        """Execute this swarm asynchronously."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_swarm(
            self.swarm,
            user_prompt,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            checkpointer=self.checkpointer_value,
        )

    async def arun_streamed(
        self,
        user_prompt: UserPrompt,
        *,
        initial_state: SwarmState[TContext] | None = None,
        resume: SwarmResume | None = None,
    ) -> SwarmRunResultStreaming[TContext]:
        """Execute this swarm with real-time event streaming."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_swarm_streamed(
            self.swarm,
            user_prompt,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            initial_state=initial_state,
            resume=resume,
            checkpointer=self.checkpointer_value,
        )


@dataclass(frozen=True, slots=True)
class GraphRunner(_ProfileConfigured):
    """Executable handle for running one graph with a ``RunnerProfile``."""

    graph: Graph[Any]
    """Graph artifact to execute."""

    profile: RunnerProfile
    """Reusable run defaults."""

    graph_hooks: list[GraphHooks[Any] | HookProvider] | None = None
    """Graph hooks and hook providers."""

    thread_id: str | None = None
    """Optional graph checkpointer thread id."""

    resume_checkpointer: Checkpointer | None = None
    """Optional checkpointer used for graph resume."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> GraphRunner:
        return replace(self, profile=profile)

    def hooks(self, hooks: list[GraphHooks[Any] | HookProvider]) -> GraphRunner:
        """Set graph hooks and hook providers."""
        return replace(self, graph_hooks=list(hooks))

    def thread(self, thread_id: str) -> GraphRunner:
        """Set the graph thread id."""
        return replace(self, thread_id=thread_id)

    def resume_from(self, checkpointer: Checkpointer, thread_id: str) -> GraphRunner:
        """Resume this graph from a persisted checkpoint."""
        return replace(self, resume_checkpointer=checkpointer, thread_id=thread_id)

    def run(
        self,
        user_prompt: UserPrompt | None = None,
        *,
        resume: GraphResume | None = None,
    ) -> GraphRunResult[Any]:
        """Execute this graph synchronously."""
        from troopai.adk.run.runner import Runner

        if self.resume_checkpointer is not None:
            if self.thread_id is None:
                raise ValueError("resume_from requires a thread_id.")
            return Runner.run_graph_from_checkpoint(
                self.graph,
                checkpointer=self.resume_checkpointer,
                thread_id=self.thread_id,
                user_prompt=user_prompt,
                context=self.context_value,
                hooks=self.graph_hooks,
                run_config=self.run_config,
                resume=resume,
            )
        if user_prompt is None:
            raise ValueError("user_prompt is required for a fresh graph run.")
        return Runner.run_graph(
            self.graph,
            user_prompt,
            context=self.context_value,
            hooks=self.graph_hooks,
            run_config=self.run_config,
            thread_id=self.thread_id,
        )

    @overload
    async def arun(
        self,
        user_prompt: UserPrompt | None = None,
        *,
        stream: Literal[False] = False,
        resume: GraphResume | None = None,
    ) -> GraphRunResult[Any]: ...

    @overload
    async def arun(
        self,
        user_prompt: UserPrompt | None = None,
        *,
        stream: Literal[True],
        resume: GraphResume | None = None,
    ) -> GraphRunResultStreaming: ...

    async def arun(
        self,
        user_prompt: UserPrompt | None = None,
        *,
        stream: bool = False,
        resume: GraphResume | None = None,
    ) -> GraphRunResult[Any] | GraphRunResultStreaming:
        """Execute this graph asynchronously."""
        from troopai.adk.run.runner import Runner

        if self.resume_checkpointer is not None:
            if self.thread_id is None:
                raise ValueError("resume_from requires a thread_id.")
            if stream:
                initial_state = await self.resume_checkpointer.load(self.thread_id, self.graph)
                if initial_state is None:
                    raise ValueError(
                        f"resume_from: no checkpoint for thread_id={self.thread_id!r} "
                        f"on graph id={self.graph.id!r}. Nothing to resume."
                    )
                effective_hooks = list(self.graph_hooks) if self.graph_hooks is not None else []
                if self.resume_checkpointer not in effective_hooks:
                    effective_hooks.append(self.resume_checkpointer)
                prompt: UserPrompt = user_prompt if user_prompt is not None else ""
                return await Runner.arun_graph_streamed(
                    self.graph,
                    prompt,
                    context=self.context_value,
                    hooks=effective_hooks,
                    run_config=self.run_config,
                    thread_id=self.thread_id,
                    initial_state=initial_state,
                    resume=resume,
                )
            return await Runner.arun_graph_from_checkpoint(
                self.graph,
                checkpointer=self.resume_checkpointer,
                thread_id=self.thread_id,
                user_prompt=user_prompt,
                context=self.context_value,
                hooks=self.graph_hooks,
                run_config=self.run_config,
                resume=resume,
            )
        if user_prompt is None:
            raise ValueError("user_prompt is required for a fresh graph run.")
        if stream:
            return await Runner.arun_graph_streamed(
                self.graph,
                user_prompt,
                context=self.context_value,
                hooks=self.graph_hooks,
                run_config=self.run_config,
                thread_id=self.thread_id,
                resume=resume,
            )
        return await Runner.arun_graph(
            self.graph,
            user_prompt,
            context=self.context_value,
            hooks=self.graph_hooks,
            run_config=self.run_config,
            thread_id=self.thread_id,
        )


@dataclass(frozen=True, slots=True)
class TaskRunner(_ProfileConfigured, Generic[TContext]):
    """Executable handle for running one task with a ``RunnerProfile``."""

    task: Task[TContext]
    """Task configuration to execute."""

    profile: RunnerProfile
    """Reusable run defaults."""

    run_hooks: RunHooks[TContext] | None = None
    """Run-level hooks for this task runner."""

    session_store: SessionStore | None = None
    """Optional conversation session."""

    memory_config: MemoryConfig | None = None
    """Optional memory configuration."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> TaskRunner[TContext]:
        return replace(self, profile=profile)

    def hooks(self, hooks: RunHooks[TContext]) -> TaskRunner[TContext]:
        """Set run-level hooks."""
        return replace(self, run_hooks=hooks)

    def session(self, session: SessionStore) -> TaskRunner[TContext]:
        """Set the conversation session."""
        return replace(self, session_store=session)

    def memory(self, memory: MemoryConfig) -> TaskRunner[TContext]:
        """Set memory configuration."""
        return replace(self, memory_config=memory)

    def run(self) -> TaskOutput:
        """Execute this task synchronously."""
        from troopai.adk.run.runner import Runner

        return Runner.run_task(
            self.task,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )

    async def arun(self) -> TaskOutput:
        """Execute this task asynchronously."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_task(
            self.task,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )

    async def arun_streamed(self) -> RunResultStreaming:
        """Execute this task with real-time event streaming."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_task_streamed(
            self.task,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )


@dataclass(frozen=True, slots=True)
class TaskPipelineRunner(_ProfileConfigured, Generic[TContext]):
    """Executable handle for running one task pipeline with a profile."""

    pipeline: TaskPipeline[TContext]
    """Task pipeline configuration to execute."""

    profile: RunnerProfile
    """Reusable run defaults."""

    run_hooks: RunHooks[TContext] | None = None
    """Run-level hooks for this pipeline runner."""

    session_store: SessionStore | None = None
    """Optional conversation session."""

    memory_config: MemoryConfig | None = None
    """Optional memory configuration."""

    state: TaskPipelineState | None = None
    """Optional state used to resume the pipeline."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> TaskPipelineRunner[TContext]:
        return replace(self, profile=profile)

    def hooks(self, hooks: RunHooks[TContext]) -> TaskPipelineRunner[TContext]:
        """Set run-level hooks."""
        return replace(self, run_hooks=hooks)

    def session(self, session: SessionStore) -> TaskPipelineRunner[TContext]:
        """Set the conversation session."""
        return replace(self, session_store=session)

    def memory(self, memory: MemoryConfig) -> TaskPipelineRunner[TContext]:
        """Set memory configuration."""
        return replace(self, memory_config=memory)

    def resume_from(self, state: TaskPipelineState) -> TaskPipelineRunner[TContext]:
        """Resume this pipeline from persisted state on async execution."""
        return replace(self, state=state)

    def run(self) -> TaskPipelineResult[TContext]:
        """Execute this task pipeline synchronously."""
        if self.state is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.arun())
                    return future.result()
            return asyncio.run(self.arun())
        from troopai.adk.run.runner import Runner

        return Runner.run_task_pipeline(
            self.pipeline,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )

    async def arun(self) -> TaskPipelineResult[TContext]:
        """Execute this task pipeline asynchronously."""
        from troopai.adk.run.runner import Runner

        if self.state is not None:
            return await Runner.arun_task_pipeline_from_state(
                self.pipeline,
                self.state,
                context=self.context_value,
                hooks=self.run_hooks,
                run_config=self.run_config,
                session=self.session_store,
                memory=self.memory_config,
            )
        return await Runner.arun_task_pipeline(
            self.pipeline,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )

    def arun_streamed(self) -> AsyncIterator[tuple[int, RunResultStreaming | None]]:
        """Stream this pipeline task by task."""
        from troopai.adk.run.runner import Runner

        return Runner.arun_task_pipeline_streamed(
            self.pipeline,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )


@dataclass(frozen=True, slots=True)
class TaskGroupRunner(_ProfileConfigured, Generic[TContext]):
    """Executable handle for running one task group with a profile."""

    group: TaskGroup[TContext]
    """Task group configuration to execute."""

    profile: RunnerProfile
    """Reusable run defaults."""

    run_hooks: RunHooks[TContext] | None = None
    """Run-level hooks for this task group runner."""

    session_store: SessionStore | None = None
    """Optional conversation session."""

    memory_config: MemoryConfig | None = None
    """Optional memory configuration."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> TaskGroupRunner[TContext]:
        return replace(self, profile=profile)

    def hooks(self, hooks: RunHooks[TContext]) -> TaskGroupRunner[TContext]:
        """Set run-level hooks."""
        return replace(self, run_hooks=hooks)

    def session(self, session: SessionStore) -> TaskGroupRunner[TContext]:
        """Set the conversation session."""
        return replace(self, session_store=session)

    def memory(self, memory: MemoryConfig) -> TaskGroupRunner[TContext]:
        """Set memory configuration."""
        return replace(self, memory_config=memory)

    def run(self) -> TaskGroupResult[TContext]:
        """Execute this task group synchronously."""
        from troopai.adk.run.runner import Runner

        return Runner.run_task_group(
            self.group,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )

    async def arun(self) -> TaskGroupResult[TContext]:
        """Execute this task group asynchronously."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_task_group(
            self.group,
            context=self.context_value,
            hooks=self.run_hooks,
            run_config=self.run_config,
            session=self.session_store,
            memory=self.memory_config,
        )


@dataclass(frozen=True, slots=True)
class FlowRunner(_ProfileContextConfigured):
    """Executable handle for running one flow with a profile context."""

    flow: Flow[Any]
    """Flow instance to execute."""

    profile: RunnerProfile
    """Reusable profile; only its context applies directly to flow runs."""

    flow_config: FlowConfig | None = None
    """Optional flow execution configuration."""

    @override
    def _with_profile(self, profile: RunnerProfile) -> FlowRunner:
        return replace(self, profile=profile)

    def config(self, config: FlowConfig | None) -> FlowRunner:
        """Set flow execution configuration."""
        return replace(self, flow_config=config)

    def run(self) -> FlowRunResult[Any]:
        """Execute this flow synchronously."""
        from troopai.adk.run.runner import Runner

        return Runner.run_flow(self.flow, config=self.flow_config, context=self.context_value)

    async def arun(self) -> FlowRunResult[Any]:
        """Execute this flow asynchronously."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_flow(self.flow, config=self.flow_config, context=self.context_value)

    def arun_streamed(self) -> FlowRunResultStreaming[Any]:
        """Execute this flow with real-time event streaming."""
        from troopai.adk.run.runner import Runner

        return Runner.arun_flow_streamed(self.flow, config=self.flow_config, context=self.context_value)

    async def arun_from_checkpoint(
        self,
        checkpoint: FlowCheckpoint,
        *,
        agent_resolutions: Mapping[str, str] | None = None,
    ) -> FlowRunResult[Any]:
        """Resume this flow from a checkpoint."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_flow_from_checkpoint(
            self.flow,
            checkpoint,
            config=self.flow_config,
            context=self.context_value,
            agent_resolutions=agent_resolutions,
        )

    async def arun_from_id(
        self,
        checkpoint_id: str,
        backend: FlowWorkerBackend,
        *,
        agent_resolutions: Mapping[str, str] | None = None,
    ) -> FlowRunResult[Any]:
        """Resume this flow by loading a checkpoint id from a backend."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_flow_from_id(
            self.flow,
            checkpoint_id,
            backend,
            config=self.flow_config,
            context=self.context_value,
            agent_resolutions=agent_resolutions,
        )

    async def arun_distributed(
        self,
        backend: FlowWorkerBackend,
        *,
        worker_id: str | None = None,
        agent_resolutions: Mapping[str, str] | None = None,
    ) -> FlowRunResult[Any]:
        """Run one distributed flow batch through a backend."""
        from troopai.adk.run.runner import Runner

        return await Runner.arun_flow_distributed(
            self.flow,
            backend,
            worker_id=worker_id,
            config=self.flow_config,
            context=self.context_value,
            agent_resolutions=agent_resolutions,
        )
