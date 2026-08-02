from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any, cast

import pytest

from troopai.adk import RunnerProfile as PackageRunnerProfile
from troopai.adk.agents import Agent
from troopai.adk.agents.agent_guardrails import AgentGuardrails
from troopai.adk.audit import InMemoryAuditSink
from troopai.adk.budgets import TenantBudget
from troopai.adk.context.context_config import CompactionConfig, ContextManagementConfig
from troopai.adk.flows import Flow, FlowConfig, flow_start
from troopai.adk.graphs.graph import Graph
from troopai.adk.run import (
    AgentRunner,
    FlowRunner,
    GraphRunner,
    RunHooks,
    Runner,
    RunnerProfile,
    SwarmRunner,
    TaskGroupRunner,
    TaskPipelineRunner,
    TaskRunner,
)
from troopai.adk.run.config import DEFAULT_RUN_CONFIG, ErrorHandler, RunConfig
from troopai.adk.run.messages import RunMessages
from troopai.adk.run.state import RunState
from troopai.adk.sandbox.clients.docker.docker_client import DockerSandboxClientOptions
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.selector import SandboxCandidate
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tasks import Task, TaskGroup, TaskPipeline
from troopai.adk.types.tokens.llm_usage import LLMUsageLimits
from troopai.adk.verbose.config import EVENT_TOOL_START, EventStyle, VerboseConfig


def _agent() -> Agent[Any]:
    return Agent(name="A", system_prompt="test")


def _swarm() -> Swarm[Any]:
    member = Agent(name="a", system_prompt="test")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(1),
    )


def _noop() -> str:
    return "ok"


def _graph() -> Graph[Any]:
    return Graph.new("g").node("solo", _noop).entry("solo").terminal("solo").compile()


@dataclass
class _FlowState:
    done: bool = False


class _Flow(Flow[_FlowState]):
    @flow_start
    async def start(self) -> None:
        self.state.done = True


def _task() -> Task[Any]:
    return Task(description="do it", agent=_agent())


def test_runner_configure_returns_immutable_runner_profile() -> None:
    profile = Runner.configure().model("base-model")

    changed = profile.model("changed-model")
    leaked_config = profile.run_config
    leaked_config.model = "mutated-outside"

    assert isinstance(profile, RunnerProfile)
    assert PackageRunnerProfile is RunnerProfile
    assert profile is not changed
    assert profile.run_config.model == "base-model"
    assert changed.run_config.model == "changed-model"

    # CPython's frozen+slots+init=False dataclass raises TypeError (super()
    # identity in the generated __setattr__) for non-field attributes on
    # current interpreters; either exception proves the profile is immutable.
    with pytest.raises((FrozenInstanceError, TypeError)):
        profile.context_value = {"user": "alice"}  # type: ignore[misc]


def test_profile_fluent_helpers_set_run_config_without_mutating_base_config() -> None:
    base = RunConfig(model="base")

    profile = (
        Runner.configure(base)
        .model("profile-model")
        .limits(tokens=50_000, requests=7)
        .verbose(enabled=False)
        .context({"tenant": "acme"})
    )

    assert base.model == "base"
    assert profile.context_value == {"tenant": "acme"}
    assert profile.run_config.model == "profile-model"
    assert profile.run_config.verbose is None
    assert profile.run_config.usage_limits == LLMUsageLimits(total_tokens_limit=50_000, request_limit=7)


def test_profile_guardrails_do_not_mutate_shared_base_config() -> None:
    base = RunConfig(guardrails=AgentGuardrails(output=[]))

    first = Runner.configure(run_config=base).guardrails(input=[])
    second = Runner.configure(run_config=base).guardrails(output=[])

    assert len(base.guardrails.input) == 0
    assert len(base.guardrails.output) == 0
    assert first.run_config.guardrails is not base.guardrails
    assert second.run_config.guardrails is not base.guardrails


def test_profile_verbose_disable_clears_base_config() -> None:
    base = RunConfig(verbose=VerboseConfig())

    disabled = Runner.configure(run_config=base).verbose(enabled=False)
    untouched = Runner.configure(run_config=base)

    assert disabled.run_config.verbose is None
    assert untouched.run_config.verbose is not None


def test_profile_run_config_copies_nested_mutable_config_objects() -> None:
    base = RunConfig(
        usage_limits=LLMUsageLimits(total_tokens_limit=10),
        verbose=VerboseConfig(),
    )
    profile = Runner.configure(base)

    leaked = profile.run_config
    assert leaked.usage_limits is not None
    assert leaked.verbose is not None
    leaked.usage_limits.total_tokens_limit = 99
    leaked.verbose.styles[EVENT_TOOL_START] = EventStyle(prefix="mutated")

    fresh = profile.run_config
    assert fresh.usage_limits is not None
    assert fresh.verbose is not None
    assert fresh.usage_limits.total_tokens_limit == 10
    assert fresh.verbose.styles[EVENT_TOOL_START].prefix != "mutated"


def test_run_config_snapshot_copies_value_config_and_shares_service_handles() -> None:
    sink = InMemoryAuditSink()
    metadata = {"trace": {"request_id": "r1"}}
    messages = RunMessages(tool_rejected="no")

    def value_error_handler(exc: Exception) -> str:
        del exc
        return "fallback"

    error_handlers: dict[type[Exception], ErrorHandler] = {ValueError: value_error_handler}
    base = RunConfig(
        tracing_metadata=metadata,
        usage_limits=LLMUsageLimits(total_tokens_limit=10),
        verbose=VerboseConfig(),
        context_management=ContextManagementConfig(
            compaction=CompactionConfig(enabled=True, instructions="summarize"),
        ),
        tenant_budget=TenantBudget(dollars_per_run=1.0),
        audit_sink=sink,
        guardrails=AgentGuardrails(output=[]),
        tenant_tool_allowlist={"tenant": {"search"}},
        messages=messages,
        error_handlers=error_handlers,
    )

    snapshot = base.snapshot()
    assert snapshot.usage_limits is not None
    assert snapshot.verbose is not None
    assert snapshot.context_management is not None
    assert snapshot.tenant_tool_allowlist is not None
    assert snapshot.messages is not None
    assert snapshot.error_handlers is not None
    assert base.tenant_tool_allowlist is not None

    assert snapshot is not base
    assert snapshot.audit_sink is sink
    assert snapshot.tenant_budget is base.tenant_budget
    assert snapshot.tracing_metadata == base.tracing_metadata
    assert snapshot.tracing_metadata is not base.tracing_metadata
    assert snapshot.tracing_metadata["trace"] is not metadata["trace"]
    assert snapshot.usage_limits is not base.usage_limits
    assert snapshot.verbose is not base.verbose
    assert snapshot.context_management is not base.context_management
    assert snapshot.guardrails is not base.guardrails
    assert snapshot.tenant_tool_allowlist == base.tenant_tool_allowlist
    assert snapshot.tenant_tool_allowlist is not base.tenant_tool_allowlist
    assert snapshot.tenant_tool_allowlist["tenant"] is not base.tenant_tool_allowlist["tenant"]
    assert snapshot.messages is not messages
    assert snapshot.error_handlers is not base.error_handlers

    snapshot.tracing_metadata["trace"]["request_id"] = "mutated"
    snapshot.usage_limits.total_tokens_limit = 99
    snapshot.verbose.styles[EVENT_TOOL_START] = EventStyle(prefix="mutated")
    snapshot.context_management.compaction.instructions = "changed"
    snapshot.guardrails.input.append(lambda _ctx, _agent, _input: None)  # type: ignore[arg-type]
    snapshot.tenant_tool_allowlist["tenant"].add("write")
    snapshot.messages.tool_rejected = "changed"

    def runtime_handler(exc: Exception) -> str:
        del exc
        return "runtime"

    snapshot.error_handlers[RuntimeError] = runtime_handler

    assert base.tracing_metadata["trace"]["request_id"] == "r1"
    assert base.usage_limits is not None
    assert base.usage_limits.total_tokens_limit == 10
    assert base.verbose is not None
    assert base.verbose.styles[EVENT_TOOL_START].prefix != "mutated"
    assert base.context_management is not None
    assert base.context_management.compaction.instructions == "summarize"
    assert base.guardrails.input == []
    assert base.tenant_tool_allowlist == {"tenant": {"search"}}
    assert base.messages is not None
    assert base.messages.tool_rejected == "no"
    assert base.error_handlers is not None
    assert RuntimeError not in base.error_handlers


def test_run_config_snapshot_copies_sandbox_value_options_and_shares_handles() -> None:
    client = object()
    selector = object()
    candidate_client = object()
    options = DockerSandboxClientOptions(
        image="python:3.12-slim",
        environment={"API_KEY": "base"},
    )
    candidate_options = DockerSandboxClientOptions(
        image="python:3.13-slim",
        environment={"TOKEN": "candidate"},
    )
    candidate = SandboxCandidate(client=candidate_client, options=candidate_options)  # type: ignore[arg-type]
    base = RunConfig(
        sandbox=SandboxRunConfig(
            client=client,
            options=options,
            selector=selector,  # type: ignore[arg-type]
            candidates=[candidate],
        )
    )

    snapshot = base.snapshot()
    assert snapshot.sandbox is not None
    assert snapshot.sandbox.client is client
    assert snapshot.sandbox.selector is selector
    assert snapshot.sandbox.options is not options
    assert snapshot.sandbox.candidates is not None
    assert snapshot.sandbox.candidates[0] is not candidate
    assert snapshot.sandbox.candidates[0].client is candidate_client
    assert snapshot.sandbox.candidates[0].options is not candidate_options

    snapshot_options = cast(DockerSandboxClientOptions, snapshot.sandbox.options)
    snapshot_options.environment["API_KEY"] = "mutated"
    assert options.environment == {"API_KEY": "base"}

    snapshot_candidate_options = cast(DockerSandboxClientOptions, snapshot.sandbox.candidates[0].options)
    snapshot_candidate_options.environment["TOKEN"] = "mutated"
    assert candidate_options.environment == {"TOKEN": "candidate"}


async def test_runner_arun_uses_fresh_config_snapshot_for_each_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from troopai.adk.run import runner as runner_module

    seen: list[RunConfig] = []

    async def fake_resume_from_state(**kwargs: Any) -> str:
        seen.append(kwargs["config"])
        return "ok"

    monkeypatch.setattr(runner_module, "resume_from_state", fake_resume_from_state)

    assert await Runner.arun(_agent(), RunState()) == "ok"
    assert await Runner.arun(_agent(), RunState()) == "ok"

    assert seen[0] is not DEFAULT_RUN_CONFIG
    assert seen[1] is not DEFAULT_RUN_CONFIG
    assert seen[0] is not seen[1]
    seen[0].tracing_metadata["leaked"] = True
    assert "leaked" not in seen[1].tracing_metadata

    provided = RunConfig(tracing_metadata={"request": {"id": "r1"}})
    assert await Runner.arun(_agent(), RunState(), run_config=provided) == "ok"
    assert seen[2] is not provided
    seen[2].tracing_metadata["request"]["id"] = "mutated"
    assert provided.tracing_metadata["request"]["id"] == "r1"


def test_flow_runner_exposes_only_flow_relevant_profile_fluent_methods() -> None:
    flow_runner = Runner.configure().flow(_Flow(_FlowState()))

    assert flow_runner.context({"request_id": "flow"}).context_value == {"request_id": "flow"}
    assert not hasattr(flow_runner, "model")
    assert not hasattr(flow_runner, "limits")
    assert not hasattr(flow_runner, "max_total_turns")
    assert not hasattr(flow_runner, "arun_for_each")


def test_profile_sets_tenant_tool_allowlist_and_audit_sink() -> None:
    sink = InMemoryAuditSink()
    allowlist = {"t1": {"search"}}

    profile = Runner.configure().tenant_tool_allowlist(allowlist).audit(sink)

    assert profile.run_config.tenant_tool_allowlist == allowlist
    assert profile.run_config.audit_sink is sink


def test_profile_governance_defaults_are_none() -> None:
    cfg = Runner.configure().run_config

    assert cfg.tenant_tool_allowlist is None
    assert cfg.audit_sink is None


def test_profile_binds_target_specific_runners() -> None:
    profile = Runner.configure()
    task = _task()

    assert isinstance(profile.agent(_agent()), AgentRunner)
    assert isinstance(profile.swarm(_swarm()), SwarmRunner)
    assert isinstance(profile.graph(_graph()), GraphRunner)
    assert isinstance(profile.task(task), TaskRunner)
    assert isinstance(profile.pipeline(TaskPipeline(tasks=(task,))), TaskPipelineRunner)
    assert isinstance(profile.task_group(TaskGroup(tasks=(task,))), TaskGroupRunner)
    assert isinstance(profile.flow(_Flow(_FlowState())), FlowRunner)


async def test_agent_runner_delegates_to_runner_arun_with_profile_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    hooks = RunHooks()
    profile = Runner.configure().model("profile-model").context({"request_id": "r1"})
    called: dict[str, Any] = {}

    async def fake_arun(agent_arg: Agent[Any], prompt: str, **kwargs: Any) -> str:
        called["agent"] = agent_arg
        called["prompt"] = prompt
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(Runner, "arun", staticmethod(fake_arun))

    result = await profile.agent(agent).max_turns(6).hooks(hooks).arun("hello")

    assert result == "ok"
    assert called["agent"] is agent
    assert called["prompt"] == "hello"
    assert called["kwargs"]["context"] == {"request_id": "r1"}
    assert called["kwargs"]["hooks"] is hooks
    assert called["kwargs"]["max_turns"] == 6
    assert called["kwargs"]["run_config"].model == "profile-model"


async def test_swarm_runner_delegates_to_runner_arun_swarm(monkeypatch: pytest.MonkeyPatch) -> None:
    swarm = _swarm()
    profile = Runner.configure().model("profile-model").context({"request_id": "r2"})
    called: dict[str, Any] = {}

    async def fake_arun_swarm(swarm_arg: Swarm[Any], prompt: str, **kwargs: Any) -> str:
        called["swarm"] = swarm_arg
        called["prompt"] = prompt
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(Runner, "arun_swarm", staticmethod(fake_arun_swarm))

    result = await profile.swarm(swarm).max_total_turns(9).arun("go")

    assert result == "ok"
    assert called["swarm"] is swarm
    assert called["prompt"] == "go"
    assert called["kwargs"]["context"] == {"request_id": "r2"}
    assert called["kwargs"]["run_config"].model == "profile-model"
    assert called["kwargs"]["run_config"].max_total_turns == 9


async def test_graph_runner_delegates_to_runner_arun_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph()
    profile = Runner.configure().model("profile-model").context({"request_id": "r3"})
    called: dict[str, Any] = {}

    async def fake_arun_graph(graph_arg: Graph[Any], prompt: str, **kwargs: Any) -> str:
        called["graph"] = graph_arg
        called["prompt"] = prompt
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(Runner, "arun_graph", staticmethod(fake_arun_graph))

    result = await profile.graph(graph).thread("thread-1").arun("go")

    assert result == "ok"
    assert called["graph"] is graph
    assert called["prompt"] == "go"
    assert called["kwargs"]["context"] == {"request_id": "r3"}
    assert called["kwargs"]["thread_id"] == "thread-1"
    assert called["kwargs"]["run_config"].model == "profile-model"


async def test_task_pipeline_group_and_flow_runners_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _task()
    pipeline = TaskPipeline(tasks=(task,))
    group = TaskGroup(tasks=(task,))
    flow = _Flow(_FlowState())
    flow_config = FlowConfig(max_steps=3)
    profile = Runner.configure().context({"request_id": "r4"})
    called: dict[str, Any] = {}

    async def fake_arun_task(task_arg: Task[Any], **kwargs: Any) -> str:
        called["task"] = (task_arg, kwargs)
        return "task-ok"

    async def fake_arun_task_pipeline(pipeline_arg: TaskPipeline[Any], **kwargs: Any) -> str:
        called["pipeline"] = (pipeline_arg, kwargs)
        return "pipeline-ok"

    async def fake_arun_task_group(group_arg: TaskGroup[Any], **kwargs: Any) -> str:
        called["group"] = (group_arg, kwargs)
        return "group-ok"

    async def fake_arun_flow(flow_arg: Flow[Any], **kwargs: Any) -> str:
        called["flow"] = (flow_arg, kwargs)
        return "flow-ok"

    monkeypatch.setattr(Runner, "arun_task", staticmethod(fake_arun_task))
    monkeypatch.setattr(Runner, "arun_task_pipeline", staticmethod(fake_arun_task_pipeline))
    monkeypatch.setattr(Runner, "arun_task_group", staticmethod(fake_arun_task_group))
    monkeypatch.setattr(Runner, "arun_flow", staticmethod(fake_arun_flow))

    assert await profile.task(task).arun() == "task-ok"
    assert await profile.pipeline(pipeline).arun() == "pipeline-ok"
    assert await profile.task_group(group).arun() == "group-ok"
    assert await profile.flow(flow).config(flow_config).arun() == "flow-ok"

    task_call = cast("tuple[Task[Any], dict[str, Any]]", called["task"])
    pipeline_call = cast("tuple[TaskPipeline[Any], dict[str, Any]]", called["pipeline"])
    group_call = cast("tuple[TaskGroup[Any], dict[str, Any]]", called["group"])
    flow_call = cast("tuple[Flow[Any], dict[str, Any]]", called["flow"])

    assert task_call[0] is task
    assert task_call[1]["context"] == {"request_id": "r4"}
    assert pipeline_call[0] is pipeline
    assert pipeline_call[1]["context"] == {"request_id": "r4"}
    assert group_call[0] is group
    assert group_call[1]["context"] == {"request_id": "r4"}
    assert flow_call[0] is flow
    assert flow_call[1]["context"] == {"request_id": "r4"}
    assert flow_call[1]["config"] is flow_config
