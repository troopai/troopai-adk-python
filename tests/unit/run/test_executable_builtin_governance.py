"""Governance parity tests for framework-executed built-in tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from troopai.adk.agents.middleware import Middleware
from troopai.adk.audit.sink import InMemoryAuditSink
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import build_tools
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.skills.skill import Skill, SkillGovernance
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _agent(*, tools: list[Any] | None = None, skills: list[Skill] | None = None) -> Any:
    return SimpleNamespace(
        name="test_agent",
        tools=tools if tools is not None else [],
        skills=skills if skills is not None else [],
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        llm_config=None,
        output_schema=None,
        hooks=None,
        middleware=Middleware(),
    )


def _call(name: str) -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id="call-1", name=name, arguments="{}")


async def _execute(agent: Any, name: str, config: RunConfig | None = None) -> str:
    results, deferred = await execute_tool_calls(
        agent=agent,
        tool_calls=[_call(name)],
        ctx_wrapper=RunContext(context=None),
        hooks=RunHooks(),
        config=config if config is not None else RunConfig(),
        model="test-model",
    )
    assert deferred is None
    assert len(results) == 1
    return str(results[0].output)


@dataclass(kw_only=True)
class _StatefulBuiltin(ExecutableBuiltinTool):
    calls: int = 0

    def __post_init__(self) -> None:
        async def invoke(ctx: ToolContext[Any], _raw_args: str) -> str:
            assert isinstance(ctx, ToolContext)
            self.calls += 1
            return "ran"

        self.on_invoke = invoke


class TestExecutableBuiltinGovernance:
    async def test_build_tools_normalizes_builtin_to_function_tool(self) -> None:
        builtin = _StatefulBuiltin(name="memory", description="Store memory", schema=_SCHEMA)

        tools = await build_tools(_agent(tools=[builtin]), RunContext(context=None))

        assert tools is not None
        assert len(tools) == 1
        assert isinstance(tools[0], FunctionTool)
        assert tools[0].name == "memory"

    async def test_permission_denial_prevents_builtin_invocation(self) -> None:
        builtin = _StatefulBuiltin(name="memory", description="Store memory", schema=_SCHEMA)

        output = await _execute(
            _agent(tools=[builtin]),
            "memory",
            RunConfig(can_use_tool=lambda _agent, _name, _ctx: False),
        )

        assert "Permission denied" in output
        assert builtin.calls == 0

    async def test_shared_executor_preserves_builtin_state_and_context(self) -> None:
        builtin = _StatefulBuiltin(name="memory", description="Store memory", schema=_SCHEMA)
        agent = _agent(tools=[builtin])

        assert await _execute(agent, "memory") == "ran"
        assert await _execute(agent, "memory") == "ran"
        assert builtin.calls == 2

    async def test_skill_governance_applies_to_adapted_builtin(self) -> None:
        async def invoke(_ctx: ToolContext[Any], _raw_args: str) -> str:
            return "x" * 1000

        builtin = ExecutableBuiltinTool(
            name="skill_memory",
            description="Store skill memory",
            schema=_SCHEMA,
            on_invoke=invoke,
        )
        skill_tools: list[Any] = [builtin]
        skill = Skill(
            name="memory_skill",
            description="Memory tools",
            tools=skill_tools,
            governance=SkillGovernance(max_result_tokens=5),
        )

        output = await _execute(_agent(skills=[skill]), "skill_memory")

        assert "[Result truncated" in output

    async def test_builtin_approval_defers_before_invocation(self) -> None:
        builtin = _StatefulBuiltin(
            name="memory",
            description="Store memory",
            schema=_SCHEMA,
            requires_approval=True,
        )

        results, deferred = await execute_tool_calls(
            agent=_agent(tools=[builtin]),
            tool_calls=[_call("memory")],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=RunConfig(),
            model="test-model",
        )

        assert results == []
        assert deferred is not None
        assert len(deferred.approvals) == 1
        assert deferred.approvals[0].tool_name == "memory"
        assert builtin.calls == 0

    async def test_successful_builtin_execution_is_audited(self) -> None:
        builtin = _StatefulBuiltin(name="memory", description="Store memory", schema=_SCHEMA)
        sink = InMemoryAuditSink()

        assert await _execute(_agent(tools=[builtin]), "memory", RunConfig(audit_sink=sink)) == "ran"

        assert len(sink.events) == 1
        assert sink.events[0].tool_name == "memory"
        assert sink.events[0].outcome == "ok"

    async def test_disabled_builtin_is_not_executed(self) -> None:
        builtin = _StatefulBuiltin(
            name="memory",
            description="Store memory",
            schema=_SCHEMA,
            enabled=False,
        )

        output = await _execute(_agent(tools=[builtin]), "memory")

        assert "disabled" in output.lower()
        assert builtin.calls == 0
