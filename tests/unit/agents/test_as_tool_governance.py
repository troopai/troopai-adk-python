"""Tests for as_tool() governance features: timeout, budget, and introspection."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents import Agent
from troopai.adk.handoffs import Handoff
from troopai.adk.llms.llm_usage import LLMUsageLimits
from troopai.adk.tools.function_tool import FunctionTool

# ── Helpers ──────────────────────────────────────────────────────────


def _agent(name: str = "TestAgent") -> Agent:
    return Agent(name=name, system_prompt="You are a test agent.")


async def _invoke(tool: FunctionTool, raw_input: str) -> str:
    """Invoke a tool's on_invoke, asserting it exists first."""
    assert tool.on_invoke is not None, "Tool must have on_invoke set"
    ctx = MagicMock()
    ctx.context = None
    ctx.get_run_config.return_value = None
    return await tool.on_invoke(ctx, raw_input)


async def _invoke_with_config(
    tool: FunctionTool,
    raw_input: str,
    run_config: Any,
) -> str:
    """Invoke a tool's on_invoke with a custom run_config on the context."""
    assert tool.on_invoke is not None, "Tool must have on_invoke set"
    ctx = MagicMock()
    ctx.context = None
    ctx.get_run_config.return_value = run_config
    return await tool.on_invoke(ctx, raw_input)


def _mock_result(final_output: str = "Done") -> MagicMock:
    """Create a mock RunResult."""
    return MagicMock(requires_action=False, final_output=final_output)


# ── TestAsToolTimeout ────────────────────────────────────────────────


class TestAsToolTimeout:
    """Test the timeout parameter on as_tool()."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_on_slow_agent(self) -> None:
        """When sub-agent exceeds timeout, parent gets error string."""
        slow_agent = _agent("SlowAgent")
        tool = slow_agent.as_tool(timeout=0.01)

        async def slow_run(
            _agent: Any,
            _user_prompt: Any = None,
            **_kwargs: Any,
        ) -> MagicMock:
            await asyncio.sleep(5.0)
            return _mock_result()

        with patch("troopai.adk.run.Runner.arun", side_effect=slow_run):
            result = await _invoke(tool, '{"input": "do something"}')

        assert "timed out" in result
        assert "SlowAgent" in result

    @pytest.mark.asyncio
    async def test_timeout_none_does_not_apply(self) -> None:
        """When timeout is None (default), no timeout enforcement."""
        agent = _agent("FastAgent")
        tool = agent.as_tool()

        with patch(
            "troopai.adk.run.Runner.arun",
            new_callable=AsyncMock,
            return_value=_mock_result("Quick result"),
        ):
            result = await _invoke(tool, '{"input": "fast task"}')

        assert result == "Quick result"

    @pytest.mark.asyncio
    async def test_timeout_success_within_limit(self) -> None:
        """When sub-agent completes within timeout, result returns normally."""
        agent = _agent("QuickAgent")
        tool = agent.as_tool(timeout=5.0)

        with patch(
            "troopai.adk.run.Runner.arun",
            new_callable=AsyncMock,
            return_value=_mock_result("Completed in time"),
        ):
            result = await _invoke(tool, '{"input": "quick task"}')

        assert result == "Completed in time"


# ── TestAsToolBudget ─────────────────────────────────────────────────


class TestAsToolBudget:
    """Test the budget parameter on as_tool()."""

    @pytest.mark.asyncio
    async def test_budget_flows_to_run_config(self) -> None:
        """Budget parameter is merged into the RunConfig passed to Runner.arun."""
        agent = _agent("BudgetAgent")
        budget = LLMUsageLimits(total_tokens_limit=5_000, request_limit=3)
        tool = agent.as_tool(budget=budget)

        captured_kwargs: dict[str, Any] = {}

        async def capture_arun(
            _agent: Any,
            _user_prompt: Any = None,
            **kwargs: Any,
        ) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_result()

        with patch("troopai.adk.run.Runner.arun", side_effect=capture_arun):
            await _invoke(tool, '{"input": "budgeted task"}')

        run_config = captured_kwargs.get("run_config")
        assert run_config is not None
        assert run_config.usage_limits is budget

    @pytest.mark.asyncio
    async def test_budget_overrides_inherited_config(self) -> None:
        """Budget takes precedence over inherited RunConfig.usage_limits."""
        from troopai.adk.run.config import RunConfig

        agent = _agent("BudgetAgent")
        inherited_limits = LLMUsageLimits(total_tokens_limit=100_000)
        explicit_budget = LLMUsageLimits(total_tokens_limit=5_000)

        inherited_config = RunConfig(usage_limits=inherited_limits)
        tool = agent.as_tool(budget=explicit_budget)

        captured_kwargs: dict[str, Any] = {}

        async def capture_arun(
            _agent: Any,
            _user_prompt: Any = None,
            **kwargs: Any,
        ) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_result()

        with patch("troopai.adk.run.Runner.arun", side_effect=capture_arun):
            await _invoke_with_config(
                tool,
                '{"input": "task"}',
                inherited_config,
            )

        run_config = captured_kwargs.get("run_config")
        assert run_config is not None
        assert run_config.usage_limits is explicit_budget
        assert run_config.usage_limits.total_tokens_limit == 5_000

    @pytest.mark.asyncio
    async def test_no_budget_preserves_config(self) -> None:
        """Without budget param, RunConfig passes through unchanged."""
        from troopai.adk.run.config import RunConfig

        agent = _agent("NoBudgetAgent")
        original_config = RunConfig(verbose=True)
        tool = agent.as_tool(run_config=original_config)

        captured_kwargs: dict[str, Any] = {}

        async def capture_arun(
            _agent: Any,
            _user_prompt: Any = None,
            **kwargs: Any,
        ) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_result()

        with patch("troopai.adk.run.Runner.arun", side_effect=capture_arun):
            await _invoke(tool, '{"input": "task"}')

        run_config = captured_kwargs.get("run_config")
        assert run_config is original_config


# ── TestAsToolTimeoutAndBudget ───────────────────────────────────────


class TestAsToolTimeoutAndBudget:
    """Test timeout + budget combined."""

    @pytest.mark.asyncio
    async def test_both_timeout_and_budget(self) -> None:
        """Both timeout and budget can be set simultaneously."""
        agent = _agent("GovernedAgent")
        budget = LLMUsageLimits(total_tokens_limit=10_000)
        tool = agent.as_tool(timeout=5.0, budget=budget)

        captured_kwargs: dict[str, Any] = {}

        async def capture_arun(
            _agent: Any,
            _user_prompt: Any = None,
            **kwargs: Any,
        ) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _mock_result("Governed result")

        with patch("troopai.adk.run.Runner.arun", side_effect=capture_arun):
            result = await _invoke(tool, '{"input": "task"}')

        assert result == "Governed result"
        run_config = captured_kwargs.get("run_config")
        assert run_config is not None
        assert run_config.usage_limits is budget


# ── TestAsToolMaxResultTokens ────────────────────────────────────────


class TestAsToolMaxResultTokens:
    """Test max_result_tokens pass-through to FunctionTool."""

    def test_max_result_tokens_set_on_tool(self) -> None:
        """max_result_tokens flows through to the FunctionTool."""
        agent = _agent("TruncAgent")
        tool = agent.as_tool(max_result_tokens=500)
        assert tool.max_result_tokens == 500

    def test_max_result_tokens_default_none(self) -> None:
        """Without max_result_tokens, FunctionTool has None."""
        agent = _agent("NoTruncAgent")
        tool = agent.as_tool()
        assert tool.max_result_tokens is None


# ── TestGetDelegateTools ─────────────────────────────────────────────


class TestGetDelegateTools:
    """Test Agent.get_delegate_tools() introspection."""

    def test_returns_only_delegate_tools(self) -> None:
        """get_delegate_tools() returns only tools wrapping agents."""
        researcher = _agent("Researcher")
        writer = _agent("Writer")

        regular_tool = FunctionTool(
            name="calculator",
            schema={"type": "object", "properties": {}},
        )

        supervisor = Agent(
            name="Supervisor",
            system_prompt="Coordinate work.",
            tools=[
                regular_tool,
                researcher.as_tool(),
                writer.as_tool(),
            ],
        )

        delegates = supervisor.get_delegate_tools()
        assert len(delegates) == 2
        delegate_names = {t.name for t in delegates}
        assert "researcher" in delegate_names
        assert "writer" in delegate_names

    def test_empty_when_no_delegates(self) -> None:
        """get_delegate_tools() returns empty list when no delegates."""
        agent = Agent(
            name="Solo",
            system_prompt="Work alone.",
            tools=[FunctionTool(name="calc", schema={"type": "object", "properties": {}})],
        )
        assert agent.get_delegate_tools() == []

    def test_empty_when_no_tools(self) -> None:
        """get_delegate_tools() returns empty list when agent has no tools."""
        agent = _agent("Empty")
        assert agent.get_delegate_tools() == []


# ── TestGetAgentGraph ────────────────────────────────────────────────


class TestGetAgentGraph:
    """Test Agent.get_agent_graph() topology introspection."""

    def test_single_agent_no_delegates(self) -> None:
        """A solo agent has empty delegates and handoffs."""
        agent = _agent("Solo")
        graph = agent.get_agent_graph()
        assert graph == {
            "name": "Solo",
            "delegates": [],
            "handoffs": [],
        }

    def test_two_level_hierarchy(self) -> None:
        """Supervisor -> [Researcher, Writer] topology."""
        researcher = _agent("Researcher")
        writer = _agent("Writer")

        supervisor = Agent(
            name="Supervisor",
            system_prompt="Coordinate.",
            tools=[researcher.as_tool(), writer.as_tool()],
        )

        graph = supervisor.get_agent_graph()
        assert graph["name"] == "Supervisor"
        assert len(graph["delegates"]) == 2

        delegate_names = {d["name"] for d in graph["delegates"]}
        assert delegate_names == {"Researcher", "Writer"}

        for delegate in graph["delegates"]:
            assert delegate["delegates"] == []
            assert delegate["handoffs"] == []

    def test_three_level_hierarchy(self) -> None:
        """Supervisor -> Manager -> Worker topology."""
        worker = _agent("Worker")
        manager = Agent(
            name="Manager",
            system_prompt="Manage work.",
            tools=[worker.as_tool()],
        )
        supervisor = Agent(
            name="Supervisor",
            system_prompt="Oversee.",
            tools=[manager.as_tool()],
        )

        graph = supervisor.get_agent_graph()
        assert graph["name"] == "Supervisor"
        assert len(graph["delegates"]) == 1
        assert graph["delegates"][0]["name"] == "Manager"
        assert len(graph["delegates"][0]["delegates"]) == 1
        assert graph["delegates"][0]["delegates"][0]["name"] == "Worker"

    def test_includes_handoff_targets(self) -> None:
        """Graph includes handoff target names."""
        support = _agent("Support")
        billing = _agent("Billing")

        triage = Agent(
            name="Triage",
            system_prompt="Route requests.",
            handoffs=[support, billing],
        )

        graph = triage.get_agent_graph()
        assert graph["name"] == "Triage"
        assert set(graph["handoffs"]) == {"Support", "Billing"}

    def test_handoff_objects_included(self) -> None:
        """Handoff wrapper objects are resolved to target names."""
        refunds = _agent("Refunds")

        triage = Agent(
            name="Triage",
            system_prompt="Route.",
            handoffs=[
                Handoff(target=refunds, description="Handle refunds"),
            ],
        )

        graph = triage.get_agent_graph()
        assert graph["handoffs"] == ["Refunds"]

    def test_cycle_detection(self) -> None:
        """Circular references produce (cycle) marker without infinite recursion."""
        agent_b = _agent("AgentB")
        agent_a = Agent(
            name="AgentA",
            system_prompt="Do A.",
            tools=[agent_b.as_tool()],
        )
        object.__setattr__(agent_b, "tools", [agent_a.as_tool()])

        graph = agent_a.get_agent_graph()
        assert graph["name"] == "AgentA"
        assert len(graph["delegates"]) == 1
        assert graph["delegates"][0]["name"] == "AgentB"
        assert graph["delegates"][0]["delegates"][0]["handoffs"] == ["(cycle)"]

    def test_mixed_delegates_and_handoffs(self) -> None:
        """Graph captures both delegate tools and handoff targets."""
        researcher = _agent("Researcher")
        support = _agent("Support")

        supervisor = Agent(
            name="Supervisor",
            system_prompt="Coordinate.",
            tools=[researcher.as_tool()],
            handoffs=[support],
        )

        graph = supervisor.get_agent_graph()
        assert len(graph["delegates"]) == 1
        assert graph["delegates"][0]["name"] == "Researcher"
        assert graph["handoffs"] == ["Support"]


# ── TestDescription ──────────────────────────────────────────────────


class TestDescription:
    """Test the description attribute and its fallback chains."""

    def test_description_default_none(self) -> None:
        """description defaults to None."""
        agent = _agent("Solo")
        assert agent.description is None

    def test_description_set(self) -> None:
        """description stores the provided value."""
        agent = Agent(
            name="Researcher",
            description="Research and summarize findings.",
            system_prompt="You are a researcher.",
        )
        assert agent.description == "Research and summarize findings."

    def test_as_tool_uses_description_as_fallback(self) -> None:
        """as_tool() uses agent.description when tool_description not set."""
        agent = Agent(
            name="Researcher",
            description="Research and summarize findings.",
            system_prompt="You are a researcher.",
        )
        tool = agent.as_tool()
        assert tool.description == "Research and summarize findings."

    def test_as_tool_tool_description_overrides(self) -> None:
        """Explicit tool_description overrides agent.description."""
        agent = Agent(
            name="Researcher",
            description="Research and summarize findings.",
            system_prompt="You are a researcher.",
        )
        tool = agent.as_tool(tool_description="Custom description.")
        assert tool.description == "Custom description."

    def test_as_tool_no_description_auto_generates(self) -> None:
        """Without description or tool_description, auto-generates from name."""
        agent = _agent("Researcher")
        tool = agent.as_tool()
        assert tool.description == "Delegate a task to the Researcher agent."

    def test_handoff_uses_target_description(self) -> None:
        """Handoff.get_description() uses target.description as fallback."""
        target = Agent(
            name="Support",
            description="Handle customer support requests.",
            system_prompt="You help customers.",
        )
        h = Handoff(target=target)
        assert h.get_description() == "Handle customer support requests."

    def test_handoff_description_overrides_target(self) -> None:
        """Explicit Handoff.description overrides target.description."""
        target = Agent(
            name="Support",
            description="Handle customer support requests.",
            system_prompt="You help customers.",
        )
        h = Handoff(target=target, description="Route to support team.")
        assert h.get_description() == "Route to support team."

    def test_handoff_no_description_auto_generates(self) -> None:
        """Without any description, auto-generates from target name."""
        target = _agent("Support")
        h = Handoff(target=target)
        assert h.get_description() == "Handoff to the Support agent to handle the request."
