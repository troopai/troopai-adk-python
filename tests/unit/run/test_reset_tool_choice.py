"""Tests for reset_tool_choice — prevents REQUIRED + run_llm_again infinite loop.

When ``LLMConfig.reset_tool_choice`` is ``None`` (default, treated as ``True``)
and the agent uses ``tool_choice="required"``, the agent loop resets tool_choice
to ``"auto"`` after tools execute. This lets the LLM respond with text on the
next turn instead of being forced into another tool call.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.run.llm_calls import resolve_llm_config

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(
    llm_config: LLMConfig | None = None,
    tools: list | None = None,
):
    """Minimal agent-like object for testing."""
    from troopai.adk.agents.agent_guardrails import AgentGuardrails
    from troopai.adk.agents.middleware import Middleware
    from troopai.adk.skills.activation import SkillActivation

    return SimpleNamespace(
        name="test_agent",
        tools=tools or [],
        llm_config=llm_config,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        output_schema=None,
        guardrails=AgentGuardrails(),
        system_prompt="You are a test agent.",
        skills=[],
        skill_activation=SkillActivation.EAGER,
        hooks=None,
        middleware=Middleware(),
    )


# ── resolve_llm_config tests ────────────────────────────────────────


class TestResolveLLMConfig:
    def test_no_override_required(self) -> None:
        """Without override, tool_choice='required' is preserved."""
        agent = _make_agent(LLMConfig(tool_choice="required"))
        result = resolve_llm_config(agent)
        assert result is not None
        assert result.tool_choice == "required"

    def test_no_override_auto(self) -> None:
        """Without override, tool_choice='auto' is preserved."""
        agent = _make_agent(LLMConfig(tool_choice="auto"))
        result = resolve_llm_config(agent)
        assert result is not None
        assert result.tool_choice == "auto"

    def test_no_override_none_config(self) -> None:
        """Without override and no llm_config, returns None."""
        agent = _make_agent(llm_config=None)
        result = resolve_llm_config(agent)
        assert result is None

    def test_no_override_disabled(self) -> None:
        """Without override, tool_choice='none' is preserved."""
        agent = _make_agent(LLMConfig(tool_choice="none"))
        result = resolve_llm_config(agent)
        assert result is not None
        assert result.tool_choice == "none"

    def test_override_auto_overrides_required(self) -> None:
        """'auto' override takes precedence over agent's 'required'."""
        agent = _make_agent(LLMConfig(tool_choice="required"))
        result = resolve_llm_config(agent, tool_choice_override="auto")
        assert result is not None
        assert result.tool_choice == "auto"

    def test_override_required_overrides_auto(self) -> None:
        """'required' override takes precedence over agent's 'auto'."""
        agent = _make_agent(LLMConfig(tool_choice="auto"))
        result = resolve_llm_config(agent, tool_choice_override="required")
        assert result is not None
        assert result.tool_choice == "required"

    def test_override_with_no_llm_config(self) -> None:
        """Override works even when agent has no llm_config."""
        agent = _make_agent(llm_config=None)
        result = resolve_llm_config(agent, tool_choice_override="required")
        assert result is not None
        assert result.tool_choice == "required"

    def test_parallel_preserved_with_override(self) -> None:
        """Override affects tool_choice but tool_execution_mode is unchanged."""
        agent = _make_agent(
            LLMConfig(
                tool_choice="required",
                tool_execution_mode="parallel",
            )
        )
        result = resolve_llm_config(agent, tool_choice_override="auto")
        assert result is not None
        assert result.tool_choice == "auto"
        assert result.tool_execution_mode == "parallel"


# ── LLMConfig tool defaults ─────────────────────────────────────────


class TestLLMConfigToolDefaults:
    def test_reset_tool_choice_default_none(self) -> None:
        """reset_tool_choice defaults to None (treated as True by the Runner)."""
        config = LLMConfig()
        assert config.reset_tool_choice is None

    def test_reset_tool_choice_explicit_false(self) -> None:
        """Can explicitly set reset_tool_choice to False."""
        config = LLMConfig(reset_tool_choice=False)
        assert config.reset_tool_choice is False

    def test_reset_tool_choice_explicit_true(self) -> None:
        """Can explicitly set reset_tool_choice to True."""
        config = LLMConfig(reset_tool_choice=True)
        assert config.reset_tool_choice is True


# ── Agent loop integration tests ─────────────────────────────────────


class TestResetToolChoiceInLoop:
    @pytest.mark.asyncio
    async def test_required_with_reset_uses_auto_on_second_call(self) -> None:
        """REQUIRED + reset_tool_choice=True → second LLM call uses 'auto'."""
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseFunctionToolCall,
            LLMResponseText,
        )

        tool = FunctionTool(
            name="get_data",
            description="Get data",
            schema={"type": "object", "properties": {}},
            on_invoke=AsyncMock(return_value="data_result"),
        )
        agent = _make_agent(
            llm_config=LLMConfig(
                tool_choice="required",
                reset_tool_choice=True,
            ),
            tools=[tool],
        )

        # First call returns a tool call; second call returns text
        llm_call_args = []
        call_count = 0

        async def mock_call_llm(
            _agent, _messages, _config, _context=None, _tool_failure_counts=None, tool_choice_override=None, **_kwargs
        ):
            nonlocal call_count
            call_count += 1
            llm_call_args.append(tool_choice_override)

            if call_count == 1:
                # First call: return a tool call
                return LLMResponse(
                    response_id="resp_1",
                    model="test",
                    response=[
                        LLMResponseFunctionToolCall(
                            call_id="call_1",
                            name="get_data",
                            arguments="{}",
                        ),
                    ],
                )
            else:
                # Second call: return text (no tool calls)
                return LLMResponse(
                    response_id="resp_2",
                    model="test",
                    response=[LLMResponseText(text="Here is the data.")],
                )

        with patch("troopai.adk.run.loop.call_llm", side_effect=mock_call_llm):
            from troopai.adk.hooks.hooks import RunHooks
            from troopai.adk.run.config import DEFAULT_RUN_CONFIG
            from troopai.adk.run.context import RunContext
            from troopai.adk.run.loop import run_agent_loop

            ctx = RunContext(context=None)
            result = await run_agent_loop(
                agent=agent,
                user_prompt="Get some data",
                context=ctx,
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                max_turns=5,
                config=DEFAULT_RUN_CONFIG,
            )

        assert call_count == 2
        # First LLM call: no override (uses agent's "required")
        assert llm_call_args[0] is None
        # Second LLM call: override to "auto" (reset_tool_choice kicked in)
        assert llm_call_args[1] == "auto"
        assert result.final_output == "Here is the data."

    @pytest.mark.asyncio
    async def test_required_without_reset_stays_required(self) -> None:
        """REQUIRED + reset_tool_choice=False → second LLM call still uses REQUIRED."""
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseFunctionToolCall,
            LLMResponseText,
        )

        tool = FunctionTool(
            name="get_data",
            description="Get data",
            schema={"type": "object", "properties": {}},
            on_invoke=AsyncMock(return_value="data_result"),
        )
        agent = _make_agent(
            llm_config=LLMConfig(
                tool_choice="required",
                reset_tool_choice=False,
            ),
            tools=[tool],
        )

        llm_call_args = []
        call_count = 0

        async def mock_call_llm(
            _agent, _messages, _config, _context=None, _tool_failure_counts=None, tool_choice_override=None, **_kwargs
        ):
            nonlocal call_count
            call_count += 1
            llm_call_args.append(tool_choice_override)

            if call_count <= 2:
                return LLMResponse(
                    response_id=f"resp_{call_count}",
                    model="test",
                    response=[
                        LLMResponseFunctionToolCall(
                            call_id=f"call_{call_count}",
                            name="get_data",
                            arguments="{}",
                        ),
                    ],
                )
            else:
                return LLMResponse(
                    response_id=f"resp_{call_count}",
                    model="test",
                    response=[LLMResponseText(text="Done.")],
                )

        with patch("troopai.adk.run.loop.call_llm", side_effect=mock_call_llm):
            from troopai.adk.hooks.hooks import RunHooks
            from troopai.adk.run.config import DEFAULT_RUN_CONFIG
            from troopai.adk.run.context import RunContext
            from troopai.adk.run.loop import run_agent_loop

            ctx = RunContext(context=None)
            _ = await run_agent_loop(
                agent=agent,
                user_prompt="Get some data",
                context=ctx,
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                max_turns=5,
                config=DEFAULT_RUN_CONFIG,
            )

        # With reset_tool_choice=False, no override is applied
        assert llm_call_args[0] is None
        assert llm_call_args[1] is None

    @pytest.mark.asyncio
    async def test_auto_strategy_no_reset(self) -> None:
        """'auto' tool_choice → reset_tool_choice has no effect (no override ever set)."""
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseFunctionToolCall,
            LLMResponseText,
        )

        tool = FunctionTool(
            name="get_data",
            description="Get data",
            schema={"type": "object", "properties": {}},
            on_invoke=AsyncMock(return_value="data_result"),
        )
        agent = _make_agent(
            llm_config=LLMConfig(tool_choice="auto"),
            tools=[tool],
        )

        llm_call_args = []
        call_count = 0

        async def mock_call_llm(
            _agent, _messages, _config, _context=None, _tool_failure_counts=None, tool_choice_override=None, **_kwargs
        ):
            nonlocal call_count
            call_count += 1
            llm_call_args.append(tool_choice_override)

            if call_count == 1:
                return LLMResponse(
                    response_id="resp_1",
                    model="test",
                    response=[
                        LLMResponseFunctionToolCall(
                            call_id="call_1",
                            name="get_data",
                            arguments="{}",
                        ),
                    ],
                )
            else:
                return LLMResponse(
                    response_id="resp_2",
                    model="test",
                    response=[LLMResponseText(text="Result.")],
                )

        with patch("troopai.adk.run.loop.call_llm", side_effect=mock_call_llm):
            from troopai.adk.hooks.hooks import RunHooks
            from troopai.adk.run.config import DEFAULT_RUN_CONFIG
            from troopai.adk.run.context import RunContext
            from troopai.adk.run.loop import run_agent_loop

            ctx = RunContext(context=None)
            await run_agent_loop(
                agent=agent,
                user_prompt="Get some data",
                context=ctx,
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                max_turns=5,
                config=DEFAULT_RUN_CONFIG,
            )

        # "auto" tool_choice: reset_tool_choice has no effect
        assert llm_call_args[0] is None
        assert llm_call_args[1] is None

    @pytest.mark.asyncio
    async def test_hitl_rejection_with_reset_uses_auto(self) -> None:
        """REQUIRED + HITL rejection + reset_tool_choice=True → resumed loop uses 'auto'."""
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseFunctionToolCall,
            LLMResponseText,
        )

        tool = FunctionTool(
            name="dangerous_action",
            description="A dangerous action",
            schema={"type": "object", "properties": {}},
            on_invoke=AsyncMock(return_value="executed"),
            requires_approval=True,
        )
        agent = _make_agent(
            llm_config=LLMConfig(
                tool_choice="required",
                reset_tool_choice=True,
            ),
            tools=[tool],
        )

        # First: run the agent loop, which should defer for approval
        llm_call_args = []
        call_count = 0

        async def mock_call_llm(
            _agent, _messages, _config, _context=None, _tool_failure_counts=None, tool_choice_override=None, **_kwargs
        ):
            nonlocal call_count
            call_count += 1
            llm_call_args.append(tool_choice_override)

            if call_count == 1:
                # First call: tool call that will be deferred
                return LLMResponse(
                    response_id="resp_1",
                    model="test",
                    response=[
                        LLMResponseFunctionToolCall(
                            call_id="call_1",
                            name="dangerous_action",
                            arguments="{}",
                        ),
                    ],
                )
            else:
                # After rejection + resumed loop: respond with text
                return LLMResponse(
                    response_id="resp_2",
                    model="test",
                    response=[LLMResponseText(text="OK, I won't do that.")],
                )

        with patch("troopai.adk.run.loop.call_llm", side_effect=mock_call_llm):
            from troopai.adk.hooks.hooks import RunHooks
            from troopai.adk.run.config import DEFAULT_RUN_CONFIG
            from troopai.adk.run.context import RunContext
            from troopai.adk.run.loop import run_agent_loop

            ctx = RunContext(context=None)
            result = await run_agent_loop(
                agent=agent,
                user_prompt="Do the dangerous thing",
                context=ctx,
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                max_turns=5,
                config=DEFAULT_RUN_CONFIG,
            )

        # Should have deferred (HITL interrupt)
        assert result.requires_action is True
        assert result.state is not None

        # Now reject the tool and resume
        state = result.state
        deferred = state.deferred_tool_requests.approvals[0]
        state.reject(deferred, "No, that's too dangerous.")

        # Reset call tracking for resumed loop
        resumed_call_args = []

        async def mock_call_llm_resumed(
            _agent, _messages, _config, _context=None, _tool_failure_counts=None, tool_choice_override=None, **_kwargs
        ):
            resumed_call_args.append(tool_choice_override)
            # After rejection, just respond with text
            return LLMResponse(
                response_id="resp_resumed",
                model="test",
                response=[LLMResponseText(text="OK, I won't do that.")],
            )

        with patch("troopai.adk.run.loop.call_llm", side_effect=mock_call_llm_resumed):
            from troopai.adk.run.resumption import resume_from_state

            resumed_result = await resume_from_state(
                agent=agent,
                state=state,
                max_turns=5,
                config=DEFAULT_RUN_CONFIG,
            )

        # The resumed LLM call should use "auto" (not "required")
        assert len(resumed_call_args) >= 1
        assert resumed_call_args[0] == "auto"
        assert resumed_result.final_output == "OK, I won't do that."
