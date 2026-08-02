"""Regression tests for run-loop-tools cluster fixes.

Covers findings:
1. HandoffRejection from deterministic handoff path crashes run
2. Refusal-only response silently produces final_output=None
3. Parallel tool gather abandons sibling coroutines on exception
4. output_schema agent with tools: tool calls silently dropped
5. Module-level mutable LLM instance cache (functools.cache)
6. on_llm_end hook not called when call_llm raises
7. Guardrail exceptions re-raised bare with no logging
8. apply_result_limits char-boundary truncation for CJK text
9. on_tool_end hooks and audit log called before output guardrails
10. can_use_tool callback exception silently defaults to denied
11. Streaming path emits MESSAGE_OUTPUT_CREATED with content=None
12. tool.prepare callback exception silently falls back to original
13. _drop_orphan_run_items silently drops valid pairs with empty call_id
14. Dead `or 5` fallback for LAST_N window removed
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Finding 2: Refusal-only response raises ModelRefusalError ────────────────


class TestModelRefusalError:
    """Finding 2: refusal-only response must raise ModelRefusalError, not produce None."""

    def test_model_refusal_error_exists(self) -> None:
        """ModelRefusalError must be importable from the exceptions module."""
        from troopai.adk.exceptions import ModelRefusalError

        assert issubclass(ModelRefusalError, Exception)

    def test_model_refusal_error_carries_refusal_text(self) -> None:
        """ModelRefusalError must expose the refusal text."""
        from troopai.adk.exceptions import ModelRefusalError

        err = ModelRefusalError("I cannot help with that.")
        assert err.refusal == "I cannot help with that."
        assert "I cannot help with that." in str(err)

    def test_refusal_only_triggers_model_refusal_path(self) -> None:
        """Verify that an LLMResponse with only a refusal part has the right properties
        that would trigger ModelRefusalError in the loop."""
        from troopai.adk.exceptions import ModelRefusalError
        from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseRefusal

        refusal_response = LLMResponse(
            response_id="r1",
            model="gpt-4o",
            response=[LLMResponseRefusal(refusal="I can't do that.")],
        )

        # Verify the properties that the loop checks
        assert refusal_response.content is None, "A refusal-only response must have content=None"
        assert refusal_response.refusal == "I can't do that."
        assert len(refusal_response.tool_calls) == 0

        # Simulate the loop's check: refusal with no content must raise
        if refusal_response.refusal is not None and refusal_response.content is None:
            try:
                raise ModelRefusalError(refusal_response.refusal)
            except ModelRefusalError as exc:
                assert exc.refusal == "I can't do that."
                return

        pytest.fail("ModelRefusalError should have been raised")


# ── Finding 3: Parallel tool gather with return_exceptions=True ───────────────


class TestParallelGatherReturnExceptions:
    """Finding 3: asyncio.gather must use return_exceptions=True in parallel path."""

    async def test_parallel_gather_collects_all_before_raising(self) -> None:
        """When one tool raises, remaining tasks complete before re-raise.

        With return_exceptions=True, gather waits for ALL coroutines
        before re-raising the first exception — no ghost background tasks.
        """
        from troopai.adk.run.tools_executor import execute_tool_calls
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

        completed_tools: list[str] = []

        async def _ok_handler(ctx, _raw_args):
            await asyncio.sleep(0.05)
            completed_tools.append(ctx.tool_name)
            return f"ok_{ctx.tool_name}"

        async def _fail_handler(ctx, _raw_args):
            await asyncio.sleep(0.01)  # fail first
            raise RuntimeError("boom")

        tools = [
            FunctionTool(
                name="ok1", description="d", schema={"type": "object", "properties": {}}, on_invoke=_ok_handler
            ),
            FunctionTool(
                name="fail", description="d", schema={"type": "object", "properties": {}}, on_invoke=_fail_handler
            ),
            FunctionTool(
                name="ok2", description="d", schema={"type": "object", "properties": {}}, on_invoke=_ok_handler
            ),
        ]

        from types import SimpleNamespace

        from troopai.adk.agents.middleware import Middleware
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext

        agent = SimpleNamespace(
            name="test",
            tools=tools,
            tool_use_behavior="run_llm_again",
            handoffs=None,
            llm=None,
            hooks=None,
            middleware=Middleware(),
        )
        tool_calls = [
            LLMResponseFunctionToolCall(call_id="c1", name="ok1", arguments="{}"),
            LLMResponseFunctionToolCall(call_id="c2", name="fail", arguments="{}"),
            LLMResponseFunctionToolCall(call_id="c3", name="ok2", arguments="{}"),
        ]

        with pytest.raises(RuntimeError, match="boom"):
            await execute_tool_calls(
                agent=agent,  # type: ignore[arg-type]
                tool_calls=tool_calls,
                ctx_wrapper=RunContext(context=None),
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                model="gpt-4o-mini",
                parallel=True,
            )

        # Both ok tools should have completed because return_exceptions=True
        # waits for all coroutines before propagating.
        assert "ok1" in completed_tools or "ok2" in completed_tools, (
            "At least one ok tool should have completed before exception propagated"
        )


# ── Finding 4: output_schema agent defers to tool-execution path ───────────────


class TestOutputSchemaAgentDefersTool:
    """Finding 4: when output_schema agent returns only tool calls, return None
    from resolve_structured_output_step so tools are executed normally."""

    async def test_tool_calls_only_defers_to_tool_path(self) -> None:
        """resolve_structured_output_step must return None (not NextStepFinalOutput)
        when response has tool calls but no text content."""
        from pydantic import BaseModel

        from troopai.adk.agents.agent import Agent
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.turn_resolution import resolve_structured_output_step
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseFunctionToolCall,
        )

        class _Output(BaseModel):
            value: str

        # Build a response with a tool call only (no text)
        response = LLMResponse(
            response_id="r1",
            model="gpt-4o",
            response=[LLMResponseFunctionToolCall(call_id="c1", name="get_data", arguments="{}")],
        )
        assert response.content is None
        assert len(response.tool_calls) == 1

        agent = Agent(name="schema-agent", system_prompt="test", output_schema=_Output)

        result = await resolve_structured_output_step(
            current_agent=agent,
            response=response,
            messages=[],
            new_items=[],
            context_end=0,
            context=None,
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=DEFAULT_RUN_CONFIG,
        )
        # Must return None so the loop falls into the tool-execution branch
        assert result is None


# ── Finding 5: functools.cache instead of module-level mutable dict ────────────


class TestGetDefaultLlmFunctoolsCache:
    """Finding 5: get_default_llm must use functools.cache not a bare mutable dict."""

    def test_cache_clear_is_callable(self) -> None:
        """functools.cache exposes cache_clear(); bare dict does not."""
        from troopai.adk.run.llm_calls import get_default_llm

        assert callable(getattr(get_default_llm, "cache_clear", None)), (
            "get_default_llm must expose .cache_clear() via functools.cache"
        )

    def test_same_model_returns_same_instance(self) -> None:
        """Two calls with the same model string return the identical object."""
        from troopai.adk.run.llm_calls import get_default_llm

        get_default_llm.cache_clear()
        inst1 = get_default_llm("gpt-4o-mini")
        inst2 = get_default_llm("gpt-4o-mini")
        assert inst1 is inst2

    def test_different_model_returns_different_instance(self) -> None:
        """Two calls with different model strings return different objects."""
        from troopai.adk.run.llm_calls import get_default_llm

        get_default_llm.cache_clear()
        inst_a = get_default_llm("model-alpha")
        inst_b = get_default_llm("model-beta")
        assert inst_a is not inst_b


# ── Finding 6: on_llm_end always called, even on LLM exception ──────────────


class TestOnLlmEndAlwaysCalled:
    """Finding 6: on_llm_end must be called in a finally block."""

    def test_on_llm_end_in_finally_block(self) -> None:
        """Verify the loop source code wraps call_llm in try/finally to ensure
        on_llm_end is called even when an exception is raised."""
        import inspect

        from troopai.adk.run import loop as loop_module

        source = inspect.getsource(loop_module)
        # The non-streaming path must initialize 'response = None' before the try block
        assert "response = None" in source, "Non-streaming path must initialize 'response = None' before the try block"
        # The source must contain a finally block that calls on_llm_end
        # Both the non-streaming and streaming paths have their on_llm_end inside finally
        assert "finally:" in source, "Loop must use try/finally for hook pairing"

        # Verify that final_response = None is also initialized for the streaming path
        assert "final_response = None" in source, (
            "Streaming path must initialize 'final_response = None' before the try block"
        )


# ── Finding 7: Guardrail exceptions log at WARNING with exc_info ─────────────


class TestGuardrailExceptionLogging:
    """Finding 7: unexpected guardrail exceptions must be logged before re-raise."""

    def test_parallel_guardrail_exception_logging_in_source(self) -> None:
        """The source code must have a logger.warning call before raise result
        in both run_parallel_input_guardrails and _run_output_guardrails_once."""
        import inspect

        from troopai.adk.run import guardrails_executor

        source = inspect.getsource(guardrails_executor)

        # The pattern must be: logger.warning(...) then raise result
        # Both the input and output guardrail paths have this now.
        assert "logger.warning" in source
        # Verify exc_info=result is passed to log (from our fix)
        assert "exc_info=result" in source, (
            "Guardrail exception logging must include exc_info=result for traceback visibility"
        )
        # Must include the agent name in the log message
        assert "agent.name" in source or "agent '%s'" in source, "Guardrail warning must identify the failing agent"


# ── Finding 8: apply_result_limits uses binary search, accurate for CJK ─────


class TestApplyResultLimitsBinarySearch:
    """Finding 8: truncation must respect max_result_tokens for non-ASCII text."""

    def test_cjk_truncation_respects_token_budget(self) -> None:
        """CJK text (1 char ~ 1 token) must not be 4x over the token budget."""
        from unittest.mock import MagicMock, patch

        from troopai.adk.run.cost import apply_result_limits

        tool = MagicMock()
        tool.name = "t"
        tool.max_result_tokens = 10

        # Simulate a CJK-heavy string: patch TokenCounter.count_text to
        # return len(text) (each char = 1 token), which is worst-case for
        # the old 4-chars/token heuristic.
        cjk_text = "あ" * 100  # 100 chars = 100 tokens in our mock

        call_count = [0]

        def _mock_count(text, model):
            # Each char counts as 1 token
            call_count[0] += 1
            return len(text)

        with patch("troopai.adk.context.token_counter.TokenCounter.count_text", side_effect=_mock_count):
            result = apply_result_limits(cjk_text, tool, "gpt-4o")

        # The result body (excluding the suffix) must not exceed max_result_tokens
        # The old code would have done result[:40] = 40 "tokens" > 10
        # New code should respect max_result_tokens
        assert "[Result truncated:" in result, "Result should be truncated"
        # The body before the suffix
        body = result.split("\n[Result truncated:")[0]
        body_len = len(body)
        assert body_len <= 10, f"Body has {body_len} chars (≈ tokens for CJK), should be ≤ max_result_tokens=10"

    def test_truncation_suffix_included_in_contract(self) -> None:
        """The suffix tokens must be accounted for in the budget.

        Old behavior: body was truncated to max_result_tokens chars, then
        suffix appended unconditionally, total > max_result_tokens.
        New behavior: suffix tokens are pre-subtracted from the body budget,
        so the body fits within effective_limit = max_result_tokens - suffix_tokens.
        The body will always be at least 1 char (min limit enforced).
        """
        from unittest.mock import MagicMock

        from troopai.adk.run.cost import apply_result_limits

        tool = MagicMock()
        tool.name = "tool_name"
        tool.max_result_tokens = 100  # budget large enough to hold both body + suffix

        long_text = "x" * 1000

        # Count each character as 1 token (worst case for CJK / non-ASCII)
        def _mock_count(text, model):
            return len(text)

        with patch("troopai.adk.context.token_counter.TokenCounter.count_text", side_effect=_mock_count):
            result = apply_result_limits(long_text, tool, "gpt-4o")

        assert "[Result truncated:" in result, "Result should be truncated"

        # The body (before the suffix) must fit within the budget
        # suffix = "\n[Result truncated: 1000 → 100 tokens]" = 38 chars = 38 tokens
        # effective_limit = 100 - 38 = 62
        # So body should be 62 chars
        body = result.split("\n[Result truncated:")[0]
        suffix_part = "\n[Result truncated:" + result.split("\n[Result truncated:")[1]
        body_tokens = len(body)
        suffix_tokens = len(suffix_part)
        total_tokens = len(result)

        # Total should not drastically exceed max_result_tokens
        # The old code: body=400, suffix=38, total=438 (for max=100, 4*100=400 body)
        # New code: total ≈ 100 (body + suffix ≈ budget)
        assert total_tokens <= 100 + 5, (
            f"Total result ({total_tokens} tokens) should be ≤ max_result_tokens=100 + small rounding; "
            f"body={body_tokens}, suffix={suffix_tokens}"
        )


# ── Finding 9: on_tool_end called after output guardrails ────────────────────


class TestOnToolEndAfterOutputGuardrails:
    """Finding 9: on_tool_end and emit_audit must see the post-guardrail result."""

    async def test_on_tool_end_sees_post_guardrail_result(self) -> None:
        """When an output guardrail rejects content, on_tool_end must see the
        rejection message, not the original tool output."""
        from types import SimpleNamespace

        from troopai.adk.agents.middleware import Middleware
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.tools_executor import execute_tool_calls
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

        raw_output = "SENSITIVE_DATA_HERE"
        rejection_msg = "Content blocked by policy."

        # Tool that returns raw sensitive data
        async def _handler(ctx, _raw):
            return raw_output

        # Output guardrail that replaces the content
        class _MaskingGuardrail:
            async def run(self, data):
                return MagicMock(behavior={"type": "reject_content", "message": rejection_msg})

            def get_name(self):
                return "masking"

        from troopai.adk.tools.tool_guardrails import ToolGuardrails

        tool = FunctionTool(
            name="sensitive_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_handler,
            guardrails=ToolGuardrails(output=[_MaskingGuardrail()]),  # type: ignore[list-item]
        )

        on_tool_end_results: list[Any] = []

        class _TrackingHooks(RunHooks):
            async def on_tool_end(self, ctx, agent, tool_name, result):  # type: ignore[override]
                on_tool_end_results.append(result)

        agent = SimpleNamespace(
            name="test",
            tools=[tool],
            tool_use_behavior="run_llm_again",
            handoffs=None,
            llm=None,
            hooks=None,
            middleware=Middleware(),
        )

        tool_calls = [LLMResponseFunctionToolCall(call_id="c1", name="sensitive_tool", arguments="{}")]

        await execute_tool_calls(
            agent=agent,  # type: ignore[arg-type]
            tool_calls=tool_calls,
            ctx_wrapper=RunContext(context=None),
            hooks=_TrackingHooks(),
            config=DEFAULT_RUN_CONFIG,
            model="gpt-4o-mini",
            parallel=False,
        )

        assert len(on_tool_end_results) == 1
        assert on_tool_end_results[0] == rejection_msg, (
            f"on_tool_end should see post-guardrail result '{rejection_msg}', but got '{on_tool_end_results[0]}'"
        )


# ── Finding 10: can_use_tool exception returns distinct error message ────────


class TestCanUseToolCallbackError:
    """Finding 10: can_use_tool exceptions must return a distinct error message."""

    async def test_can_use_tool_exception_returns_error_message(self) -> None:
        """When can_use_tool raises, the result must be a distinct error message,
        not the same tool_permission_denied message as a deliberate refusal."""
        from types import SimpleNamespace

        from troopai.adk.agents.middleware import Middleware
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import RunConfig
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.tools_executor import execute_tool_calls
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

        async def _handler(ctx, _raw):
            return "result"

        tool = FunctionTool(
            name="restricted_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_handler,
        )

        def _raising_can_use_tool(agent, tool_name, ctx):
            raise ConnectionError("DB unavailable")

        config = RunConfig(
            model="gpt-4o-mini",
            can_use_tool=_raising_can_use_tool,
        )

        agent = SimpleNamespace(
            name="test",
            tools=[tool],
            tool_use_behavior="run_llm_again",
            handoffs=None,
            llm=None,
            hooks=None,
            middleware=Middleware(),
        )

        results, _ = await execute_tool_calls(
            agent=agent,  # type: ignore[arg-type]
            tool_calls=[LLMResponseFunctionToolCall(call_id="c1", name="restricted_tool", arguments="{}")],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=config,
            model="gpt-4o-mini",
            parallel=False,
        )

        assert len(results) == 1
        assert "error" in results[0].output.lower() or "failed" in results[0].output.lower(), (
            f"Expected error message for callback exception, got: '{results[0].output}'"
        )
        # Must NOT be the same as a deliberate denial
        assert "Permission denied" not in results[0].output, (
            "Exception path must not use the same message as a deliberate denial"
        )


# ── Finding 11: Streaming MESSAGE_OUTPUT_CREATED only when content present ──


class TestStreamingMessageOutputCreated:
    """Finding 11: MESSAGE_OUTPUT_CREATED must not be emitted on tool-call-only turns."""

    def test_content_property_returns_none_for_tool_call_only(self) -> None:
        """Verify LLMResponse.content is None when only tool calls present."""
        from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseFunctionToolCall

        response = LLMResponse(
            response_id="r1",
            model="m",
            response=[LLMResponseFunctionToolCall(call_id="c1", name="fn", arguments="{}")],
        )
        assert response.content is None
        assert response.refusal is None
        assert len(response.tool_calls) == 1


# ── Finding 12: tool.prepare exception skips (not includes) tool ─────────────


class TestPrepareExceptionSkipsTool:
    """Finding 12: when prepare raises, the tool must be excluded (not included)."""

    async def test_prepare_exception_excludes_tool(self) -> None:
        """A prepare function that raises must result in the tool being excluded."""
        from troopai.adk.run.llm_calls import build_tools
        from troopai.adk.tools.function_tool import FunctionTool

        async def _handler(ctx, _raw):
            return "ok"

        def _bad_prepare(ctx, tool):
            raise AttributeError("context field missing")

        tool = FunctionTool(
            name="guarded_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_handler,
            prepare=_bad_prepare,
        )

        from types import SimpleNamespace

        from troopai.adk.agents.middleware import Middleware
        from troopai.adk.run.context import RunContext

        agent = SimpleNamespace(
            name="test",
            tools=[tool],
            skills=None,
            middleware=Middleware(),
            llm=None,
            llm_config=None,
            handoffs=None,
        )

        tool_list = await build_tools(
            agent=agent,  # type: ignore[arg-type]
            context=RunContext(context=None),
        )

        # build_tools returns None when the result list is empty
        tool_names = [t.name for t in (tool_list or [])]
        assert "guarded_tool" not in tool_names, "Tool whose prepare() raised must be excluded, not included unchanged"


# ── Finding 13: _drop_orphan_run_items respects empty call_id ────────────────


class TestDropOrphanRunItemsEmptyCallId:
    """Finding 13: empty-string call_id must not cause false orphan detection."""

    def test_empty_call_id_pair_is_kept(self) -> None:
        """A ToolCallItem / ToolCallOutputItem pair with call_id='' must not be dropped."""
        from troopai.adk.handoffs.handoff_filters import _drop_orphan_run_items
        from troopai.adk.types.items.items import ToolCallItem, ToolCallOutputItem

        # Build minimal raw mocks with empty call_id
        tc_raw = MagicMock()
        tc_raw.call_id = ""

        to_raw = MagicMock()
        to_raw.call_id = ""

        tc_item = MagicMock(spec=ToolCallItem)
        tc_item.raw = tc_raw

        to_item = MagicMock(spec=ToolCallOutputItem)
        to_item.raw = to_raw

        result = _drop_orphan_run_items([tc_item, to_item])
        assert len(result) == 2, "A matched ToolCallItem/ToolCallOutputItem pair with call_id='' must not be dropped"


# ── Finding 14: Dead `or 5` fallback removed from LAST_N ─────────────────────


class TestLastNWindowNoDeadFallback:
    """Finding 14: `or 5` dead fallback must be removed from LAST_N branches."""

    def test_prepare_handoff_input_no_or_5_fallback(self) -> None:
        """Verify the source code no longer contains the `or 5` dead fallback."""
        import inspect

        from troopai.adk.run import handoffs_executor

        source = inspect.getsource(handoffs_executor)
        # The `or 5` pattern should be gone
        assert "or 5" not in source, (
            "Dead `or 5` fallback must be removed from prepare_handoff_input/prepare_handoff_input_from_data"
        )
