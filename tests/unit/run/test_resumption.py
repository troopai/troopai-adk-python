"""Regression tests for HITL state resumption (resumption.py).

Covers four confirmed defects in the resume paths:

1. tenant_id dropped on resume — bare ``RunContext(context=...)`` left
   ``tenant_id`` ``None``, silently disabling the per-tenant tool
   allowlist after human approval.
2. nested agent-tool resume ignored the caller-supplied ``context``
   override, using the stale serialized ``state.context``.
3. streamed resumption dropped pre-deferral ``new_items`` when a nested
   agent-tool re-deferred mid-loop.
4. nested agent-tool resume double-wrapped verbose hooks.
"""

from types import SimpleNamespace
from unittest.mock import patch

from troopai.adk.run.config import RunConfig
from troopai.adk.run.resumption import resume_from_state, resume_from_state_streamed
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.items import ItemHelpers

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools, name="test_agent"):
    from troopai.adk.agents.agent_guardrails import AgentGuardrails
    from troopai.adk.agents.middleware import Middleware
    from troopai.adk.skills.activation import SkillActivation

    return SimpleNamespace(
        name=name,
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        system_prompt=None,
        llm_config=None,
        output_type=None,
        guardrails=AgentGuardrails(),
        hooks=None,
        middleware=Middleware(),
        verbose=None,
        skills=[],
        skill_activation=SkillActivation.LAZY,
    )


async def _echo_handler(_ctx, _raw_args):
    return "executed"


def _make_deferred_tool(tool_name="admin_delete", call_id="tc_1"):
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name=tool_name,
        tool_arguments={"id": "123"},
        raw_arguments='{"id": "123"}',
    )


def _make_state(approved_tools, context=None):
    """Create a minimal RunState with approved tools."""
    from troopai.adk.run.state import RunState

    raw_messages = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": approved.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": approved.tool_name,
                        "arguments": approved.raw_arguments,
                    },
                }
                for approved in approved_tools
            ],
        },
    ]

    state = RunState(
        conversation_history=list(ItemHelpers.messages_to_run_items(raw_messages)),
        context=context,
        # Copy so approve()'s in-place removal doesn't mutate the caller's list.
        deferred_tool_requests=DeferredToolRequests(approvals=list(approved_tools)),
        original_user_prompt="test",
        current_agent_name="test_agent",
        turn_count=1,
    )
    for tool in list(approved_tools):
        state.approve(tool)
    return state


def _stub_loop_result(agent):
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.run import RunResult

    return RunResult(
        final_output="done",
        user_prompt="test",
        new_items=[],
        context=RunContext(context=None),
        last_agent=agent,
    )


# ── Finding 1: tenant_id must survive HITL resume ────────────────────


class TestTenantIdPreservedOnResume:
    async def test_resumed_ctx_wrapper_carries_tenant_id(self) -> None:
        """The resumed loop must receive a ctx_wrapper bearing config.tenant_id."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])
        state = _make_state([_make_deferred_tool("admin_delete", "tc_1")])
        # allowlist permits this tool for the tenant so execution proceeds
        # and the resumed loop is reached.
        config = RunConfig(
            tenant_id="acme",
            tenant_tool_allowlist={"acme": {"admin_delete"}},
            fail_on_tool_error=False,
        )

        with patch("troopai.adk.run.loop.run_agent_loop") as mock_loop:
            mock_loop.return_value = _stub_loop_result(agent)
            _ = await resume_from_state(agent=agent, state=state, config=config)

        ctx_wrapper = mock_loop.call_args.kwargs["ctx_wrapper"]
        assert ctx_wrapper.tenant_id == "acme"
        loop_context = mock_loop.call_args.kwargs["context"]
        assert loop_context.tenant_id == "acme"

    async def test_allowlist_still_denies_disallowed_tool_on_resume(self) -> None:
        """A tenant allowlist that excludes the tool must deny it post-approval."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])
        state = _make_state([_make_deferred_tool("admin_delete", "tc_1")])
        # Tenant is allowed only "safe_tool"; admin_delete must be denied
        # even though a human approved it.
        config = RunConfig(
            tenant_id="acme",
            tenant_tool_allowlist={"acme": {"safe_tool"}},
            tenant_allowlist_soft_deny=True,
            fail_on_tool_error=False,
        )

        with patch("troopai.adk.run.loop.run_agent_loop") as mock_loop:
            mock_loop.return_value = _stub_loop_result(agent)
            _ = await resume_from_state(agent=agent, state=state, config=config)

        messages = mock_loop.call_args.kwargs.get("initial_messages", [])
        tool_msgs = [m for m in messages if m.get("type") == "function_call_output"]
        outputs = [str(m.get("output", "")) for m in tool_msgs]
        # Denied — the tool body ("executed") must NOT have run.
        assert all("executed" not in o for o in outputs)
        assert any("Permission denied" in o for o in outputs)

    def test_streamed_run_context_carries_tenant_id(self) -> None:
        """The streamed resume result's context must bear config.tenant_id."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])
        state = _make_state([_make_deferred_tool("admin_delete", "tc_1")])
        config = RunConfig(tenant_id="acme")

        result = resume_from_state_streamed(agent=agent, state=state, config=config)
        assert result.context.tenant_id == "acme"


# ── Finding 2: caller-supplied context override reaches nested resume ──


class TestNestedResumeUsesEffectiveContext:
    async def test_sync_nested_resume_passes_caller_context(self) -> None:
        """resume_nested_agent_tool must receive the caller-supplied context."""
        from troopai.adk.tools.deferred_tool import DeferredToolCallMetadata

        nested = DeferredToolCall(
            tool_call_id="tc_nested",
            tool_name="sub_agent_tool",
            tool_arguments={},
            raw_arguments="{}",
            metadata=DeferredToolCallMetadata(nested_agent=True, nested_agent_name="sub"),
        )
        state = _make_state([nested], context="stale-serialized")
        agent = _make_agent([])
        live_context = "live-shared-context"
        config = RunConfig(fail_on_tool_error=False)

        with (
            patch("troopai.adk.run.resumption.resume_nested_agent_tool") as mock_nested,
            patch("troopai.adk.run.loop.run_agent_loop") as mock_loop,
        ):
            mock_nested.return_value = "sub-result"
            mock_loop.return_value = _stub_loop_result(agent)
            _ = await resume_from_state(agent=agent, state=state, config=config, context=live_context)

        assert mock_nested.call_args.kwargs["context"] == live_context


# ── Finding 4: verbose hooks must not be double-wrapped on nested resume ──


class TestNestedResumePassesUnwrappedHooks:
    async def test_sync_nested_resume_passes_pre_verbose_hooks(self) -> None:
        """Nested resume must get the user (pre-verbose) hooks, not the
        already-VerboseHooks-wrapped chain, so the inner resume wraps once."""
        from troopai.adk.tools.deferred_tool import DeferredToolCallMetadata
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.hooks import VerboseHooks

        nested = DeferredToolCall(
            tool_call_id="tc_nested",
            tool_name="sub_agent_tool",
            tool_arguments={},
            raw_arguments="{}",
            metadata=DeferredToolCallMetadata(nested_agent=True, nested_agent_name="sub"),
        )
        state = _make_state([nested])
        agent = _make_agent([])
        config = RunConfig(verbose=VerboseConfig(enabled=True), fail_on_tool_error=False)

        with (
            patch("troopai.adk.run.resumption.resume_nested_agent_tool") as mock_nested,
            patch("troopai.adk.run.loop.run_agent_loop") as mock_loop,
        ):
            mock_nested.return_value = "sub-result"
            mock_loop.return_value = _stub_loop_result(agent)
            _ = await resume_from_state(agent=agent, state=state, config=config)

        forwarded_hooks = mock_nested.call_args.kwargs["hooks"]
        # The forwarded hooks (None here, since the caller passed no hooks)
        # must NOT be a VerboseHooks instance — verbose wrapping happens
        # exactly once, inside the nested resume's own Runner.arun.
        assert not isinstance(forwarded_hooks, VerboseHooks)


# ── Finding 3: streamed resume keeps pre-deferral items on re-deferral ──


class TestStreamedResumeKeepsItemsOnReDeferral:
    async def test_pre_deferral_items_survive_nested_redeferral(self) -> None:
        """A first approved tool's output item must remain on
        ``result.new_items`` even when a later nested agent-tool re-defers."""
        from troopai.adk.exceptions import AgentToolDeferral
        from troopai.adk.tools.deferred_tool import DeferredToolCallMetadata

        first_tool = FunctionTool(
            name="safe_tool",
            description="Safe",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([first_tool])

        first_call = _make_deferred_tool("safe_tool", "tc_first")
        nested_call = DeferredToolCall(
            tool_call_id="tc_nested",
            tool_name="sub_agent_tool",
            tool_arguments={},
            raw_arguments="{}",
            metadata=DeferredToolCallMetadata(nested_agent=True, nested_agent_name="sub"),
        )
        state = _make_state([first_call, nested_call])
        config = RunConfig(fail_on_tool_error=False)

        # The nested sub-agent re-defers during resumption.
        async def _redefer(**_kwargs):
            raise AgentToolDeferral(
                agent_name="sub",
                deferred_requests=DeferredToolRequests(approvals=[_make_deferred_tool("inner", "tc_inner")]),
                state=_make_state([_make_deferred_tool("inner", "tc_inner")]),
            )

        with patch("troopai.adk.run.resumption.resume_nested_agent_tool", side_effect=_redefer):
            result = resume_from_state_streamed(agent=agent, state=state, config=config)
            async for _event in result.stream_events():
                pass

        # The run re-deferred (requires action) but the first tool's output
        # item must still be present on the result.
        assert result.deferred_requests is not None
        call_ids = [
            item.raw.call_id for item in result.new_items if hasattr(item, "raw") and hasattr(item.raw, "call_id")
        ]
        assert "tc_first" in call_ids


def _new_item_call_ids(new_items) -> list[str]:
    return [item.raw.call_id for item in new_items if hasattr(item, "raw") and hasattr(item.raw, "call_id")]


async def _redefer_deferral(**_kwargs):
    from troopai.adk.exceptions import AgentToolDeferral

    raise AgentToolDeferral(
        agent_name="sub",
        deferred_requests=DeferredToolRequests(approvals=[_make_deferred_tool("inner", "tc_inner")]),
        state=_make_state([_make_deferred_tool("inner", "tc_inner")]),
    )


def _make_nested_call(call_id: str = "tc_nested_orig") -> DeferredToolCall:
    from troopai.adk.tools.deferred_tool import DeferredToolCallMetadata

    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name="sub_agent_tool",
        tool_arguments={},
        raw_arguments="{}",
        metadata=DeferredToolCallMetadata(nested_agent=True, nested_agent_name="sub"),
    )


# ── Finding: sync resume must thread ONE RunContext (usage/hooks parity) ──


class TestSyncResumeSingleRunContext:
    async def test_loop_context_is_the_ctx_wrapper(self) -> None:
        """run_agent_loop must receive the SAME object for ``context`` and
        ``ctx_wrapper``; a second, fresh RunContext would leave the
        hooks/tools ctx_wrapper with zero usage."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])
        state = _make_state([_make_deferred_tool("admin_delete", "tc_1")])
        config = RunConfig(fail_on_tool_error=False)

        with patch("troopai.adk.run.loop.run_agent_loop") as mock_loop:
            mock_loop.return_value = _stub_loop_result(agent)
            _ = await resume_from_state(agent=agent, state=state, config=config)

        assert mock_loop.call_args.kwargs["context"] is mock_loop.call_args.kwargs["ctx_wrapper"]


# ── Finding: nested re-deferral keeps original call_id + processes remaining ──


class TestSyncNestedReDeferral:
    async def test_redeferral_reuses_original_call_id(self) -> None:
        """A nested agent-tool that re-defers must surface a new deferred
        request under its ORIGINAL call_id, not a synthetic ``nested_<id>``
        one (which would orphan the parent tool_use on the next resume)."""
        nested = _make_nested_call("tc_nested_orig")
        state = _make_state([nested])
        agent = _make_agent([])
        config = RunConfig(fail_on_tool_error=False)

        with (
            patch("troopai.adk.run.resumption.resume_nested_agent_tool", side_effect=_redefer_deferral),
            patch("troopai.adk.run.loop.run_agent_loop") as mock_loop,
        ):
            result = await resume_from_state(agent=agent, state=state, config=config)

        # The loop is not resumed while a tool_use is still pending.
        mock_loop.assert_not_called()
        assert result.deferred_requests is not None
        redeferred = result.deferred_requests.approvals
        assert [c.tool_call_id for c in redeferred] == ["tc_nested_orig"]
        assert not redeferred[0].tool_call_id.startswith("nested_")

    async def test_remaining_approved_tool_still_executes(self) -> None:
        """When a nested tool re-defers, a LATER approved tool must still run
        (the old code aborted the loop and dropped its decision)."""
        safe_tool = FunctionTool(
            name="safe_tool",
            description="Safe",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([safe_tool])

        nested_first = _make_nested_call("tc_nested_orig")
        safe_second = _make_deferred_tool("safe_tool", "tc_second")
        state = _make_state([nested_first, safe_second])
        config = RunConfig(fail_on_tool_error=False)

        with (
            patch("troopai.adk.run.resumption.resume_nested_agent_tool", side_effect=_redefer_deferral),
            patch("troopai.adk.run.loop.run_agent_loop") as mock_loop,
        ):
            result = await resume_from_state(agent=agent, state=state, config=config)

        mock_loop.assert_not_called()
        # The nested tool re-deferred under its original id...
        assert result.deferred_requests is not None
        assert [c.tool_call_id for c in result.deferred_requests.approvals] == ["tc_nested_orig"]
        # ...and the later safe tool's result is still present (not dropped).
        assert "tc_second" in _new_item_call_ids(result.new_items)


class TestStreamedNestedReDeferralFirst:
    async def test_remaining_tool_runs_when_nested_defers_first(self) -> None:
        """Streamed: a nested tool that re-defers FIRST must not abort the loop;
        a later approved tool must still execute, and the re-deferral must keep
        the original call_id."""
        safe_tool = FunctionTool(
            name="safe_tool",
            description="Safe",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([safe_tool])

        nested_first = _make_nested_call("tc_nested_orig")
        safe_second = _make_deferred_tool("safe_tool", "tc_second")
        state = _make_state([nested_first, safe_second])
        config = RunConfig(fail_on_tool_error=False)

        with patch("troopai.adk.run.resumption.resume_nested_agent_tool", side_effect=_redefer_deferral):
            result = resume_from_state_streamed(agent=agent, state=state, config=config)
            async for _event in result.stream_events():
                pass

        assert result.deferred_requests is not None
        assert [c.tool_call_id for c in result.deferred_requests.approvals] == ["tc_nested_orig"]
        assert "tc_second" in _new_item_call_ids(result.new_items)
