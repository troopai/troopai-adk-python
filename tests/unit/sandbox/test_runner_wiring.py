"""Tests for the Runner sandbox-bracket wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.config import SandboxRunConfig


def _make_session() -> Any:
    s = MagicMock()
    s.start = AsyncMock()
    s.stop = AsyncMock()
    s.shutdown = AsyncMock()
    s.aclose = AsyncMock()
    # The run lifecycle now calls apply_manifest after start() when a
    # manifest is configured; the double must model it as awaitable.
    s.apply_manifest = AsyncMock()
    return s


def _make_client(session: Any) -> Any:
    c = MagicMock()
    c.create = AsyncMock(return_value=session)
    return c


class TestSandboxBracketDetection:
    """The bracket should open ONLY when an agent / config triggers it."""

    @pytest.mark.asyncio
    async def test_no_sandbox_path_preserves_no_op(self) -> None:
        """Plain Agent + no run_config.sandbox: bracket is no-op."""
        # When neither trigger applies, _maybe_open_sandbox_bracket
        # returns without touching the stack.
        import contextlib

        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import _maybe_open_sandbox_bracket

        plain_agent = Agent(name="plain", system_prompt="hi")
        config = RunConfig()
        run_context: RunContext[None] = RunContext.make(None)
        async with contextlib.AsyncExitStack() as stack:
            await _maybe_open_sandbox_bracket(
                stack=stack,
                agent=plain_agent,
                config=config,
                run_context=run_context,
            )
            # No handle was published.
            assert not hasattr(run_context, "_sandbox_handle") or (
                getattr(run_context, "_sandbox_handle", None) is None
            )

    @pytest.mark.asyncio
    async def test_sandbox_agent_opens_bracket(self) -> None:
        """SandboxAgent triggers the bracket; session.start() runs."""
        import contextlib

        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import _maybe_open_sandbox_bracket

        session = _make_session()
        client = _make_client(session)
        agent = SandboxAgent(name="sandboxed", system_prompt="hi")
        config = RunConfig(sandbox=SandboxRunConfig(client=client))
        run_context: RunContext[None] = RunContext.make(None)
        async with contextlib.AsyncExitStack() as stack:
            await _maybe_open_sandbox_bracket(
                stack=stack,
                agent=agent,
                config=config,
                run_context=run_context,
            )
            # Session.start called.
            session.start.assert_called_once()
        # On stack exit: aclose called.
        session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_runconfig_sandbox_only_opens_bracket(self) -> None:
        """Plain Agent + run_config.sandbox: bracket opens."""
        import contextlib

        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import _maybe_open_sandbox_bracket

        session = _make_session()
        client = _make_client(session)
        agent = Agent(name="plain", system_prompt="hi")
        config = RunConfig(sandbox=SandboxRunConfig(client=client))
        run_context: RunContext[None] = RunContext.make(None)
        async with contextlib.AsyncExitStack() as stack:
            await _maybe_open_sandbox_bracket(
                stack=stack,
                agent=agent,
                config=config,
                run_context=run_context,
            )
            session.start.assert_called_once()
        session.aclose.assert_called_once()


class TestSandboxAgentDefaultManifestMerge:
    """When SandboxAgent.default_manifest is set, it merges into the config."""

    @pytest.mark.asyncio
    async def test_default_manifest_falls_through(self) -> None:
        import contextlib

        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import _maybe_open_sandbox_bracket
        from troopai.adk.types.sandbox.manifest import Manifest

        session = _make_session()
        client = _make_client(session)
        manifest = Manifest(root="/workspace")
        agent = SandboxAgent(
            name="sandboxed",
            system_prompt="hi",
            default_manifest=manifest,
        )
        # Config has NO manifest — agent's default should win.
        config = RunConfig(sandbox=SandboxRunConfig(client=client))
        run_context: RunContext[None] = RunContext.make(None)
        async with contextlib.AsyncExitStack() as stack:
            await _maybe_open_sandbox_bracket(
                stack=stack,
                agent=agent,
                config=config,
                run_context=run_context,
            )
            # The client.create was called with manifest=<agent's default>
            client.create.assert_called_once()
            assert client.create.call_args.kwargs["manifest"] is manifest


class TestCapabilityToolMerging:
    """Capability tools are merged into the agent's tool list."""

    @pytest.mark.asyncio
    async def test_capability_tools_appear_on_cloned_agent(self) -> None:
        """After the sandbox bracket opens, the agent clone has cap tools."""
        import contextlib

        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import (
            _maybe_clone_agent_with_capability_tools,
            _maybe_open_sandbox_bracket,
        )
        from troopai.adk.sandbox.agent import SandboxAgent
        from troopai.adk.sandbox.capabilities.shell import ShellCapability

        session = _make_session()
        client = _make_client(session)
        agent = SandboxAgent(
            name="coder",
            system_prompt="hi",
            capabilities=[ShellCapability()],
        )
        config = RunConfig(sandbox=SandboxRunConfig(client=client))
        run_context: RunContext[None] = RunContext.make(None)
        async with contextlib.AsyncExitStack() as stack:
            await _maybe_open_sandbox_bracket(
                stack=stack,
                agent=agent,
                config=config,
                run_context=run_context,
            )
            cloned = _maybe_clone_agent_with_capability_tools(
                agent=agent,
                run_context=run_context,
            )
            # The clone has the ShellCapability's tool (run_command).
            tool_names = {t.name for t in cloned.tools}
            assert "run_command" in tool_names
            # The original agent is unchanged.
            assert all(getattr(t, "name", None) != "run_command" for t in agent.tools)

    @pytest.mark.asyncio
    async def test_no_sandbox_returns_agent_unchanged(self) -> None:
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.runner import _maybe_clone_agent_with_capability_tools

        agent = Agent(name="plain", system_prompt="hi")
        run_context: RunContext[None] = RunContext.make(None)
        result = _maybe_clone_agent_with_capability_tools(
            agent=agent,
            run_context=run_context,
        )
        assert result is agent  # no clone needed


class TestSandboxUsageOnRunResult:
    """Runner.arun attaches SandboxUsage onto RunResult.sandbox_usage."""

    @pytest.mark.asyncio
    async def test_sandbox_usage_attached_after_arun(self) -> None:
        """After a sandboxed arun, result.sandbox_usage is non-None.

        The model is mocked to return a plain text response (no tool
        calls), so exec_count stays 0 — but the SandboxUsage object
        must be attached because a sandbox session ran. Confirming
        exec_count==1 from a real run_command call would require the
        mock LLM to emit a function-call turn that drives the tool
        executor; that integration is covered in run_command
        observability tests. This test focuses on the Runner wiring:
        does the usage object reach RunResult?
        """
        from unittest.mock import patch

        from troopai.adk.run.runner import Runner
        from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

        session = _make_session()
        client = _make_client(session)
        agent = SandboxAgent(name="sandboxed", system_prompt="hi")
        config = RunConfig(sandbox=SandboxRunConfig(client=client))

        async def _fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(
                response_id="r1",
                model="fake",
                response=[LLMResponseText(text="done")],
            )

        with (
            patch(
                "troopai.adk.run.loop.call_llm",
                new=AsyncMock(side_effect=_fake_call_llm),
            ),
            patch(
                "troopai.adk.run.runner.run_blocking_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "troopai.adk.run.runner.run_parallel_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "troopai.adk.run.runner.run_output_guardrails",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await Runner.arun(agent, "go", run_config=config)

        assert result.sandbox_usage is not None
