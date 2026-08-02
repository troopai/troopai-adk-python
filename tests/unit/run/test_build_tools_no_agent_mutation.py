"""Tests pinning the H2 invariant: ``build_tools`` MUST NOT mutate ``agent.tools``.

``Agent`` is a configuration object that may be shared across concurrent
``Runner.arun()`` calls.  A previous implementation appended wired
``FunctionTool`` wrappers (for ``ShellTool`` and ``ApplyPatchTool``) back
onto ``agent.tools`` during ``build_tools``.  That was a race condition:
two concurrent runs would observe each other's wrapper tools.

The fix emits the wrapper locally and resolves it on demand via
``resolve_function_tool``.  These tests verify:

1. ``build_tools`` returns a wired ``FunctionTool`` without appending it
   to ``agent.tools``.
2. ``resolve_function_tool`` can reconstruct an equivalent wrapper on
   demand, without agent mutation.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from troopai.adk.run.llm_calls import build_tools, resolve_function_tool
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.local.apply_patch_tool import ApplyPatchEditor, ApplyPatchTool
from troopai.adk.tools.local.shell_tool import ShellExecutor, ShellTool


class _NoopShellExecutor(ShellExecutor):
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> str:
        return "ok"


class _NoopEditor(ApplyPatchEditor):
    async def apply(self, patch: str) -> str:
        return "ok"


def _make_agent(tools: list) -> Any:
    """Minimal agent-like stub accepted by ``build_tools`` / ``resolve_function_tool``."""
    return SimpleNamespace(
        name="test_agent",
        tools=tools,
        llm=None,
        llm_config=None,
        skills=[],
        handoffs=None,
        output_schema=None,
        system_prompt="test",
    )


class TestBuildToolsDoesNotMutateAgent:
    @pytest.mark.asyncio
    async def test_shell_tool_does_not_mutate_agent_tools(self) -> None:
        shell = ShellTool(executor=_NoopShellExecutor())
        agent = _make_agent([shell])
        before_ids = [id(t) for t in agent.tools]
        before_len = len(agent.tools)

        result = await build_tools(agent)

        assert result is not None
        assert any(isinstance(t, FunctionTool) and t.name == "shell" for t in result)
        assert len(agent.tools) == before_len
        assert [id(t) for t in agent.tools] == before_ids
        # Wrapper MUST NOT be interned onto agent.tools.
        assert all(not isinstance(t, FunctionTool) for t in agent.tools)

    @pytest.mark.asyncio
    async def test_apply_patch_tool_does_not_mutate_agent_tools(self) -> None:
        patcher = ApplyPatchTool(editor=_NoopEditor())
        agent = _make_agent([patcher])
        before_len = len(agent.tools)

        result = await build_tools(agent)

        assert result is not None
        assert any(isinstance(t, FunctionTool) and t.name == "apply_patch" for t in result)
        assert len(agent.tools) == before_len
        assert all(not isinstance(t, FunctionTool) for t in agent.tools)

    @pytest.mark.asyncio
    async def test_repeated_build_tools_calls_do_not_grow_agent_tools(self) -> None:
        """Concurrent/sequential builds on a shared agent must not accumulate state."""
        shell = ShellTool(executor=_NoopShellExecutor())
        patcher = ApplyPatchTool(editor=_NoopEditor())
        agent = _make_agent([shell, patcher])
        original_len = len(agent.tools)

        for _ in range(5):
            await build_tools(agent)

        assert len(agent.tools) == original_len


class TestResolveFunctionTool:
    async def test_resolves_wired_shell_tool_on_demand(self) -> None:
        shell = ShellTool(executor=_NoopShellExecutor())
        agent = _make_agent([shell])

        resolved = await resolve_function_tool(agent, "shell")

        assert resolved is not None
        assert isinstance(resolved, FunctionTool)
        assert resolved.name == "shell"

    async def test_resolves_wired_apply_patch_tool_on_demand(self) -> None:
        patcher = ApplyPatchTool(editor=_NoopEditor())
        agent = _make_agent([patcher])

        resolved = await resolve_function_tool(agent, "apply_patch")

        assert resolved is not None
        assert isinstance(resolved, FunctionTool)
        assert resolved.name == "apply_patch"

    async def test_returns_none_for_shell_tool_without_executor(self) -> None:
        shell = ShellTool(executor=None)
        agent = _make_agent([shell])

        assert await resolve_function_tool(agent, "shell") is None

    async def test_returns_none_for_apply_patch_without_editor(self) -> None:
        patcher = ApplyPatchTool(editor=None)
        agent = _make_agent([patcher])

        assert await resolve_function_tool(agent, "apply_patch") is None

    async def test_returns_none_for_unknown_tool(self) -> None:
        agent = _make_agent([])
        assert await resolve_function_tool(agent, "nonexistent") is None
