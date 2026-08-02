"""Tests for ``ShellCapability`` + ``RunCommandTool`` (P13 + P27)."""

from __future__ import annotations

import json

import pytest

from troopai.adk.exceptions.exceptions import SandboxCommandRejected
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.local import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.sandbox.guardrails.command_guardrail import SandboxCommandGuardrail
from troopai.adk.sandbox.tools.run_command_tool import (
    RunCommandArgs,
    make_run_command_tool,
)


@pytest.fixture
async def local_session() -> object:
    client = LocalSubprocessSandboxClient(warn_banner=False)
    session = await client.create(options=LocalSandboxClientOptions())
    await session.start()
    yield session
    await session.aclose()


class TestRunCommandArgs:
    def test_minimal(self) -> None:
        args = RunCommandArgs(command="ls")
        assert args.command == "ls"
        assert args.timeout is None
        assert args.shell is True

    def test_full(self) -> None:
        args = RunCommandArgs(command="ls", timeout=5.0, shell=False)
        assert args.timeout == 5.0
        assert args.shell is False


class TestMakeRunCommandTool:
    @pytest.mark.asyncio
    async def test_invoke_returns_dict(self, local_session: object) -> None:
        tool = make_run_command_tool(session=local_session)  # type: ignore[arg-type]
        # The FunctionTool's on_invoke takes (ctx, raw_args_json).
        assert tool.on_invoke is not None
        raw = json.dumps({"command": "echo hello", "shell": True})
        result = await tool.on_invoke(None, raw)  # type: ignore[arg-type]
        assert "hello" in result["stdout"]
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_command_policy_blocks(self, local_session: object) -> None:
        policy = SandboxCommandGuardrail(allowlist=["ls"])
        tool = make_run_command_tool(
            session=local_session,  # type: ignore[arg-type]
            command_policy=policy,
        )
        assert tool.on_invoke is not None
        raw = json.dumps({"command": "rm -rf /"})
        with pytest.raises(SandboxCommandRejected):
            await tool.on_invoke(None, raw)  # type: ignore[arg-type]


class TestShellCapability:
    def test_unbound_returns_no_tools(self) -> None:
        cap = ShellCapability()
        assert cap.tools() == []

    @pytest.mark.asyncio
    async def test_bound_returns_run_command_tool(
        self,
        local_session: object,
    ) -> None:
        cap = ShellCapability()
        cap.bind(local_session)
        tools = cap.tools()
        assert len(tools) == 1
        assert tools[0].name == "run_command"

    @pytest.mark.asyncio
    async def test_instructions_returns_primer(self) -> None:
        cap = ShellCapability()
        from troopai.adk.types.sandbox.manifest import Manifest

        primer = await cap.instructions(Manifest())
        assert primer is not None
        assert "run_command" in primer
