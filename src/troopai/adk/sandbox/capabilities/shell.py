"""``ShellCapability`` — exposes a sandbox shell command tool.

When bound to a live session, the capability surfaces a single
``run_command`` FunctionTool to the agent loop + injects a short
usage primer into the system prompt.

Optionally accepts a ``SandboxCommandGuardrail`` that is
plumbed through the FunctionTool's invocation path so policy
verdicts surface as ``SandboxCommandRejected`` (and propagate via
``RunResult.guardrail_results`` upstream).
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, override

from pydantic import Field

from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.sandbox.tools.run_command_tool import make_run_command_tool

if TYPE_CHECKING:
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.sandbox.manifest import Manifest

__all__ = ["ShellCapability"]


_SHELL_INSTRUCTIONS = dedent(
    """
    When you need to run a shell command, use the ``run_command``
    tool. The tool returns stdout, stderr, exit code, and duration.
    Non-zero exit codes are reported in the result (not raised) so
    you can decide whether the operation succeeded.

    Tips:
    - Prefer ``rg`` and ``rg --files`` for text/file discovery when available.
    - Use ``run_command`` with ``shell=True`` (default) for pipelines / redirection.
    - Always check the exit code before treating output as success.
    """,
).strip()


class ShellCapability(SandboxCapability):
    """Capability that exposes the run_command FunctionTool.

    Attributes:
        command_policy: Optional ``SandboxCommandGuardrail``
            evaluated before each run_command invocation.
        tool_name: Override the FunctionTool's name (default
            ``"run_command"``).
        tool_description: Override the FunctionTool's description
            (default: built-in primer).
    """

    type: Literal["shell"] = "shell"
    """Discriminator."""

    command_policy: Any = Field(default=None, exclude=True)
    """Optional ``SandboxCommandGuardrail``; excluded from serialization."""

    tool_name: str = "run_command"
    """FunctionTool name surfaced to the LLM."""

    tool_description: str | None = None
    """Optional FunctionTool description override."""

    @override
    def tools(self) -> list[FunctionTool]:
        if self.session is None:
            return []
        return [
            make_run_command_tool(
                session=self.session,
                user=self.run_as,
                name=self.tool_name,
                description=self.tool_description,
                command_policy=self.command_policy,
                observability=self.observability,
            ),
        ]

    @override
    async def instructions(self, manifest: Manifest | None) -> str | None:
        _ = manifest
        return _SHELL_INSTRUCTIONS
