"""``RunCommandTool`` — the model-facing shell-command primitive.

The Shell capability wires this into the agent's tool list when a
sandbox session is bound. The tool forwards parsed args to
``session.run(...)`` and, when a ``SandboxObservability`` handle is
bound, emits per-command usage, a span, audit events, and lifecycle
hooks around the call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from troopai.adk.exceptions.exceptions import SandboxCommandRejected
from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.sandbox.guardrails.command_guardrail import SandboxCommandGuardrail
    from troopai.adk.sandbox.observability.observability import SandboxObservability
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.sandbox.permissions import User

logger = logging.getLogger(__name__)

__all__ = ["RunCommandArgs", "make_run_command_tool"]


_DEFAULT_DESCRIPTION = (
    "Run a shell command inside the sandbox. Returns the captured "
    "stdout, stderr, and exit code. Non-zero exit codes are returned "
    "in the result (not raised) so the agent can decide how to react."
)


class RunCommandArgs(BaseModel):
    """Input shape for the run-command tool."""

    command: str = Field(
        ...,
        description="The shell command to run inside the sandbox.",
    )
    timeout: float | None = Field(
        default=None,
        description=("Optional wall-clock timeout in seconds. The backend's default applies when omitted."),
    )
    shell: bool = Field(
        default=True,
        description=(
            "When True (default), wrap the command in a shell so "
            "pipelines / redirection work. When False, the command "
            "is invoked directly as argv."
        ),
    )


def make_run_command_tool(
    *,
    session: BaseSandboxSession,
    user: User | str | None = None,
    name: str = "run_command",
    description: str | None = None,
    command_policy: SandboxCommandGuardrail | None = None,
    observability: SandboxObservability | None = None,
) -> FunctionTool:
    """Construct a FunctionTool that runs commands inside ``session``.

    Args:
        session: Live ``BaseSandboxSession`` to forward calls to.
        user: Optional user identity the command runs as.
        name: Tool name surfaced to the LLM (default ``run_command``).
        description: Tool description (default: built-in primer).
        command_policy: Optional ``SandboxCommandGuardrail`` — when set,
            the command is evaluated before forwarding; a rejected command
            emits a ``violation`` audit event (when observability is bound)
            and raises ``SandboxCommandRejected``.
        observability: Optional run-scoped ``SandboxObservability`` — when
            set, the tool emits usage / span / audit / hooks around the call.
    """

    async def _on_invoke(ctx: ToolContext, raw_args: str) -> dict[str, Any]:
        del ctx
        parsed = RunCommandArgs.model_validate_json(raw_args)
        if command_policy is not None:
            verdict = command_policy.evaluate(parsed.command)
            if not verdict.allowed:
                if observability is not None:
                    await observability.on_violation(parsed.command, verdict.reason)
                raise SandboxCommandRejected(command=parsed.command, reason=verdict.reason)
        logger.debug(
            "sandbox.run_command",
            extra={"command": parsed.command, "timeout": parsed.timeout},
        )
        if observability is not None:
            await observability.before_exec(parsed.command)
        result = await session.run(
            parsed.command,
            timeout=parsed.timeout,
            shell=parsed.shell,
            user=user,
        )
        if observability is not None:
            await observability.after_exec(parsed.command, result)
        return {
            "stdout": result.decoded_stdout(errors="replace"),
            "stderr": result.decoded_stderr(errors="replace"),
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
        }

    return FunctionTool(
        name=name,
        description=description or _DEFAULT_DESCRIPTION,
        schema=RunCommandArgs,
        on_invoke=_on_invoke,
    )
