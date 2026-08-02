"""Shell tool for executing commands in a controlled environment.

Provides ``ShellTool`` — a tool that enables the LLM to execute shell
commands.  Unlike built-in provider tools, shell commands run locally
via an abstract ``ShellExecutor``.  The tool includes approval gates
and environment configuration for safety.

Example::

    from troopai.adk.tools import ShellTool

    agent = Agent(
        name="DevOps Agent",
        system_prompt="Help with system administration tasks.",
        tools=[ShellTool(approval=True)],
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from troopai.adk.tools.builtin.builtin_tool import BuiltinTool
from troopai.adk.utils import MaybeAwaitable


class ShellExecutor(ABC):
    """Abstract executor for shell commands.

    Concrete implementations handle actual command execution
    (e.g. local subprocess, Docker container, remote SSH).
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        working_directory: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> str:
        """Execute a shell command and return the output.

        Args:
            command: The shell command to execute.
            timeout: Optional timeout in seconds.
            working_directory: Optional working directory.
            environment: Optional environment variables to set.

        Returns:
            The command's stdout/stderr output as a string.
        """


@dataclass(kw_only=True)
class ShellTool(BuiltinTool):
    """A tool for executing shell commands with safety controls.

    Shell commands run locally via a ``ShellExecutor`` — not by the
    LLM provider.  The tool includes approval gates for safety.

    When ``executor`` is ``None``, the LLM layer is responsible for
    providing a default execution mechanism.

    Attributes:
        name: The tool type identifier. Always ``"shell"``.
        executor: Abstract executor for running commands. ``None`` means
            the framework or LLM layer provides a default execution
            mechanism.
        approval: Whether commands require human approval. ``False``
            executes immediately; ``True`` always requires approval;
            a callable receives command details and returns ``bool``.
        environment: Default environment variables. Merged with the
            process environment when executing commands.
        working_directory: Default working directory for commands.
        timeout: Default command timeout in seconds.
    """

    name: str = "shell"
    """str: The tool type identifier. Always ``"shell"``."""

    executor: ShellExecutor | None = None
    """Optional[ShellExecutor]: Executor for running shell commands.

    ``None`` means the framework or LLM layer provides a default
    execution mechanism.
    """

    approval: bool | Callable[..., MaybeAwaitable[bool]] = False
    """Whether shell commands require human approval.

    - ``False``: Commands execute immediately.
    - ``True``: All commands require approval.
    - Callable: Receives command details and returns bool.
    """

    environment: dict[str, str] | None = None
    """Optional[dict[str, str]]: Default environment variables.

    Merged with the process environment when executing commands.
    """

    working_directory: str | None = None
    """Optional[str]: Default working directory for commands."""

    timeout: float | None = None
    """Optional[float]: Default command timeout in seconds."""
