"""Run the external deploy CLIs (docker/gcloud/kubectl/aws/helm).

``CommandRunner`` is the seam between deploy logic and the host's
binaries. :class:`SubprocessRunner` shells out to the operator's
installed CLIs; tests inject :class:`RecordingRunner` to assert the exact
argv without touching a cloud. The framework imports no cloud SDK — the
deploy path drives the same commands an operator would run by hand, so
there are no extra runtime dependencies.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one external command.

    Attributes:
        returncode: Process exit status (``0`` on success).
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Seam over external-process execution."""

    def which(self, tool: str) -> bool:
        """Return ``True`` if *tool* resolves on the system ``PATH``."""
        ...

    def run(self, args: Sequence[str], *, cwd: Path | None = None, input_text: str | None = None) -> CommandResult:
        """Run *args* and capture the result; never raises on non-zero exit.

        ``input_text`` is fed to the process stdin — used to pipe a token to
        ``docker login --password-stdin`` without putting it in argv.
        """
        ...


@dataclass(frozen=True)
class SubprocessRunner:
    """``CommandRunner`` that shells out to the operator's installed CLIs."""

    def which(self, tool: str) -> bool:
        """Return ``True`` if *tool* is resolvable on ``PATH``.

        Args:
            tool: The CLI name (e.g. ``"docker"``).

        Returns:
            Whether the executable is found.
        """
        return shutil.which(tool) is not None

    def run(self, args: Sequence[str], *, cwd: Path | None = None, input_text: str | None = None) -> CommandResult:
        """Run *args* with output captured.

        The executable is a fixed deploy-CLI name resolved from ``PATH``
        and the argv is a list (no shell, no string interpolation of
        untrusted input).

        Args:
            args: The argv list; first element is the executable.
            cwd: Optional working directory.
            input_text: Optional text fed to the process stdin.

        Returns:
            The captured :class:`CommandResult`.

        Raises:
            ValueError: If *args* is empty.
        """
        if len(args) == 0:
            raise ValueError("run() requires a non-empty argument list.")
        logger.debug("exec: %s (cwd=%s)", " ".join(args), cwd)
        completed = subprocess.run(list(args), cwd=cwd, input=input_text, capture_output=True, text=True, check=False)
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass
class RecordingRunner:
    """Test ``CommandRunner`` that records calls instead of executing.

    Attributes:
        calls: The argv lists passed to :meth:`run`, in order.
        inputs: The ``input_text`` passed to each call, aligned with ``calls``.
        available: Tools :meth:`which` reports present; all present when
            empty.
        results: Per-call canned results, indexed by call order; calls
            past the end return success.
    """

    calls: list[list[str]] = field(default_factory=list)
    inputs: list[str | None] = field(default_factory=list)
    available: set[str] = field(default_factory=set)
    results: list[CommandResult] = field(default_factory=list)

    def which(self, tool: str) -> bool:
        """Report *tool* present (all present when ``available`` is empty)."""
        return len(self.available) == 0 or tool in self.available

    def run(self, args: Sequence[str], *, cwd: Path | None = None, input_text: str | None = None) -> CommandResult:  # noqa: ARG002
        """Record *args* / ``input_text`` and return the canned (or success) result.

        ``cwd`` is part of the :class:`CommandRunner` contract; the
        recorder ignores it.
        """
        self.calls.append(list(args))
        self.inputs.append(input_text)
        index = len(self.calls) - 1
        if index < len(self.results):
            return self.results[index]
        return CommandResult(0, "", "")


class DeployToolMissing(RuntimeError):
    """A required external CLI is not installed."""

    def __init__(self, tool: str) -> None:
        super().__init__(f"required tool {tool!r} is not installed or not on PATH")
        self.tool = tool


class DeployCommandFailed(RuntimeError):
    """An external deploy command exited non-zero."""

    def __init__(self, command: list[str], result: CommandResult) -> None:
        detail = result.stderr.strip() if len(result.stderr.strip()) > 0 else result.stdout.strip()
        super().__init__(f"command failed (exit {result.returncode}): {' '.join(command)}\n{detail}")
        self.command = command
        self.result = result


def require_tool(runner: CommandRunner, tool: str) -> None:
    """Raise :class:`DeployToolMissing` if *tool* is unavailable to *runner*.

    Args:
        runner: The command runner to query.
        tool: The CLI name.

    Raises:
        DeployToolMissing: When the tool is not on ``PATH``.
    """
    if not runner.which(tool):
        raise DeployToolMissing(tool)


def run_checked(
    runner: CommandRunner, args: Sequence[str], *, cwd: Path | None = None, input_text: str | None = None
) -> CommandResult:
    """Run a command and raise on non-zero exit.

    Args:
        runner: The command runner.
        args: The argv list.
        cwd: Optional working directory.
        input_text: Optional text fed to the process stdin.

    Returns:
        The successful :class:`CommandResult`.

    Raises:
        DeployCommandFailed: If the command exits non-zero.
    """
    result = runner.run(args, cwd=cwd, input_text=input_text)
    if result.returncode != 0:
        raise DeployCommandFailed(list(args), result)
    return result
