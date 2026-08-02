"""Capability-bound function tools.

Each module here defines a FunctionTool factory that takes a live
``BaseSandboxSession`` (+ optional user identity) and returns a
fully-configured tool the agent loop can invoke. The corresponding
capability (Shell, Filesystem, Skills) wires the factory into its
``tools()`` method.
"""

from __future__ import annotations

from troopai.adk.sandbox.tools.run_command_tool import (
    RunCommandArgs,
    make_run_command_tool,
)

__all__ = [
    "RunCommandArgs",
    "make_run_command_tool",
]
