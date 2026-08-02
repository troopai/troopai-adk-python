"""Local-execution tools.

Tools that the **developer's environment** executes — not the LLM
provider, and not the framework's `on_invoke` callback. Each class
holds typed config plus a user-supplied callable (``ShellTool.executor``,
``ApplyPatchTool.editor``); the run loop expands each instance into a
``FunctionTool`` at build time using that callable as the executor.

Public exports:

- :class:`ShellTool` / :class:`ShellExecutor` — local shell-command tool.
- :class:`ApplyPatchTool` / :class:`ApplyPatchEditor` — local
  file-patch editor.

Tools NOT in this folder:

- ``MemoryTool`` and ``JITContextAwareTool`` — the framework executes
  these (``ExecutableBuiltinTool.on_invoke`` for memory; build-time
  expansion with framework-supplied executor for JIT). They live at
  the parent ``tools/`` level.
- ``HostedTool`` subclasses — the LLM provider executes these
  server-side. They live in ``tools/hosted/``.
"""

from troopai.adk.tools.local.apply_patch_tool import (
    ApplyPatchEditor,
    ApplyPatchTool,
)
from troopai.adk.tools.local.shell_tool import ShellExecutor, ShellTool

__all__ = [
    "ApplyPatchEditor",
    "ApplyPatchTool",
    "ShellExecutor",
    "ShellTool",
]
