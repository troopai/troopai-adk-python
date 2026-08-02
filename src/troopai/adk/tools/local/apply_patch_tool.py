"""Apply patch tool for diff-based file editing.

Provides ``ApplyPatchTool`` — a tool that enables the LLM to modify
files via unified diff patches.  Patches are applied locally via an
abstract ``ApplyPatchEditor``.  The tool includes approval gates for
safety.

Example::

    from troopai.adk.tools import ApplyPatchTool

    agent = Agent(
        name="Code Editor",
        system_prompt="Help users edit code files.",
        tools=[ApplyPatchTool(approval=True)],
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from troopai.adk.tools.builtin.builtin_tool import BuiltinTool
from troopai.adk.utils import MaybeAwaitable


class ApplyPatchEditor(ABC):
    """Abstract editor for applying patches to files.

    Concrete implementations handle actual file modification
    (e.g. local filesystem, virtual filesystem, Git worktree).
    """

    @abstractmethod
    async def apply(self, patch: str) -> str:
        """Apply a unified diff patch.

        Args:
            patch: The unified diff patch string.

        Returns:
            A summary of applied changes (e.g. files modified, lines changed).
        """


@dataclass(kw_only=True)
class ApplyPatchTool(BuiltinTool):
    """A tool for applying diff-based file patches with safety controls.

    Patches are applied locally via an ``ApplyPatchEditor`` — not by
    the LLM provider.  The tool includes approval gates for safety.

    When ``editor`` is ``None``, the LLM layer is responsible for
    providing a default editing mechanism.

    Attributes:
        name: The tool type identifier. Always ``"apply_patch"``.
        editor: Abstract editor for applying patches. ``None`` means
            the framework or LLM layer provides a default editing
            mechanism.
        approval: Whether patches require human approval. ``False``
            applies immediately; ``True`` always requires approval;
            a callable receives patch details and returns ``bool``.
    """

    name: str = "apply_patch"
    """str: The tool type identifier. Always ``"apply_patch"``."""

    editor: ApplyPatchEditor | None = None
    """Optional[ApplyPatchEditor]: Editor for applying patches.

    ``None`` means the framework or LLM layer provides a default
    editing mechanism.
    """

    approval: bool | Callable[..., MaybeAwaitable[bool]] = False
    """Whether patches require human approval.

    - ``False``: Patches apply immediately.
    - ``True``: All patches require approval.
    - Callable: Receives patch details and returns bool.
    """
