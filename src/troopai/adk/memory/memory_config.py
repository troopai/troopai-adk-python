"""Memory configuration for Runner integration.

Defines :class:`MemoryConfig` which controls how memory is injected
into and extracted from agent runs.  Everything is opt-in — no hidden
token costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from troopai.adk.memory.extractor import MemoryExtractor
from troopai.adk.memory.memory import Memory


class MemoryInjectionPosition(StrEnum):
    """Where to inject retrieved memories into the input."""

    SYSTEM_SUFFIX = "system_suffix"
    """Append to the system prompt as a suffix."""

    DEVELOPER_MESSAGE = "developer_message"
    """Insert as a developer-role message before the user prompt."""


@dataclass
class MemoryConfig:
    """Configuration for memory integration with the Runner.

    Controls both injection (memories → context) and extraction
    (context → memories).  All features are opt-in.

    Attributes:
        memory: The memory backend to use.
        namespace: Default namespace for this run.
        inject: Whether to auto-inject relevant memories before the
            agent loop.
        inject_limit: Maximum number of memories to inject.
        inject_position: Where to place injected memories in the input.
        auto_extract: Whether to extract memories after each run.
        extractor: Required if ``auto_extract=True``.
    """

    memory: Memory
    namespace: str
    inject: bool = False
    inject_limit: int = 5
    inject_position: MemoryInjectionPosition = MemoryInjectionPosition.DEVELOPER_MESSAGE
    auto_extract: bool = False
    extractor: MemoryExtractor | None = None

    def __post_init__(self) -> None:
        if self.auto_extract and self.extractor is None:
            raise ValueError(
                "MemoryConfig: auto_extract=True requires an extractor; "
                "set extractor=LLMExtractor(...) to enable extraction."
            )
