"""Context management for TroopAI Agents.

Provides token counting, context editing, compaction (LLM summarisation),
and a unified :class:`ContextManager` that orchestrates all strategies.
"""

from .compaction import CompactionResult, ContextCompactor
from .context_config import (
    CacheStrategy,
    CompactionConfig,
    ContextEditingConfig,
    ContextManagementConfig,
    TokenUsage,
)
from .context_editing import ContextEditor
from .context_manager import ContextManager
from .directives import (
    CompactDirective,
    ContextDirective,
    DirectiveStore,
    DropDirective,
    apply_directives,
)
from .token_counter import TokenCounter

__all__ = [
    "CacheStrategy",
    "CompactDirective",
    "CompactionConfig",
    "CompactionResult",
    "ContextCompactor",
    "ContextDirective",
    "ContextEditingConfig",
    "ContextEditor",
    "ContextManagementConfig",
    "ContextManager",
    "DirectiveStore",
    "DropDirective",
    "TokenCounter",
    "TokenUsage",
    "apply_directives",
]
