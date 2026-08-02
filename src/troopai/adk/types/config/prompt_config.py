"""Schema model for a declarative dynamic system prompt.

A ``DynamicPromptRef`` names a Python callable (a ``DynamicSystemPrompt``)
that computes the system prompt at run time from the run context and agent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DynamicPromptRef(BaseModel):
    """Reference to a dynamic system-prompt callable.

    Attributes:
        dynamic: Dotted-path reference to a ``DynamicSystemPrompt`` callable.
    """

    model_config = ConfigDict(extra="forbid")

    dynamic: str = Field(min_length=1)
    """Dotted-path reference to a ``DynamicSystemPrompt`` callable."""
