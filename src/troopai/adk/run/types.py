"""Run-layer type aliases."""

from __future__ import annotations

from troopai.adk.types.input import LLMInputContentItem

type UserPrompt = str | list[LLMInputContentItem]
"""The user's input to an agent run.

Either a plain string (most common) or a pre-built list of
provider-agnostic input items (for multimodal input, session history
replay, or manually constructed conversations).
"""
