"""System prompt model for structured agent instructions.

Provides a structured way to define system prompts with discrete sections
(role, context, guidelines, tone, constraints, examples, output format, knowledge)
that are rendered into a single prompt string.

Example:
    from troopai.adk.prompts import SystemPrompt, SystemPromptTone

    prompt = SystemPrompt(
        role="You are a senior Python code reviewer specializing in security.",
        context="You work at a fintech company. Code must comply with PCI-DSS.",
        guidelines=["Flag security vulnerabilities immediately", "Always suggest type hints"],
        tone=SystemPromptTone.TECHNICAL,
        constraints=["Never execute code", "Ask for full file if snippet is incomplete"],
        output_format="Use Markdown with headers for each finding."
    )

    # Use directly as agent's system_prompt
    agent = Agent(name="Reviewer", system_prompt=prompt)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ..run import RunContext
from ..utils import MaybeAwaitable

if TYPE_CHECKING:
    from ..agents import Agent


class SystemPromptTone(StrEnum):
    """Predefined tone options for system prompts."""

    FORMAL = "formal"
    """Polished, respectful language with no contractions or slang."""

    INFORMAL = "informal"
    """Relaxed, everyday language with contractions and casual phrasing."""

    TECHNICAL = "technical"
    """Precise, domain-specific terminology targeting expert audiences."""

    CONVERSATIONAL = "conversational"
    """Natural dialogue-style responses, as if speaking to a colleague."""

    FRIENDLY = "friendly"
    """Warm, approachable language that puts the reader at ease."""

    PROFESSIONAL = "professional"
    """Business-appropriate language balancing clarity and courtesy."""


class SystemPrompt(BaseModel):
    """Structured system prompt with discrete sections.

    All fields except ``role`` are optional. When rendered, only non-empty
    fields are included, producing a clean prompt string.

    Attributes:
        role: Core identity and expertise of the agent (required).
        context: Background information or domain context.
        guidelines: Ordered list of behavioral guidelines.
        tone: Desired response tone (enum or free-form string).
        constraints: Hard constraints the agent must follow.
        examples: Few-shot examples to include in the prompt.
        output_format: Instructions for response formatting.
        knowledge: Domain-specific facts or reference material.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    """Core identity and expertise of the agent. Always rendered first.

    Example: ``"You are a senior Python code reviewer specializing in security."``
    """

    context: str | None = None
    """Background information or domain context that frames the conversation.

    Example: ``"You work at a fintech company. Code must comply with PCI-DSS."``
    """

    guidelines: list[str] | None = None
    """Ordered list of behavioral guidelines the agent should follow.

    Example: ``["Flag security vulnerabilities immediately", "Always suggest type hints"]``
    """

    tone: SystemPromptTone | str | None = None
    """Desired response tone. Accepts a ``SystemPromptTone`` enum or a free-form string.

    Example: ``SystemPromptTone.TECHNICAL`` or ``"friendly but precise"``
    """

    constraints: list[str] | None = None
    """Hard constraints the agent must never violate.

    Example: ``["Never execute code", "Ask for full file if snippet is incomplete"]``
    """

    examples: list[str] | None = None
    """Few-shot examples to include in the prompt for in-context learning.

    Example: ``["Q: What is X?\\nA: X is a framework for ..."]``
    """

    output_format: str | None = None
    """Instructions for how the agent should format its responses.

    Example: ``"Respond in JSON"`` or ``"Use Markdown with headers"``
    """

    knowledge: str | None = None
    """Domain-specific facts or reference material the agent should rely on.

    Example: ``"Python 3.12 introduced type parameter syntax (PEP 695)."``
    """

    def generate(self) -> str:
        """Generate a single system prompt string.

        Sections are assembled in order: role, context, guidelines, tone,
        constraints, examples, output_format, knowledge. A field that is
        ``None`` is skipped entirely. A field that is an empty list (``[]``)
        is NOT ``None``, so its section header is emitted with no bullet
        items beneath it.

        Returns:
            The complete prompt as a single newline-separated string, with
            leading and trailing whitespace stripped.
        """
        sections: list[str] = [self.role]

        if self.context is not None:
            sections.append(f"## Context\n{self.context}")

        if self.guidelines is not None:
            items = "\n".join(f"- {g}" for g in self.guidelines)
            sections.append(f"## Guidelines\n{items}")

        if self.tone is not None:
            sections.append(f"## Tone\n{self.tone}")

        if self.constraints is not None:
            items = "\n".join(f"- {c}" for c in self.constraints)
            sections.append(f"## Constraints\n{items}")

        if self.examples is not None:
            items = "\n\n".join(self.examples)
            sections.append(f"## Examples\n{items}")

        if self.output_format is not None:
            sections.append(f"## Output Format\n{self.output_format}")

        if self.knowledge is not None:
            sections.append(f"## Knowledge\n{self.knowledge}")

        return "\n".join(sections).strip()


@dataclass
class DynamicSystemPromptData:
    """Data passed to a DynamicSystemPrompt callable.

    Attributes:
        context: The run context for the current agent execution, containing
            relevant runtime information and usage metrics.
        agent: The agent instance for which the system prompt is being
            generated.
    """

    context: RunContext[Any]
    """The run context for the current agent execution, containing relevant information"""

    agent: Agent[Any]
    """The agent instance for which the system prompt is being generated"""


DynamicSystemPrompt = Callable[[DynamicSystemPromptData], MaybeAwaitable[str | SystemPrompt]]
"""A callable that receives a ``DynamicSystemPromptData`` bundle and returns a
system prompt string or ``SystemPrompt`` instance. May be sync or async.

Example (sync)::

    def my_prompt(data: DynamicSystemPromptData) -> SystemPrompt:
        guidelines = fetch_guidelines(data.context.context["tenant_id"])
        return SystemPrompt(role="You are a helpful assistant.", knowledge=guidelines)

Example (async)::

    async def my_prompt(data: DynamicSystemPromptData) -> SystemPrompt:
        guidelines = await fetch_guidelines(data.context.context["tenant_id"])
        return SystemPrompt(role="You are a helpful assistant.", knowledge=guidelines)
"""
