"""Trim a FunctionTool's output to a character or token budget.

This module provides :func:`trim_tool_output`, a lightweight wrapper that
returns a new :class:`FunctionTool` whose ``on_invoke`` stringifies and
trims the original tool's output before handing it back to the runner.

Unlike :attr:`FunctionTool.max_result_tokens` (which operates inside the
runner on the *post-execution* result), this helper composes on the tool
itself — handy when wiring third-party tools whose raw output is too
chatty for the model, without subclassing or re-decorating.
"""

from __future__ import annotations

import logging
from typing import Any

from troopai.adk.context.token_counter import TokenCounter
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

DEFAULT_TRUNCATION_MARKER = "... [truncated]"
"""Suffix appended when a tool result is trimmed."""


def trim_tool_output(
    tool: FunctionTool,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    marker: str = DEFAULT_TRUNCATION_MARKER,
) -> FunctionTool:
    """Return a new ``FunctionTool`` whose output is trimmed to a budget.

    Wraps ``tool.on_invoke`` so that, after execution, the returned value
    is stringified (when ``response_format="text"``) or the content part
    of the tuple is stringified (when
    ``response_format="content_and_artifact"``) and then truncated to
    fit ``max_chars`` and/or ``max_tokens``. The original tool is left
    untouched — a shallow clone is returned via
    :func:`dataclasses.replace`.

    Args:
        tool: The source :class:`FunctionTool`. Must have a non-``None``
            ``on_invoke`` callback.
        max_chars: Hard character cap. When the trimmed text exceeds
            this, it is cut to ``max_chars - len(marker)`` and the
            marker is appended. ``None`` disables the char cap.
        max_tokens: Hard token cap. Requires ``model`` to be set.
            Uses :meth:`TokenCounter.count_text` and, on overflow,
            progressively shrinks via a proportional cut + rechecks
            until the budget is satisfied.
        model: litellm model identifier (e.g. ``"gpt-4o"``) used for
            token counting. Required when ``max_tokens`` is set.
        marker: Suffix appended when truncation occurs. Defaults to
            :data:`DEFAULT_TRUNCATION_MARKER`.

    Returns:
        A new ``FunctionTool`` — same ``name``, ``description``,
        ``schema``, etc. — whose ``on_invoke`` applies trimming.

    Raises:
        ValueError: If neither ``max_chars`` nor ``max_tokens`` is
            provided, or if ``max_tokens`` is set without ``model``,
            or if the source tool has no ``on_invoke``.

    Example:
        >>> from troopai.adk.tools import function_tool
        >>> from troopai.adk.tools.tool_output_trimmer import trim_tool_output
        >>>
        >>> @function_tool
        ... def fetch_docs(query: str) -> str:
        ...     return retrieve_huge_document(query)  # might be 50k chars
        >>>
        >>> trimmed = trim_tool_output(
        ...     fetch_docs,
        ...     max_tokens=2000,
        ...     model="gpt-4o",
        ... )
        >>> agent = Agent(name="rag", tools=[trimmed])
    """
    if max_chars is None and max_tokens is None:
        raise ValueError("trim_tool_output requires at least one of max_chars or max_tokens")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if max_chars is not None and max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if max_tokens is not None and model is None:
        raise ValueError("trim_tool_output requires model when max_tokens is set")
    if tool.on_invoke is None:
        raise ValueError(f"Cannot trim output of tool {tool.name!r}: on_invoke is None")
    if tool.streaming:
        raise ValueError(
            f"Cannot trim output of streaming tool {tool.name!r}: result is an async generator, not a string"
        )

    original_invoke = tool.on_invoke
    response_format = tool.response_format
    tool_name = tool.name

    def _apply_budget(text: str) -> str:
        """Apply max_chars then max_tokens to ``text`` in place."""
        trimmed = text

        if max_chars is not None and len(trimmed) > max_chars:
            if len(marker) >= max_chars:
                # Marker alone would breach the hard cap — drop it and cut
                # to the cap so the result never exceeds max_chars.
                trimmed = trimmed[:max_chars]
            else:
                cutoff = max_chars - len(marker)
                trimmed = trimmed[:cutoff] + marker
            logger.debug(
                "trim_tool_output: char cap applied (tool=%s, before=%d, after=%d)",
                tool_name,
                len(text),
                len(trimmed),
            )

        if max_tokens is not None and model is not None:
            token_count = TokenCounter.count_text(trimmed, model)
            if token_count > max_tokens:
                # Proportional shrink + recheck loop. Two passes almost
                # always converge; cap at 5 to bound worst-case overhead.
                for _ in range(5):
                    ratio = max_tokens / max(token_count, 1)
                    # 0.9 headroom so we undershoot and avoid another pass
                    cutoff = max(1, int(len(trimmed) * ratio * 0.9))
                    trimmed = trimmed[:cutoff] + marker
                    token_count = TokenCounter.count_text(trimmed, model)
                    if token_count <= max_tokens:
                        break
                logger.debug(
                    "trim_tool_output: token cap applied (tool=%s, target=%d, final=%d)",
                    tool_name,
                    max_tokens,
                    token_count,
                )

        return trimmed

    async def trimmed_invoke(ctx: ToolContext, input: str) -> Any:
        result = await original_invoke(ctx, input)

        if response_format == "content_and_artifact":
            if isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                return _apply_budget(str(content)), artifact
            logger.warning(
                "trim_tool_output: tool %r has response_format='content_and_artifact' but returned %s; passing through",
                tool_name,
                type(result).__name__,
            )
            return result

        return _apply_budget(str(result))

    return tool.clone(on_invoke=trimmed_invoke)
