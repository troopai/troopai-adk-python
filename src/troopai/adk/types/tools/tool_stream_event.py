"""Streaming event yielded by a streaming function tool.

A *streaming function tool* is a :class:`FunctionTool` whose
``on_invoke`` returns ``AsyncIterator[ToolStreamEvent]`` instead of a
single value. Yielding :class:`ToolStreamEvent` instances exposes
incremental progress to consumers of ``Runner.arun(stream=True)``
(surfaced as ``RunItemType.TOOL_PARTIAL_OUTPUT`` events) while the
LLM still sees exactly one tool-result message — the value carried
on the terminal ``"done"`` event.

Mirrors the shape of ``LLMStreamEvent`` so streaming tools and
streaming LLM responses share the same discriminator vocabulary.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal


@dataclasses.dataclass
class ToolStreamEvent:
    """One streaming event yielded by a streaming function tool.

    Part-based events with the same discriminator vocabulary as
    ``LLMStreamEvent``:

    - ``"part_start"``: a new logical part begins. ``index``
      identifies which part.
    - ``"part_delta"``: incremental progress. ``delta`` carries the
      text fragment.
    - ``"part_end"``: a part is finalized.
    - ``"done"``: streaming complete. ``response`` carries the final
      accumulated value the LLM will see as the tool's result.

    Attributes:
        type: Event type discriminator.
        index: Logical part index (for ``part_start`` / ``part_delta``
            / ``part_end``).
        delta: Text fragment for ``part_delta`` events.
        response: Final accumulated value for the ``done`` event.
            Stringified by the executor before being sent to the LLM,
            mirroring how a non-streaming tool's return value is
            handled.
    """

    type: Literal["part_start", "part_delta", "part_end", "done"]
    """Event type discriminator."""

    index: int | None = None
    """Logical part index (which part this event applies to)."""

    delta: str | None = None
    """Text fragment (``part_delta`` events only)."""

    response: Any = None
    """Final accumulated value (``done`` events only).

    Sent to the LLM as the tool's single result message.
    """
