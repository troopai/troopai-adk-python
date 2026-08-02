"""``HandoffCollapseMode`` — typed collapse policy for handoff history.

Three discrete strategies live on the same axis: keep the
conversation as-is, wrap it into a single system message, or wrap
it into a single user message. An enum disambiguates them where a
bool flag would collapse two distinct wrap-targets into one value.
"""

from __future__ import annotations

from enum import StrEnum


class HandoffCollapseMode(StrEnum):
    """Policy for collapsing transferred handoff history.

    - ``OFF`` (default): replay each transferred message individually
      so the target agent sees the full conversation history.
    - ``SYSTEM_MESSAGE``: collapse the entire transferred history into a
      single block, reducing token count at the cost of message-level
      fidelity. Named for the intent of folding the prior conversation
      into background context, but the block is emitted as a ``user``
      message rather than a system one: after a handoff the target
      agent's own system prompt is injected at index 0, so a collapsed
      system message there would be overwritten — silently discarding
      the transferred history. A user block survives as the target's
      prior context, preceded by that injected system prompt.
    - ``USER_MESSAGE``: the same single-block collapse, likewise emitted
      as a ``user`` message so the target treats the prior conversation
      as the user's full prior context. Emission matches
      ``SYSTEM_MESSAGE``.
    """

    OFF = "off"
    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"
