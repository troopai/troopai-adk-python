"""Handoff history transfer strategy."""

from enum import StrEnum


class HandoffStrategy(StrEnum):
    """Controls which conversation messages are transferred during handoff.

    - ``FULL``: Include the entire conversation history.
    - ``LAST_N``: Include the last N messages (requires ``window`` on
      :class:`HandoffConfig`).
    - ``INTENT_ONLY``: Include only the detected intent.
    - ``SUMMARY``: Include a summarized version of the conversation
      (routes through the LLM — accumulates usage).
    """

    FULL = "full"
    """Include the entire conversation history."""

    LAST_N = "last_n"
    """Include the last N messages (requires ``window``)."""

    INTENT_ONLY = "intent_only"
    """Include only the detected intent."""

    SUMMARY = "summary"
    """Include a summarized version of the conversation."""
