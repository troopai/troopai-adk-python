from __future__ import annotations

import re


def to_snake_case(name: str) -> str:
    """Convert a display name to a valid snake_case tool name.

    Strips characters that provider tool-name validation rejects
    (anything outside ``[a-zA-Z0-9_]``), so a display name like
    ``"Refund/Billing Agent"`` becomes a wire-safe ``"refund_billing_agent"``
    rather than passing slashes, dots, parentheses, or non-ASCII through
    to the provider API.

    Args:
        name: The display name to convert (e.g. ``"Research Agent"``
            or ``"CustomerSupport"``).

    Returns:
        A lowercase, underscore-separated identifier suitable for use
        as a tool name (e.g. ``"research_agent"``).

    Examples:
        >>> to_snake_case("Research Agent")
        'research_agent'
        >>> to_snake_case("CustomerSupport")
        'customer_support'
        >>> to_snake_case("GPT-4 Helper")
        'gpt_4_helper'
        >>> to_snake_case("Refund/Billing Agent")
        'refund_billing_agent'
    """
    # Insert underscores before uppercase runs (camelCase / PascalCase)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Replace non-alphanumeric characters with underscores
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    # Collapse multiple underscores and strip leading/trailing
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()
