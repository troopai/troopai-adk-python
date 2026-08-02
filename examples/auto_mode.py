"""Auto-mode helpers for running examples unattended.

When ``TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto`` is set in the environment,
these helpers return deterministic, pre-canned answers instead of blocking
on ``input()`` — so an example can run end-to-end in a batch runner (see
``examples/run_examples.py``) or CI with no human at the terminal. In a
normal interactive run they defer to ``input()``, and even then they fall
back to the supplied default if stdin is closed (``EOFError``), so an
example is never left hanging.

Examples opt in by importing these helpers and calling them in place of a
bare ``input()``; nothing is monkey-patched globally.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

AUTO_MODE_ENV = "TROOPAI_EXAMPLES_INTERACTIVE_MODE"


def is_auto_mode() -> bool:
    """Return ``True`` when examples should bypass interactive prompts.

    Reads ``TROOPAI_EXAMPLES_INTERACTIVE_MODE``; auto mode is active when its
    value is ``"auto"`` (case-insensitive).

    Returns:
        Whether auto mode is enabled for this process.
    """
    return os.environ.get(AUTO_MODE_ENV, "").lower() == "auto"


def input_with_fallback(prompt: str, fallback: str) -> str:
    """Return free-text input, or ``fallback`` when no human can answer.

    In auto mode the ``fallback`` is returned immediately. Otherwise the real
    ``input()`` is used, falling back to ``fallback`` if stdin is closed
    (``EOFError``) — e.g. when launched with ``stdin`` redirected from
    ``/dev/null``.

    Args:
        prompt: The text shown to a human in interactive mode.
        fallback: The canned answer used in auto mode or on ``EOFError``.

    Returns:
        The human's response, or ``fallback``.
    """
    if is_auto_mode():
        logger.info("[auto-input] %s -> %s", prompt.strip(), fallback)
        return fallback
    try:
        return input(prompt)
    except EOFError:
        logger.info("[no-stdin] %s -> %s", prompt.strip(), fallback)
        return fallback


def confirm_with_fallback(prompt: str, *, default: bool = True) -> bool:
    """Return a yes/no confirmation, or ``default`` when no human can answer.

    In auto mode (or when stdin is closed) ``default`` is returned. Otherwise
    the answer is read from ``input()`` and parsed: an empty line yields
    ``default``; ``y`` / ``yes`` is ``True``; anything else is ``False``.

    Args:
        prompt: The text shown to a human in interactive mode.
        default: The confirmation used in auto mode or on ``EOFError``.

    Returns:
        The parsed confirmation, or ``default``.
    """
    if is_auto_mode():
        logger.info("[auto-confirm] %s -> %s", prompt.strip(), default)
        return default
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        logger.info("[no-stdin] %s -> %s", prompt.strip(), default)
        return default
    if len(answer) == 0:
        return default
    return answer in {"y", "yes"}
