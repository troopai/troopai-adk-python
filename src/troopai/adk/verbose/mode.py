"""Verbose-mode resolution.

Resolves a :class:`~troopai.adk.verbose.config.VerboseConfig`'s ``mode``
field to a concrete backend (``line``, ``panel``, or ``off``) based on
the current environment. A developer can force a specific backend by
setting ``mode`` explicitly; ``"auto"`` defers to environment detection
so the same code is safe in TTYs, CI containers, piped shells, and
stdout-redirected runs.

Resolution order (high → low precedence):

1. ``mode == "off"`` → ``"off"`` (no rendering at all).
2. ``mode == "line"`` or ``"panel"`` → developer override, returned as-is.
3. ``NO_COLOR`` env set → ``"line"`` (panel output relies on ANSI colour,
   so the user-facing contract is "no colour anywhere").
4. ``FORCE_COLOR`` env set → ``"panel"`` when Rich is importable, else
   ``"line"`` (the user asked for colour explicitly, but we still refuse
   to spawn Rich if it isn't installed).
5. Output stream is not a TTY → ``"line"`` (captured logs, piped shells,
   redirected stdout — panels just garble).
6. ``CI`` / ``TERM=dumb`` → ``"line"`` (many CI providers set a TTY by
   default but still render ANSI poorly in collected logs).
7. Rich is not importable → ``"line"``.
8. Otherwise → ``"panel"``.

Every downgrade emits a single ``logger.debug`` line so operators can
verify mode via file logs if a run renders differently from what they
expected.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import IO, Literal

from troopai.adk.verbose.config import VerboseConfig

logger = logging.getLogger(__name__)

VerboseMode = Literal["auto", "line", "panel", "off"]
"""Developer-facing mode selector on :class:`VerboseConfig`."""

ResolvedMode = Literal["line", "panel", "off"]
"""Concrete backend returned by :func:`resolve_mode`."""


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


def is_no_color() -> bool:
    """Return ``True`` when the ``NO_COLOR`` env var is set.

    Honours the `NO_COLOR standard <https://no-color.org/>`_: any
    non-empty value disables colour universally.
    """
    value = os.environ.get("NO_COLOR", "")
    return len(value) > 0


def is_force_color() -> bool:
    """Return ``True`` when the ``FORCE_COLOR`` env var is set to a
    non-empty, non-``"0"`` value.

    Rich, Node.js and many CLI tools respect this to opt into colour
    even when the output stream is not a TTY (e.g. when CI forwards to
    a colour-capable log collector).
    """
    value = os.environ.get("FORCE_COLOR", "")
    if len(value) == 0:
        return False
    return value != "0"


def is_ci() -> bool:
    """Return ``True`` when running under a CI provider.

    Checks the ``CI`` env var (set by virtually every major CI
    provider — GitHub Actions, GitLab CI, CircleCI, Jenkins, Travis,
    Buildkite…) and ``TERM=dumb``.
    """
    ci = os.environ.get("CI", "")
    if len(ci) > 0 and ci.lower() != "false":
        return True
    term = os.environ.get("TERM", "")
    return term == "dumb"


def is_tty(stream: IO[str] | None) -> bool:
    """Return ``True`` when *stream* is an interactive terminal.

    Uses ``stream.isatty()`` when available. A missing stream or a
    stream without ``isatty`` (e.g. ``io.StringIO`` in tests) is
    treated as non-TTY.

    Args:
        stream: The output stream to probe. ``None`` is treated as
            non-TTY.

    Returns:
        ``True`` when *stream* reports itself as an interactive
        terminal; ``False`` in all other cases.
    """
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (ValueError, OSError):
        return False


def is_rich_available() -> bool:
    """Return ``True`` when ``rich`` can be imported.

    Rich is a soft dependency of the verbose module — the same pattern
    as the OpenTelemetry tracer. If it isn't on the path, the panel
    backend cannot run and we fall back to line mode.
    """
    return importlib.util.find_spec("rich") is not None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_mode(config: VerboseConfig) -> ResolvedMode:
    """Resolve *config*'s declared mode to a concrete backend.

    See the module docstring for the full precedence ladder. Downgrades
    are logged at ``DEBUG`` level; the function never raises.

    Args:
        config: The :class:`~troopai.adk.verbose.config.VerboseConfig`
            whose ``mode`` field and ``enabled`` flag are inspected.

    Returns:
        ``"off"`` when rendering is disabled or the mode is ``"off"``;
        ``"line"`` when the line backend is forced or when the
        environment prevents the panel backend; ``"panel"`` when Rich
        is available and the environment supports it.
    """
    if config.enabled is False:
        return "off"

    declared = config.mode

    if declared == "off":
        return "off"

    if declared == "line":
        return "line"

    if declared == "panel":
        if is_rich_available() is False:
            logger.debug(
                "verbose mode 'panel' requested but rich is not installed; falling back to 'line'",
            )
            return "line"
        return "panel"

    # declared == "auto" — environment-aware selection.
    if is_no_color() is True:
        logger.debug("verbose auto mode: NO_COLOR set, selecting 'line'")
        return "line"

    if is_force_color() is True:
        if is_rich_available() is True:
            logger.debug(
                "verbose auto mode: FORCE_COLOR set and rich available, selecting 'panel'",
            )
            return "panel"
        logger.debug(
            "verbose auto mode: FORCE_COLOR set but rich missing, selecting 'line'",
        )
        return "line"

    stream = config.resolve_output()
    if is_tty(stream) is False:
        logger.debug("verbose auto mode: output is not a TTY, selecting 'line'")
        return "line"

    if is_ci() is True:
        logger.debug(
            "verbose auto mode: CI environment detected, selecting 'line'",
        )
        return "line"

    if is_rich_available() is False:
        logger.debug(
            "verbose auto mode: rich not installed, selecting 'line'",
        )
        return "line"

    return "panel"
