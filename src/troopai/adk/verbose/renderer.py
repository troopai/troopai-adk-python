"""Render verbose-output events to a terminal stream.

Two rendering back-ends ship:

1. **Rich** (preferred when ``use_rich=True`` and ``rich`` is
   installed): prints an icon-prefixed styled line, and for events
   whose payload is long (tool arguments, LLM message bodies,
   handoff reasons) wraps them in a subtle bordered panel.
2. **ANSI fallback** (when Rich is missing or ``use_rich=False``):
   plain ANSI escape sequences over the configured output stream.
   Covers the common named colours used by the default style table;
   unknown colour names degrade to plain text without error.

Soft-importing ``rich`` follows the same pattern as
:mod:`troopai.adk.tracing.otel` — the framework has
no hard dependency on Rich; it just gets prettier output when the
user opts in.
"""

from __future__ import annotations

import datetime
import logging
import re
import shutil
import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

from troopai.adk.verbose.config import VerboseConfig

if TYPE_CHECKING:
    from rich.console import Console as _RichConsole

logger = logging.getLogger(__name__)

# Matches CSI, OSC, and similar ANSI escape sequences. Used by the ANSI
# fallback to prevent attacker-controlled tool output from forging log
# lines via cursor movement, clear-line, or OSC exfiltration sequences.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Upper bound on object size (via ``sys.getsizeof``) before ``repr``
# is called inside ``format_payload``. This is a best-effort,
# shallow guard — not a replacement for tools returning bounded data.
_MAX_REPR_OBJECT_BYTES = 4 * 1024 * 1024

# Field names that almost certainly carry secrets. When a tool is
# invoked with a ``dict`` input containing a matching key, the value
# is replaced by ``[REDACTED]`` before rendering.
_SECRET_KEY_RE = re.compile(
    r"(password|passwd|token|secret|api[_-]?key|authorization|bearer|credential|private[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)

# Bounds the depth ``redact_secrets`` descends into nested
# dict/list containers. Tool outputs are arbitrary objects, so a
# self-referential structure (``d = {}; d["x"] = d``) or one nested
# deeper than the interpreter's recursion limit would otherwise raise
# ``RecursionError`` out of the verbose hooks — which must never raise.
# Beyond this depth the value is returned unredacted (best-effort).
_MAX_REDACT_DEPTH = 20


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from untrusted text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def redact_secrets(value: Any, _depth: int = 0) -> Any:
    """Return *value* with obviously-secret dict fields replaced.

    Only ``dict`` inputs are inspected. Keys matching common secret
    patterns (``password``, ``token``, ``api_key``, ``authorization``,
    etc.) have their values replaced by ``"[REDACTED]"``. Other
    values pass through unchanged. The redaction is shallow — nested
    dicts are walked recursively, but non-dict/list containers are
    left alone to avoid breaking representations of custom objects.

    Descent is bounded by :data:`_MAX_REDACT_DEPTH`: tool outputs are
    arbitrary objects, so a self-referential container or one nested
    deeper than the interpreter's recursion limit must not raise
    ``RecursionError`` out of the verbose hooks. Beyond that depth the
    sub-value is returned unredacted (best-effort).

    Args:
        value: Arbitrary object. Typically a ``dict`` of tool
            arguments or a tool result. Non-dict/non-list values are
            returned as-is.

    Returns:
        A new ``dict`` or ``list`` with secret-key values replaced
        by ``"[REDACTED]"``; or the original *value* unchanged when
        it is neither a ``dict`` nor a ``list`` (or the depth bound
        is reached).
    """
    if _depth >= _MAX_REDACT_DEPTH:
        return value
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if isinstance(k, str) and _SECRET_KEY_RE.search(k) is not None
            else redact_secrets(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item, _depth + 1) for item in value]
    return value


# ---------------------------------------------------------------------------
# ANSI colour table (fallback path only)
# ---------------------------------------------------------------------------
#
# We deliberately keep this small — just the colour names used by the
# default style table. Richer palettes are the Rich renderer's job.

_ANSI_RESET = "\033[0m"

_ANSI_COLORS = {
    "": "",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bold cyan": "\033[1;36m",
    "bold green": "\033[1;32m",
    "bold red": "\033[1;31m",
    "bold yellow": "\033[1;33m",
}


def _ansi_wrap(text: str, color: str) -> str:
    """Wrap *text* in the ANSI escapes for *color*, or return as-is.

    Unknown colour names are *not* an error — they pass through
    uncoloured. This keeps the ANSI renderer a permissive subset of
    Rich's style vocabulary.
    """
    code = _ANSI_COLORS.get(color)
    if code is None or len(code) == 0:
        return text
    return f"{code}{text}{_ANSI_RESET}"


def _timestamp() -> str:
    """Return an HH:MM:SS timestamp for line prefixes."""
    return datetime.datetime.now().strftime("%H:%M:%S")


class VerboseRenderer:
    """Render events to the configured output stream.

    One renderer per ``VerboseConfig`` instance. Instances are cheap
    — the Rich Console is lazy-constructed on first use, so a run
    with ``enabled=False`` pays zero rendering cost.
    """

    def __init__(self, config: VerboseConfig) -> None:
        """Bind the renderer to *config*.

        Args:
            config: The :class:`VerboseConfig` that owns this renderer.
                Style lookups, colour resolution, and the output stream
                are all read from *config* at render time.
        """
        self._config = config
        self._console: _RichConsole | None = None
        self._rich_available = False
        self._rich_probed = False

    # ------------------------------------------------------------------
    # Rich soft-import
    # ------------------------------------------------------------------

    def _probe_rich(self) -> bool:
        """Decide whether Rich is usable and cache the result.

        We probe once per renderer to avoid repeated import attempts
        on the hot path. Failure is silent — the ANSI fallback keeps
        output flowing.
        """
        if self._rich_probed:
            return self._rich_available
        self._rich_probed = True
        if not self._config.use_rich:
            return False
        if find_spec("rich") is not None:
            self._rich_available = True
        else:
            logger.debug("Rich not installed; falling back to ANSI renderer")
            self._rich_available = False
        return self._rich_available

    def _get_console(self) -> Any | None:
        """Return (or lazily build) a ``rich.Console`` pinned to our output."""
        if self._console is not None:
            return self._console
        if not self._probe_rich():
            return None
        from rich.console import Console as _Console

        self._console = _Console(
            file=self._config.resolve_output(),
            no_color=not self._config.resolve_use_color(),
            force_terminal=False,
            highlight=False,
        )
        return self._console

    # ------------------------------------------------------------------
    # Public render entrypoints
    # ------------------------------------------------------------------

    def render_line(
        self,
        event: str,
        headline: str,
        payload: str | None = None,
    ) -> None:
        """Emit a single event.

        *headline* is always shown. *payload* is shown when
        ``EventStyle.show_payload`` is True and the string is
        non-empty. When Rich is active and *payload* is longer than
        ~200 chars it renders as a subtle bordered panel; otherwise
        it is appended after the headline on the same line.

        Args:
            event: Dotted event name used to look up the
                :class:`~troopai.adk.verbose.config.EventStyle`
                (colour, icon, prefix).
            headline: Short one-line description always shown
                regardless of payload visibility.
            payload: Optional body text. Shown only when the resolved
                :class:`~troopai.adk.verbose.config.EventStyle` has
                ``show_payload=True`` and this string is non-empty.
        """
        if not self._config.enabled:
            return
        style = self._config.get_style(event)
        effective_color = style.color if self._config.resolve_use_color() else ""
        time_prefix = f"{_timestamp()} " if self._config.show_timestamps else ""
        icon = f"{style.icon} " if len(style.icon) > 0 else ""
        prefix = f"[{style.prefix}] " if len(style.prefix) > 0 else ""
        full_headline = f"{time_prefix}{icon}{prefix}{headline}"

        console = self._get_console()
        show_body = style.show_payload and payload is not None and len(payload) > 0

        if console is not None:
            self._render_rich(console, full_headline, effective_color, show_body, payload)
        else:
            self._render_ansi(full_headline, effective_color, show_body, payload)

    # ------------------------------------------------------------------
    # Back-end implementations
    # ------------------------------------------------------------------

    def _render_rich(
        self,
        console: Any,
        headline: str,
        color: str,
        show_body: bool,
        payload: str | None,
    ) -> None:
        """Render via Rich. ``color`` is a Rich-compatible style string.

        The payload is wrapped in ``rich.text.Text`` before being
        passed to ``Panel`` so that attacker-controlled markup tags
        (``[bold]``, ``[link=...]``) in tool output are rendered as
        literal characters, not interpreted as Rich styling.
        """
        from rich.panel import Panel
        from rich.text import Text

        headline_style = color if len(color) > 0 else ""
        headline_text = Text(headline, style=headline_style)
        console.print(headline_text)
        if show_body and payload is not None and len(payload) > 0:
            if len(payload) > 200:
                width = min(shutil.get_terminal_size((100, 20)).columns, 120)
                console.print(
                    Panel(
                        Text(payload),
                        border_style=color if len(color) > 0 else "dim",
                        width=width,
                        padding=(0, 1),
                    )
                )
            else:
                console.print(Text(f"    {payload}", style="dim"))

    def _render_ansi(
        self,
        headline: str,
        color: str,
        show_body: bool,
        payload: str | None,
    ) -> None:
        """ANSI fallback renderer.

        Strips ANSI escape sequences from *payload* before writing so
        a tool cannot inject cursor-movement or clear-line escapes to
        overwrite prior log output. Writes the whole event in a
        single ``stream.write`` call so concurrent writers can't
        interleave at character granularity on custom ``TextIO``
        sinks.
        """
        stream = self._config.resolve_output()
        line = _ansi_wrap(headline, color) if len(color) > 0 else headline
        buf = [line, "\n"]
        if show_body and payload is not None and len(payload) > 0:
            safe_payload = _strip_ansi(payload)
            for body_line in safe_payload.splitlines() or [safe_payload]:
                buf.append(f"    {body_line}\n")
        stream.write("".join(buf))
        stream.flush()


def format_payload(value: Any, *, max_chars: int = 1000) -> str:
    """Format an arbitrary payload for display.

    Guards against three failure modes before truncation:

    1. Objects whose shallow size (``sys.getsizeof``) exceeds
       :data:`_MAX_REPR_OBJECT_BYTES` are rendered as a placeholder
       instead of calling ``repr`` — avoids OOM on large in-memory
       structures (Pydantic models, NumPy arrays) that a tool may
       return.
    2. A misbehaving ``__repr__`` that raises is caught and replaced
       with a placeholder — rendering must never crash the run.
    3. After materialisation, the result is truncated at *max_chars*
       with a trailing ellipsis so a tool that returns 50 MB of JSON
       cannot flood the terminal.

    Args:
        value: Arbitrary Python object to render. ``None`` returns an
            empty string; ``str`` is used verbatim; everything else
            goes through ``repr``.
        max_chars: Maximum length of the returned string before
            truncation with a trailing ellipsis. Defaults to ``1000``.

    Returns:
        A display-ready string, never longer than ``max_chars`` plus
        the ellipsis suffix; empty string when *value* is ``None``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            size = sys.getsizeof(value)
        except Exception:
            size = 0
        if size > _MAX_REPR_OBJECT_BYTES:
            return f"<{type(value).__name__} too large to display: {size} bytes>"
        try:
            text = repr(value)
        except Exception as exc:
            return f"<{type(value).__name__} repr failed: {exc}>"
    if len(text) > max_chars:
        return f"{text[:max_chars]}… ({len(text) - max_chars} more chars)"
    return text
