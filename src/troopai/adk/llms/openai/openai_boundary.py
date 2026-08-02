"""Boundary-normalisation helpers shared by both OpenAI model classes.

The framework stores ``metadata``, ``extra_headers``, and
``extra_query`` as ``Mapping[str, object]`` (richer than the
SDK's typed wire shapes). These helpers coerce those framework
values to the shape ``openai.AsyncOpenAI`` accepts at the API
boundary — one place, two callers.

Defined separately (rather than duplicated per model) so the
``Authorization`` / ``x-api-key`` blocklist in
:func:`headers_as_sdk` is a single load-bearing check instead of
two easily-drifting copies.

Every helper returns ``Any`` on purpose: the SDK's typed parameter
(``Mapping[str, str | Omit]`` for headers, ``Dict[str, str]`` for
metadata, ``Mapping[str, str | list[str] | None]`` for query)
rejects ``None`` under strict type checking, but the runtime
accepts ``None`` to mean "no extras". Returning ``Any`` lets the
callers pass ``None`` through without a per-site
``type: ignore``. The runtime contract — ``None`` or a normalised
``dict`` — is preserved.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Every Unicode line terminator a single-line log-record splitter recognises:
# LF, CR, NEL (\x85), LS ( ), PS ( ). Stripping only \n/\r would still
# let an upstream-controlled string forge log lines via the three Unicode
# breaks. Mirrors the anthropic/gemini boundaries so the defense does not drift.
_LOG_BREAKS: tuple[str, ...] = ("\n", "\r", "\x85", " ", " ")


# Case-insensitive header names that MUST NOT flow through
# ``extra_headers``. The ``openai`` SDK builds ``Authorization`` from
# the ``api_key`` constructor arg; letting callers override it via
# ``extra_headers`` would (a) silently rotate auth mid-session, (b)
# log the credential at the INFO-level request line in debug builds,
# and (c) bypass the SDK's per-client credential isolation. Block
# unconditionally and log a warning so misconfiguration is visible.
_HEADER_BLOCKLIST: frozenset[str] = frozenset({"authorization", "x-api-key", "openai-api-key"})


def metadata_as_sdk(metadata: Optional[Mapping[str, object]]) -> Any:
    """Normalise framework ``Metadata`` to the SDK's ``Dict[str, str]`` shape.

    The framework stores metadata as ``Mapping[str, object]`` so
    callers can log typed metadata upstream; the OpenAI SDK's
    ``metadata`` parameter is ``Dict[str, str]``. Stringify values
    at the boundary. ``None`` passes through so the SDK can treat
    it as "no metadata".

    Args:
        metadata: Framework metadata mapping, or ``None``.

    Returns:
        A ``dict[str, str]`` with values stringified, or ``None`` when
        the input is ``None``.
    """
    if metadata is None:
        return None
    return {k: v if isinstance(v, str) else str(v) for k, v in metadata.items()}


def headers_as_sdk(headers: Optional[Mapping[str, object]]) -> Any:
    """Normalise framework-typed ``extra_headers`` to the SDK's expected shape.

    Three boundary concerns, one helper:

    1. Stringify non-str values so the ``Mapping[str, str | Omit]``
       contract the SDK enforces at type-check time is met.
    2. Strip security-sensitive headers
       (``Authorization`` / ``x-api-key`` / ``openai-api-key``,
       case-insensitive) — credentials belong in the SDK's
       ``api_key`` constructor parameter, not in
       ``extra_headers``. Each stripped header is logged at
       WARNING so misconfiguration is visible in production logs.
    3. Pass ``None`` through unchanged so the SDK treats it as
       "no extras".

    Args:
        headers: Framework extra headers mapping, or ``None``.

    Returns:
        A ``dict[str, str]`` with blocked headers removed and values
        stringified, or ``None`` when the input is ``None``.
    """
    if headers is None:
        return None
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _HEADER_BLOCKLIST:
            logger.warning(
                "Stripped security-sensitive header from extra_headers: %s. "
                "Pass credentials via the SDK's api_key constructor argument instead.",
                k,
            )
            continue
        out[k] = v if isinstance(v, str) else str(v)
    return out


def query_as_sdk(query: Optional[Mapping[str, object]]) -> Any:
    """Normalise framework-typed ``extra_query`` to the SDK's expected shape.

    The SDK accepts query params as a dict; framework callers may
    hand in a ``Mapping``. Copy to a concrete dict and pass ``None``
    through unchanged.

    Args:
        query: Framework extra query mapping, or ``None``.

    Returns:
        A concrete ``dict`` copy, or ``None`` when the input is
        ``None``.
    """
    if query is None:
        return None
    return dict(query)


def sanitize_for_log(value: str) -> str:
    """Strip line-break characters from a value before logging.

    Log-record fields are sometimes assembled into a single-line
    format; letting an upstream-controlled string inject any of the
    Unicode line terminators (``\\n``, ``\\r``, ``\\x85`` (NEL),
    ``\\u2028`` (LS), ``\\u2029`` (PS)) would split the record and
    open a log-forging vector (fake log lines, spoofed severities in
    downstream parsers). The OpenAI API echoes back a ``model`` string
    on every response — routed models can legally contain ``/`` / ``:``
    etc., but line terminators are never valid. Strip them all at the
    boundary rather than asking every log call site to remember.

    Args:
        value: The string to sanitize (e.g. a model name from the
            API response).

    Returns:
        The input string with every line-break character replaced by a
        space.
    """
    out = value
    for ch in _LOG_BREAKS:
        out = out.replace(ch, " ")
    return out
