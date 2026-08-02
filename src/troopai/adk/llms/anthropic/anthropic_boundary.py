"""Boundary-normalisation helpers for the Anthropic SDK.

Mirrors the OpenAI ``openai_boundary`` module: the framework stores
``metadata``, ``extra_headers``, and similar fields as
``Mapping[str, object]`` (richer than the Anthropic SDK's typed wire
shapes), so we coerce them to the shape ``anthropic.AsyncAnthropic``
accepts at the API boundary. Single source of truth for the
authentication-header blocklist and value stringification.

Every helper returns ``Any`` on purpose: the SDK's typed parameter
(``Mapping[str, str | Omit]`` for headers, etc.) rejects ``None``
under strict type checking, but the runtime accepts ``None`` to mean
"no extras". Returning ``Any`` lets the callers pass ``None`` through
without a per-site ``type: ignore``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


# Case-insensitive header names that MUST NOT flow through
# ``extra_headers``. The ``anthropic`` SDK builds ``x-api-key`` from
# the ``api_key`` constructor arg; letting callers override it via
# ``extra_headers`` would silently rotate auth mid-session and bypass
# the SDK's per-client credential isolation. ``proxy-authorization``
# is also blocked because the SDK manages proxy auth via
# ``http_client``, not via per-request headers — leaking proxy
# credentials into the request line risks them being logged
# upstream. ``x-anthropic-api-key`` is included as a defense-in-depth
# alias even though it is not an official header name. Block
# unconditionally and log a warning so misconfiguration is visible.
_HEADER_BLOCKLIST: frozenset[str] = frozenset(
    {
        "authorization",
        "x-api-key",
        "anthropic-api-key",
        "x-anthropic-api-key",
        "proxy-authorization",
    }
)


# Characters that line-oriented log parsers interpret as record
# separators. ``\n`` and ``\r`` are the obvious ones, but Python's
# ``str.splitlines`` and many SIEM ingestion pipelines also treat
# ``\x85`` (NEL), `` `` (Line Separator), and `` ``
# (Paragraph Separator) as breaks. Strip them all so a model- or
# user-controlled string cannot inject a fake log line.
_LOG_BREAKS: tuple[str, ...] = ("\n", "\r", "\x85", " ", " ")


def metadata_as_sdk(metadata: Mapping[str, object] | None) -> Any:
    """Normalise framework ``Metadata`` to Anthropic's ``MetadataParam`` shape.

    Anthropic's ``MetadataParam`` is ``TypedDict({"user_id"?: str})``
    — a single optional ``user_id`` field. We forward only that key
    when present so a richer framework metadata dict does not break
    the SDK's type contract; everything else is dropped at the
    boundary. Dropped keys are logged at DEBUG so a misconfigured
    caller sees the truncation rather than wondering why their
    ``session_id`` never reached the API.

    Args:
        metadata: Framework metadata mapping, or ``None``.

    Returns:
        ``{"user_id": str}`` when a ``user_id`` key is present;
        ``None`` otherwise.
    """
    if metadata is None:
        return None
    extra_keys = [k for k in metadata if k != "user_id"]
    if len(extra_keys) > 0:
        logger.debug(
            "Anthropic metadata: dropping %d non-user_id keys (%s) — Anthropic's MetadataParam only accepts 'user_id'.",
            len(extra_keys),
            ", ".join(sorted(extra_keys)),
        )
    user_id = metadata.get("user_id")
    if user_id is None:
        return None
    return {"user_id": str(user_id)}


def headers_as_sdk(headers: Mapping[str, object] | None) -> Any:
    """Normalise framework ``extra_headers`` to the Anthropic SDK shape.

    Three boundary concerns, one helper:

    1. Stringify non-str values so the ``Mapping[str, str | Omit]``
       contract the SDK enforces is met.
    2. Strip security-sensitive headers
       (``Authorization`` / ``x-api-key`` / ``anthropic-api-key``,
       case-insensitive) — credentials belong in the SDK's
       ``api_key`` constructor parameter, not in
       ``extra_headers``.
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


def sanitize_for_log(value: str) -> str:
    """Strip line-break characters before logging.

    Log-record fields are sometimes assembled into a single-line
    format; letting an upstream-controlled string inject any of the
    Unicode line terminators (``\\n``, ``\\r``, ``\\x85`` (NEL),
    ``\\u2028`` (LS), ``\\u2029`` (PS)) would split the record and
    open a log-forging vector — fake log lines, spoofed severities
    in downstream parsers. Strip all of them at the boundary
    rather than asking every log call site to remember.

    Args:
        value: The string to sanitize (e.g. a model name from an API
            response).

    Returns:
        The input string with all line-break characters replaced by
        spaces.
    """
    out = value
    for ch in _LOG_BREAKS:
        out = out.replace(ch, " ")
    return out
