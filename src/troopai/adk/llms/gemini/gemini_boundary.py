"""Boundary-normalisation helpers for the Gemini SDK.

Mirrors the OpenAI / Anthropic boundary modules: the framework stores
``extra_headers`` as ``Mapping[str, object]`` (richer than the SDK's
typed wire shape), so we coerce them at the API boundary. Single
source of truth for the auth-header blocklist and value
stringification.

The SDK's ``HttpOptions(headers=...)`` parameter is the boundary
target; ``headers_as_sdk`` produces a value the SDK accepts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


# Case-insensitive header names that MUST NOT flow through
# ``extra_headers``. The Gemini SDK builds ``x-goog-api-key`` from
# the ``api_key`` constructor arg; letting callers override it via
# ``extra_headers`` would silently rotate auth mid-session and bypass
# the SDK's per-client credential isolation. ``authorization`` is
# blocked because it carries OAuth bearer tokens for Vertex AI; the
# SDK manages those via the ``credentials`` constructor parameter.
# ``proxy-authorization`` is blocked for proxy-credential isolation.
_HEADER_BLOCKLIST: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-goog-api-key",
        "x-google-api-key",
        # ``x-goog-user-project`` re-attributes Vertex AI billing /
        # quota to a different GCP project. Block it from
        # ``extra_headers`` so a misconfigured caller cannot
        # silently route costs elsewhere.
        "x-goog-user-project",
    }
)


# Characters that line-oriented log parsers interpret as record
# separators. Strip them all so a model- or user-controlled string
# cannot inject a fake log line.
_LOG_BREAKS: tuple[str, ...] = ("\n", "\r", "\x85", " ", " ")


def headers_as_sdk(headers: Mapping[str, object] | None) -> Any:
    """Normalise framework ``extra_headers`` to the Gemini SDK shape.

    Three boundary concerns, one helper:

    1. Stringify non-str values so the SDK's ``dict[str, str]``
       contract is met.
    2. Strip security-sensitive headers (``Authorization`` /
       ``x-goog-api-key`` / ``x-google-api-key`` /
       ``proxy-authorization``, case-insensitive). Credentials belong
       in the SDK's ``api_key`` / ``credentials`` constructor
       parameters, not in ``extra_headers``.
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
                "Pass credentials via the SDK's api_key / credentials constructor argument instead.",
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
    open a log-forging vector.

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
