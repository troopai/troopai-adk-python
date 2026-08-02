"""``File`` and ``Image`` — first-class multimodal argument types.

Both types accept exactly one of three carrier shapes — ``data``
(bytes), ``path`` (filesystem path), ``url`` (HTTP or
``data:``-scheme URL) — validated at construction. The Pydantic
core schema is a plain string so JSON-schema-driven tool calling
sees a URI parameter; a custom validator constructs the instance
from either a string (the LLM-provided shape) or an existing
:class:`File` / :class:`Image` (the developer-supplied shape).

See ``types/multimodal/__init__.py`` for the usage walkthrough.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_DATA_URL_RE = re.compile(
    # ``data:<mime>(;<param>=<value>)*[;base64],<body>`` per RFC 2397.
    # Parameters can appear in any order; ``;base64`` is detected
    # case-insensitively inside the captured ``params`` blob rather
    # than as a fixed position so callers can't fall through silently.
    r"^data:(?P<mime>[\w.+\-]+/[\w.+\-]+)?(?P<params>(?:;[^,]*)*),(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)

_URL_SCHEMES = ("http://", "https://", "data:", "file://")


def _looks_like_url(value: str) -> bool:
    """Return whether ``value`` is an absolute URL the framework
    recognizes (http, https, data, file)."""
    return value.startswith(_URL_SCHEMES)


@dataclass(frozen=True)
class File:
    """A multimodal file argument.

    Exactly one of ``data``, ``path``, ``url`` MUST be set. The other
    two MUST be ``None``. Construction enforces this; serialization
    preserves the carrier shape — a ``url``-backed ``File`` survives
    round-trip as a URL, a ``data``-backed one as a data-URL, and a
    ``path``-backed one as the absolute path string.

    Attributes:
        ALLOWED_MIME_PREFIXES: Class-level MIME prefix allowlist. ``None``
            means any non-empty MIME is accepted. Subclasses narrow this.
        data: Raw binary content. ``None`` when the file is referenced
            by path or URL.
        path: Local filesystem path. ``None`` when the file is carried
            inline or by URL.
        url: HTTP(S) / ``data:`` / ``file://`` URL. ``None`` when the
            file is carried inline or by path.
        mime_type: Optional MIME hint (``"image/png"``, ``"text/csv"``,
            …). The framework prefers this over MIME extracted from the
            URL or filename when both are present.
    """

    # Subclasses (notably :class:`Image`) narrow this set if a stricter
    # contract makes sense. ``File`` itself accepts any non-empty MIME.
    ALLOWED_MIME_PREFIXES: ClassVar[tuple[str, ...] | None] = None

    data: bytes | None = None
    """Raw binary content. ``None`` when the file is carried by path or URL."""

    path: Path | None = None
    """Local filesystem path. ``None`` when the file is carried inline or by URL."""

    url: str | None = None
    """HTTP(S) / ``data:`` / ``file://`` URL. ``None`` when the file is
    carried inline or by path."""

    mime_type: str | None = None
    """Optional MIME hint."""

    def __post_init__(self) -> None:
        present_names = [
            name for name, value in (("data", self.data), ("path", self.path), ("url", self.url)) if value is not None
        ]
        if len(present_names) != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one of "
                f"data / path / url to be set; got {len(present_names)} "
                f"({', '.join(present_names) if present_names else 'none'})."
            )
        if self.url is not None and not _looks_like_url(self.url):
            raise ValueError(f"{type(self).__name__}.url must start with one of {_URL_SCHEMES}; got {self.url!r}.")
        if self.mime_type is not None and len(self.mime_type) == 0:
            raise ValueError(f"{type(self).__name__}.mime_type must be non-empty when set.")
        allowed = type(self).ALLOWED_MIME_PREFIXES
        if (
            allowed is not None
            and self.mime_type is not None
            and not any(self.mime_type.startswith(p) for p in allowed)
        ):
            raise ValueError(
                f"{type(self).__name__}.mime_type {self.mime_type!r} does not match any allowed prefix in {allowed}."
            )

    def kind(self) -> Literal["data", "path", "url"]:
        """Return which carrier holds this file's content.

        Exactly one of ``data`` / ``path`` / ``url`` is set per the
        construction invariant, so this discriminates safely without
        the caller checking three Nones.

        Returns:
            ``"data"``, ``"path"``, or ``"url"`` indicating which
            carrier field is set.
        """
        if self.data is not None:
            return "data"
        if self.path is not None:
            return "path"
        # url is the remaining option per the construction invariant.
        return "url"

    def read_bytes(self) -> bytes:
        """Return the file's raw bytes, reading the path if needed.

        For URL-backed files, raises — fetching arbitrary URLs is the
        caller's responsibility (HTTP client choice, timeout policy,
        auth headers) and is intentionally outside this method's
        contract. ``data:``-scheme URLs with inline base64 are handled
        by ``_from_string`` at construction time, so a URL-backed
        instance at this point genuinely needs external fetching.

        Returns:
            The file's bytes.

        Raises:
            ValueError: when called on a URL-backed instance.
        """
        if self.data is not None:
            return self.data
        if self.path is not None:
            return self.path.read_bytes()
        raise ValueError(
            f"{type(self).__name__}.read_bytes() requires a data- or path-backed "
            "instance; for URL-backed files fetch the URL externally."
        )

    def as_data_url(self) -> str:
        """Return a ``data:<mime>;base64,…`` URL for inline carriage.

        Useful when handing a ``File`` to a provider message that
        doesn't accept a separate binary attachment channel. Works
        only when the file is backed by ``data`` or ``path`` (URL-
        backed files are returned as-is when already a data-URL,
        otherwise raise — fetching arbitrary URLs is the caller's
        responsibility, not this method's).

        Returns:
            A ``data:`` URL string.

        Raises:
            ValueError: when called on a URL-backed ``File`` whose
                URL is not already a ``data:`` URL.
        """
        mime = self.mime_type if self.mime_type is not None else self._default_mime()
        if self.data is not None:
            body = base64.b64encode(self.data).decode("ascii")
            return f"data:{mime};base64,{body}"
        if self.path is not None:
            body = base64.b64encode(self.path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{body}"
        # URL-backed
        if self.url is not None and self.url.startswith("data:"):
            return self.url
        raise ValueError(
            f"{type(self).__name__}.as_data_url() requires data- or path-backed "
            "carriage, or a URL that is already a data-URL. Fetch the URL "
            "externally and pass the bytes as `data=...` instead."
        )

    @classmethod
    def _default_mime(cls) -> str:
        """Return the MIME emitted for a mime-less data-URL.

        Must satisfy this class's own ``ALLOWED_MIME_PREFIXES`` so a
        mime-less ``data``- or ``path``-backed instance survives its
        own serialize → deserialize round-trip: ``as_data_url`` emits
        this MIME, ``_from_string`` reads it back, and ``__post_init__``
        re-validates it. ``File`` (no prefix restriction) keeps the
        generic binary default; a subclass with a prefix ending in
        ``"/"`` (e.g. ``Image`` → ``("image/",)``) gets a concrete
        subtype appended (``"image/octet-stream"``).
        """
        allowed = cls.ALLOWED_MIME_PREFIXES
        if allowed is None:
            return "application/octet-stream"
        prefix = allowed[0]
        return (prefix + "octet-stream") if prefix.endswith("/") else prefix

    @classmethod
    def _from_string(cls, value: str) -> File:
        """Construct from a string — the LLM-facing carrier shape.

        Heuristic:
        - URL scheme prefix (``http://``, ``https://``, ``file://``,
          ``data:``) → ``url=value``; data URLs additionally extract
          ``mime_type`` and decoded ``data`` when ``;base64`` is set.
        - Otherwise → treat as filesystem path → ``path=Path(value)``.

        Args:
            value: The string the LLM produced.

        Returns:
            A new instance of the calling class with the inferred
            carrier set.
        """
        if len(value) == 0:
            raise ValueError(f"{cls.__name__} cannot be constructed from an empty string.")
        if _looks_like_url(value):
            if value.startswith("data:"):
                m = _DATA_URL_RE.match(value)
                if m is not None:
                    params = (m.group("params") or "").lower()
                    if ";base64" in params:
                        return cls(
                            data=base64.b64decode(m.group("body")),
                            mime_type=m.group("mime"),
                        )
                # Non-base64 data URL (rare but valid per RFC 2397) —
                # keep as url-backed; the body is URL-encoded text.
            return cls(url=value)
        return cls(path=Path(value))

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Tell Pydantic this type round-trips through a string.

        Accepts either an existing instance (developer-supplied) or a
        string (LLM-supplied). Emits a JSON schema of
        ``{"type": "string", "format": "uri"}`` so the LLM is asked
        for a URI.
        """
        del source_type, handler

        def _validate(value: object) -> File:
            if isinstance(value, cls):
                return value
            if isinstance(value, str):
                return cls._from_string(value)
            raise ValueError(
                f"{cls.__name__} expected a {cls.__name__} instance or string; got {type(value).__name__}."
            )

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda inst: (
                    inst.url
                    if inst.url is not None
                    else (str(inst.path) if inst.path is not None else inst.as_data_url())
                ),
                when_used="json",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: Any,
    ) -> dict[str, Any]:
        """Override the JSON schema to advertise a URI string.

        ``handler`` is :class:`pydantic.json_schema.GetJsonSchemaHandler`;
        we ignore the inferred ``CoreSchema`` because the validator
        accepts any object — the LLM-facing contract is a URI string.
        """
        del schema, handler
        return {
            "type": "string",
            "format": "uri",
            "description": (
                f"A {cls.__name__} as a URI: an http(s) URL, a "
                "data: URL (inline base64), a file:// URL, or a "
                "local filesystem path."
            ),
        }


@dataclass(frozen=True)
class Image(File):
    """A multimodal image argument.

    Narrows the parent ``File`` contract: when ``mime_type`` is set,
    it MUST start with ``image/``. Otherwise identical to :class:`File`.
    """

    ALLOWED_MIME_PREFIXES: ClassVar[tuple[str, ...] | None] = ("image/",)
