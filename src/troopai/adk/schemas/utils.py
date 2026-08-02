"""JSON Schema normalization and strict transformation.

This module provides two complementary layers for preparing JSON schemas
before sending them to LLM providers:

1. **General normalization** (Strands-style): Ensures schemas have sensible
   defaults (type, required) so they work across any LLM provider.

2. **Strict transformation** (OpenAI-style): Mutates schemas to satisfy OpenAI's
   ``strict: true`` structured output requirements — all properties required,
   no additionalProperties, oneOf→anyOf conversion, $ref resolution, etc.

Typical usage::

    # For any provider — lightweight defaults
    schema = normalize_schema(raw_schema)

    # For OpenAI strict mode — full compliance
    schema = ensure_strict_schema(raw_schema)
"""

from __future__ import annotations

import copy
import enum
import logging
from typing import Any, TypeGuard

from troopai.adk.exceptions import UserError


class SchemaEnforcement(enum.StrEnum):
    """Controls how a JSON schema is processed before being sent to the LLM.

    Attributes:
        NONE: No schema transformation is applied.  The raw schema is
            sent as-is to the provider.
        NORMALIZED: Provider-agnostic normalization via
            :func:`normalize_schema`.  Ensures sensible defaults
            (``type``, ``properties``, ``required``) without imposing
            strict-mode constraints.  Suitable for providers that do
            not support or require strict schemas.
        STRICT: Full OpenAI strict-mode compliance via
            :func:`ensure_strict_schema`.  Enforces
            ``additionalProperties: false``, forces all properties into
            ``required``, converts ``oneOf`` to ``anyOf``, flattens
            single-element ``allOf``, resolves ``$ref`` with sibling
            keys, and removes ``default: null``.
        COMPACT: Strict-mode compliance plus metadata stripping for
            cost optimization.  Applies :func:`ensure_strict_schema`
            then removes ``title`` keys and ``description`` from
            ``const``-valued properties (discriminator fields where
            the constant is self-documenting).

    Example::

        from troopai.adk.schemas import SchemaEnforcement

        tool = FunctionTool(
            name="search",
            description="Search the database",
            schema=SearchInput,
            enforcement=SchemaEnforcement.STRICT,
        )
    """

    NONE = "none"
    NORMALIZED = "normalized"
    STRICT = "strict"
    COMPACT = "compact"


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema enforcement dispatcher
# ---------------------------------------------------------------------------


def enforce_schema(
    schema: dict[str, Any],
    enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED,
) -> dict[str, Any]:
    """Apply schema enforcement rules to a JSON schema.

    Central utility used by both ``FunctionTool.get_json_schema()`` (tool
    input schemas) and ``AgentOutputSchema`` (structured output schemas)
    to apply consistent schema processing.

    Args:
        schema: The raw JSON schema dict.  For ``STRICT`` and
            ``NORMALIZED`` enforcement, callers should pass a copy if
            the original must be preserved (``ensure_strict_schema``
            mutates in place).
        enforcement: The enforcement level to apply.

            - ``STRICT``: Full OpenAI strict-mode compliance via
              :func:`ensure_strict_schema`.
            - ``NORMALIZED``: Provider-agnostic defaults via
              :func:`normalize_schema`.
            - ``NONE``: Return the schema as-is.

    Returns:
        The processed schema.  For ``STRICT`` this is the mutated
        input dict.  For ``NORMALIZED`` this is a shallow copy.
        For ``NONE`` this is the original dict reference.

    Example::

        from troopai.adk.schemas import enforce_schema, SchemaEnforcement

        raw = {"type": "object", "properties": {"name": {"type": "string"}}}

        strict = enforce_schema(raw.copy(), SchemaEnforcement.STRICT)
        # → additionalProperties: false, all required, etc.

        normalized = enforce_schema(raw, SchemaEnforcement.NORMALIZED)
        # → sensible defaults applied

        raw_out = enforce_schema(raw, SchemaEnforcement.NONE)
        # → unchanged
    """
    match enforcement:
        case SchemaEnforcement.STRICT:
            return ensure_strict_schema(schema)
        case SchemaEnforcement.COMPACT:
            return ensure_compact_schema(schema)
        case SchemaEnforcement.NORMALIZED:
            return normalize_schema(schema)
        case SchemaEnforcement.NONE:
            return schema


_COMPOSITION_KEYWORDS = ("anyOf", "oneOf", "allOf", "not")
"""JSON Schema composition keywords that define type constraints."""

_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
"""Canonical empty schema used as a fallback."""


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------


def _is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    """Check whether *obj* is a ``dict``.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if *obj* is a ``dict``, narrowing the type to
        ``dict[str, object]``.
    """
    return isinstance(obj, dict)


def _is_list(obj: object) -> TypeGuard[list[object]]:
    """Check whether *obj* is a ``list``.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if *obj* is a ``list``, narrowing the type to
        ``list[object]``.
    """
    return isinstance(obj, list)


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    """Check whether *obj* has more than *n* keys.

    Uses early exit to avoid iterating the entire dict.

    Args:
        obj: The dictionary to inspect.
        n: The threshold count.

    Returns:
        ``True`` if ``len(obj) > n``.
    """
    return any(i > n for i, _ in enumerate(obj, start=1))


# ---------------------------------------------------------------------------
# Layer 1: General normalization (provider-agnostic, Strands-style)
# ---------------------------------------------------------------------------


def _normalize_property(prop_def: Any) -> dict[str, Any] | bool:
    """Normalize a single property definition with sensible defaults.

    Behaviour summary:

    - A boolean schema (``true`` / ``false``) is returned verbatim.  In JSON
      Schema ``true`` accepts any value and ``false`` accepts none, so it must
      not be rewritten into a typed stub.
    - Any other non-dict value is malformed and is replaced with a minimal
      ``{"type": "string"}`` stub.
    - Nested objects (``type: "object"`` with ``properties``) are
      recursively normalized via :func:`normalize_schema`.
    - Properties containing a ``$ref`` are returned as-is because the
      referenced definition is expected to carry its own type.
    - Properties using composition keywords (``anyOf``, ``oneOf``,
      ``allOf``, ``not``) do not receive a default ``type`` since the
      composition itself defines the type constraint.

    No ``description`` is synthesised: a fabricated ``"Property <name>"``
    string is a framework-added token the developer never opted into, and it
    spends tokens on every request while carrying no information the property
    key does not already convey.

    Args:
        prop_def: The raw property definition.  May be a ``dict`` (the normal
            case), a ``bool`` (a boolean JSON Schema), or any other value
            (treated as malformed and replaced with a string stub).

    Returns:
        The normalized property definition.  A ``bool`` input is returned
        unchanged; a ``dict`` input gains a default ``type`` where
        appropriate.

    References:
        - JSON Schema property definitions:
          https://json-schema.org/understanding-json-schema/reference/object#properties
        - JSON Schema boolean schemas:
          https://json-schema.org/understanding-json-schema/basics#boolean-schemas
    """
    if isinstance(prop_def, bool):
        return prop_def

    if not isinstance(prop_def, dict):
        return {"type": "string"}

    if prop_def.get("type") == "object" and "properties" in prop_def:
        return normalize_schema(prop_def)

    normalized = prop_def.copy()

    # $ref carries its own type — skip defaults.
    if "$ref" in normalized:
        return normalized

    has_composition = any(kw in normalized for kw in _COMPOSITION_KEYWORDS)
    if not has_composition:
        normalized.setdefault("type", "string")
    return normalized


def normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON schema with sensible defaults (provider-agnostic).

    Ensures that ``type``, ``properties``, and ``required`` keys exist and
    that every non-boolean property has at least a ``type``.  This is a
    lightweight pass that does **not** enforce strict-mode constraints such as
    ``additionalProperties: false`` or forcing all properties into
    ``required``, and never synthesises a ``description``.

    Args:
        schema: The JSON schema dictionary to normalize.  Must be a
            ``dict``; the original is not mutated.

    Returns:
        A shallow copy of *schema* with the following defaults applied:

        - ``type`` defaults to ``"object"``.
        - ``properties`` defaults to ``{}``.
        - ``required`` defaults to ``[]``.
        - Each property is individually normalized via
          :func:`_normalize_property`.

    References:
        - Strands SDK ``normalize_schema``:
          https://github.com/strands-agents/sdk-python/blob/main/src/strands/tools/tools.py
        - JSON Schema object type:
          https://json-schema.org/understanding-json-schema/reference/object

    See Also:
        :func:`ensure_strict_schema`: For OpenAI strict-mode compliance
        on top of general normalization.
    """
    normalized = schema.copy()
    if not any(kw in normalized for kw in _COMPOSITION_KEYWORDS):
        normalized.setdefault("type", "object")
        normalized.setdefault("properties", {})
        normalized.setdefault("required", [])

    if "properties" in normalized:
        normalized["properties"] = {name: _normalize_property(defn) for name, defn in normalized["properties"].items()}

    return normalized


# ---------------------------------------------------------------------------
# Layer 2: Strict transformation (OpenAI-style)
# ---------------------------------------------------------------------------


def ensure_strict_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Transform a JSON schema to conform to OpenAI's strict structured output spec.

    This **mutates** the schema in place (and returns it) so that it
    satisfies all constraints required by OpenAI's ``strict: true``
    function-calling / structured-output mode:

    - ``additionalProperties: false`` on every object.
    - All properties listed in ``required``.
    - ``oneOf`` converted to ``anyOf``.
    - Single-element ``allOf`` flattened into the parent schema.
    - ``$ref`` merged inline when sibling keys are present.
    - ``default: null`` entries removed.
    - ``$defs`` / ``definitions`` recursively processed.

    For an empty schema (``{}``), a canonical empty object schema is
    returned instead.

    Args:
        schema: The JSON schema dictionary to transform.  The dict is
            mutated in place; callers should pass a copy if the
            original must be preserved.

    Returns:
        The transformed schema.  For non-empty inputs this is the same
        ``dict`` reference passed in (mutated).  For empty inputs a
        fresh copy of :data:`_EMPTY_SCHEMA` is returned.

    Raises:
        TypeError: If *schema* (or any nested sub-schema encountered
            during recursion) is not a ``dict``.
        UserError: If ``additionalProperties`` is explicitly set to
            ``True`` on any object sub-schema.
        ValueError: If a ``$ref`` uses an unsupported format (i.e. does
            not start with ``#/``) or resolves to a non-dict value.

    References:
        - OpenAI structured output / strict mode:
          https://platform.openai.com/docs/guides/structured-outputs
        - OpenAI Agents SDK ``ensure_strict_json_schema``:
          https://github.com/openai/openai-agents-python/blob/main/src/agents/strict_schema.py

    See Also:
        :func:`normalize_schema`: For lightweight, provider-agnostic
        normalization without strict-mode enforcement.
    """
    if schema == {}:
        return copy.deepcopy(_EMPTY_SCHEMA)
    result = _ensure_strict_schema(schema, path=(), root=schema)
    # Convert const → enum for provider compatibility (Gemini strips const).
    _convert_const_to_enum(result)
    return result


def ensure_compact_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Apply strict-mode compliance then strip metadata for cost optimization.

    Runs :func:`ensure_strict_schema` first, then removes:

    - All ``"title"`` keys (Pydantic metadata, not useful for the LLM).
    - ``"description"`` from properties that have a ``"const"`` value
      (discriminator fields where the constant is self-documenting).

    This reduces token count for ``response_format`` JSON schemas
    without affecting LLM behavior.

    Args:
        schema: The JSON schema dictionary.  A copy should be passed
            if the original must be preserved (mutations from
            :func:`ensure_strict_schema` apply).

    Returns:
        The transformed schema with metadata stripped.
    """
    strict = ensure_strict_schema(schema)
    _strip_schema_metadata(strict)
    return strict


def _strip_schema_metadata(schema: object) -> None:
    """Recursively remove non-essential metadata from a JSON schema.

    Strips:

    - ``"title"`` keys everywhere (Pydantic-generated, adds tokens
      but does not influence LLM output).
    - ``"description"`` from properties that carry a ``"const"`` value
      (discriminator fields where the constant value is
      self-documenting).

    Mutates *schema* in place.

    Args:
        schema: A JSON schema node (dict, list, or scalar).
    """
    if not isinstance(schema, dict):
        return

    # Remove title everywhere
    schema.pop("title", None)

    # Remove description from const-valued (or single-element enum) properties.
    # ``ensure_strict_schema`` converts ``{"const": X}`` to ``{"enum": [X]}``
    # before this function runs, so we must also match the enum-of-one form to
    # correctly strip discriminator-field descriptions in COMPACT mode.
    if "const" in schema or ("enum" in schema and len(schema.get("enum", [])) == 1):
        schema.pop("description", None)

    # Recurse into sub-schemas
    for key in ("properties", "$defs", "definitions"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            for child in sub.values():
                _strip_schema_metadata(child)

    for key in ("items", "additionalProperties"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            _strip_schema_metadata(sub)

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                _strip_schema_metadata(variant)


def _convert_const_to_enum(schema: object) -> None:
    """Recursively convert ``const`` to single-element ``enum`` arrays.

    Some LLM providers (notably Gemini via litellm) strip ``const`` from
    schemas because it's not in their supported field set, but they DO
    support ``enum``.  Since ``{"const": "x"}`` and ``{"enum": ["x"]}``
    are semantically equivalent in JSON Schema, this lossless conversion
    ensures discriminator values survive provider schema transformations.

    Mutates *schema* in place.

    Args:
        schema: A JSON schema node (dict, list, or scalar).
    """
    if not isinstance(schema, dict):
        return

    if "const" in schema:
        schema["enum"] = [schema.pop("const")]

    # Recurse into sub-schemas
    for key in ("properties", "$defs", "definitions"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            for child in sub.values():
                _convert_const_to_enum(child)

    for key in ("items", "additionalProperties"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            _convert_const_to_enum(sub)

    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                _convert_const_to_enum(variant)


def _ensure_strict_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Recursive engine for :func:`ensure_strict_schema`.

    Walks a JSON schema tree depth-first, mutating each node to satisfy
    OpenAI strict-mode constraints.

    Args:
        json_schema: The current schema node to process.  Must be a
            ``dict``; raises :class:`TypeError` otherwise.
        path: The traversal path from the root to this node, used for
            diagnostic messages (e.g. ``("properties", "address")``).
        root: The top-level schema dict, needed to resolve ``$ref``
            JSON pointers.

    Returns:
        The mutated *json_schema* dict.

    Raises:
        TypeError: If *json_schema* is not a ``dict``.
        UserError: If ``additionalProperties`` is ``True`` on an object
            node.
        ValueError: If a ``$ref`` cannot be resolved within *root*.
    """

    if not _is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dict; path={path}")

    # --- Process $defs and definitions -----------------------------------
    defs = json_schema.get("$defs")
    if _is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if _is_dict(definitions):
        for def_name, def_schema in definitions.items():
            _ensure_strict_schema(def_schema, path=(*path, "definitions", def_name), root=root)

    # --- additionalProperties --------------------------------------------
    _type = json_schema.get("type")
    if _type == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    elif _type == "object" and json_schema.get("additionalProperties") is True:
        raise UserError(
            "additionalProperties should not be set for object types. "
            "This could be because you're using an older version of Pydantic, "
            "or because you configured additional properties to be allowed. "
            "If you really need this, update the function or output tool "
            "to not use a strict schema."
        )

    # --- properties → all required ---------------------------------------
    properties = json_schema.get("properties")
    if _is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # --- items ------------------------------------------------------------
    items = json_schema.get("items")
    if _is_dict(items):
        json_schema["items"] = _ensure_strict_schema(items, path=(*path, "items"), root=root)

    # --- anyOf ------------------------------------------------------------
    any_of = json_schema.get("anyOf")
    if _is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_schema(variant, path=(*path, "anyOf", str(i)), root=root) for i, variant in enumerate(any_of)
        ]

    # --- oneOf → anyOf (strict mode does not support oneOf) ---------------
    one_of = json_schema.get("oneOf")
    if _is_list(one_of):
        existing_any_of = json_schema.get("anyOf", [])
        if not _is_list(existing_any_of):
            existing_any_of = []
        json_schema["anyOf"] = existing_any_of + [
            _ensure_strict_schema(variant, path=(*path, "oneOf", str(i)), root=root) for i, variant in enumerate(one_of)
        ]
        json_schema.pop("oneOf")

    # --- allOf (flatten single-element) -----------------------------------
    all_of = json_schema.get("allOf")
    if _is_list(all_of):
        if len(all_of) == 1:
            resolved = _ensure_strict_schema(all_of[0], path=(*path, "allOf", "0"), root=root)
            for k, v in resolved.items():
                json_schema.setdefault(k, v)
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_schema(entry, path=(*path, "allOf", str(i)), root=root) for i, entry in enumerate(all_of)
            ]

    # --- Remove default: null (not allowed in strict mode) ----------------
    if json_schema.get("default") is None and "default" in json_schema:
        json_schema.pop("default")

    # --- $ref with sibling keys → inline merge ----------------------------
    ref = json_schema.get("$ref")
    if ref and _has_more_than_n_keys(json_schema, 1):
        if not isinstance(ref, str):
            raise TypeError(f"Received non-string $ref: {ref!r}")

        resolved_ref = _resolve_ref(root=root, ref=ref)
        if not _is_dict(resolved_ref):
            raise ValueError(f"Expected `$ref: {ref}` to resolve to a dict but got {resolved_ref}")

        # Merge resolved ref into json_schema; existing keys take priority
        for key, value in resolved_ref.items():
            json_schema.setdefault(key, value)
        json_schema.pop("$ref")
        return _ensure_strict_schema(json_schema, path=path, root=root)

    return json_schema


def _resolve_ref(*, root: dict[str, object], ref: str) -> object:
    """Resolve a JSON pointer ``$ref`` within *root*.

    Follows a ``#/``-prefixed JSON pointer path (e.g. ``#/$defs/Foo``)
    through the *root* schema, returning the referenced sub-schema.

    Args:
        root: The top-level schema dictionary that contains the
            ``$defs`` or ``definitions`` being referenced.
        ref: A JSON pointer string.  Must start with ``#/``.

    Returns:
        The resolved sub-schema object found at the pointer location.

    Raises:
        ValueError: If *ref* does not start with ``#/``, or if the
            traversal encounters a non-dict entry before reaching the
            target.
        KeyError: If any segment of the pointer path does not exist as
            a key in the current dict during traversal.

    References:
        - JSON Pointer (RFC 6901): https://www.rfc-editor.org/rfc/rfc6901
        - JSON Schema ``$ref``:
          https://json-schema.org/understanding-json-schema/structuring#dollarref
    """
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; does not start with #/")

    path = ref[2:].split("/")
    resolved: object = root
    for key in path:
        if not _is_dict(resolved):
            raise ValueError(f"Encountered non-dict entry while resolving {ref} - {resolved}")
        resolved = resolved[key]

    return resolved
