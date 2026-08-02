"""Optional inline-``$ref`` resolver for MCP ``inputSchema``.

MCP servers may emit JSON Schemas that reference shared definitions
via intra-document ``$ref`` (``"#/$defs/Foo"``). Anthropic and
OpenAI Responses honour these natively; OpenAI Chat Completions and
some smaller providers reject them. This module inlines every
intra-document reference so the resulting schema is self-contained.

External-URI ``$ref`` (``"http://..."``, ``"file://..."``) is
intentionally NOT resolved — fetching schemas at runtime is a
network-side-effect we refuse to take on.

Cycle handling: a visited-set keyed by JSON-pointer string detects
cycles and surfaces them with a readable trail (``#/$defs/A →
#/$defs/B → #/$defs/A``). A separate depth backstop catches
non-cyclic but pathologically nested schemas.
"""

from __future__ import annotations

import logging
from typing import Any

from troopai.adk.mcp.exceptions import MCPSchemaConversionError

logger = logging.getLogger(__name__)

_MAX_DEPTH = 64
"""Hard cap on recursion depth for non-cyclic schemas. The visited
set catches every cycle; this catches genuinely deep nesting that
would otherwise consume Python's call stack. 64 is well below
Python's default ``sys.getrecursionlimit()`` and comfortably above
realistic JSON-Schema nesting (rarely > 8)."""


def inline_intra_document_refs(
    schema: dict[str, Any],
    *,
    tool_name: str,
) -> dict[str, Any]:
    """Return a new schema dict with intra-document ``$ref``s inlined.

    Args:
        schema: A JSON Schema dict, typically ``mcp.types.Tool.inputSchema``.
        tool_name: For diagnostics in raised exceptions.

    Returns:
        A deep-copied schema with every ``"#/<path>"`` reference
        replaced by the referenced subschema. The original ``$defs``
        / ``definitions`` blocks are removed once unreferenced.

    Raises:
        MCPSchemaConversionError: If a ``$ref`` cannot be resolved
            (broken pointer, external URI, recursion deeper than
            ``_MAX_DEPTH``, or a cycle).
    """
    if "$ref" not in _flatten(schema) and "$defs" not in schema and "definitions" not in schema:
        # Fast path — no refs anywhere; copy and return.
        copy: dict[str, Any] = _deep_copy(schema)
        return copy

    root = _deep_copy(schema)
    defs = _collect_defs(root)
    resolved = _resolve(
        root,
        root=root,
        defs=defs,
        depth=0,
        active_refs=(),
        tool_name=tool_name,
    )
    if isinstance(resolved, dict):
        # Drop the now-redundant $defs / definitions blocks.
        resolved.pop("$defs", None)
        resolved.pop("definitions", None)
    return resolved if isinstance(resolved, dict) else {"type": "object", "properties": {}}


def _resolve(
    node: Any,
    *,
    root: dict[str, Any],
    defs: dict[str, Any],
    depth: int,
    active_refs: tuple[str, ...],
    tool_name: str,
) -> Any:
    if depth > _MAX_DEPTH:
        raise MCPSchemaConversionError(f"MCP tool {tool_name!r} schema $ref recursion exceeded {_MAX_DEPTH}")
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in active_refs:
                trail = " → ".join((*active_refs, ref))
                raise MCPSchemaConversionError(f"MCP tool {tool_name!r} schema $ref cycle detected: {trail}")
            target = _follow_pointer(ref, root=root, defs=defs, tool_name=tool_name)
            inlined = _resolve(
                target,
                root=root,
                defs=defs,
                depth=depth + 1,
                active_refs=(*active_refs, ref),
                tool_name=tool_name,
            )
            # Merge sibling keys from the original ref-bearing node so
            # callers like ``{"$ref": "#/$defs/X", "description": "..."}``
            # keep the description.
            if isinstance(inlined, dict):
                merged: dict[str, Any] = dict(inlined)
                for k, v in node.items():
                    if k == "$ref":
                        continue
                    merged[k] = _resolve(
                        v,
                        root=root,
                        defs=defs,
                        depth=depth + 1,
                        active_refs=active_refs,
                        tool_name=tool_name,
                    )
                return merged
            return inlined
        return {
            k: _resolve(
                v,
                root=root,
                defs=defs,
                depth=depth + 1,
                active_refs=active_refs,
                tool_name=tool_name,
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [
            _resolve(
                item,
                root=root,
                defs=defs,
                depth=depth + 1,
                active_refs=active_refs,
                tool_name=tool_name,
            )
            for item in node
        ]
    return node


def _follow_pointer(
    pointer: str,
    *,
    root: dict[str, Any],
    defs: dict[str, Any],
    tool_name: str,
) -> Any:
    """Resolve a JSON Pointer per RFC 6901.

    Strict path-walk: every token must exist at the corresponding
    nesting level. The Draft-07 ``definitions`` vs ``$defs`` spelling
    is handled at the *first* token only — a pointer rooted at one
    spelling is rewritten to the other when only the alternative
    block exists. Per-step fallbacks (which silently let
    ``#/foo/A`` find ``defs["A"]``) are NOT supported.

    Args:
        pointer: A JSON Pointer string starting with ``"#/"``; external
            URIs raise immediately.
        root: The top-level schema dict used as the resolution root.
        defs: The merged ``$defs`` / ``definitions`` block from
            ``_collect_defs``.
        tool_name: For diagnostics in raised exceptions.

    Returns:
        The subschema or value at the pointer's location.

    Raises:
        MCPSchemaConversionError: If the pointer is external, the path
            cannot be resolved, or the ``$defs`` / ``definitions``
            spelling swap fails.
    """
    if not pointer.startswith("#/"):
        raise MCPSchemaConversionError(
            f"MCP tool {tool_name!r} uses external $ref {pointer!r}; "
            "external references are not resolved by the inline resolver."
        )
    parts = pointer[2:].split("/") if len(pointer) > 2 else []
    parts = [p.replace("~1", "/").replace("~0", "~") for p in parts]
    parts = _canonicalise_defs_root(parts, root=root, defs=defs, pointer=pointer, tool_name=tool_name)

    node: Any = root
    for token in parts:
        if isinstance(node, dict) and token in node:
            node = node[token]
            continue
        raise MCPSchemaConversionError(f"MCP tool {tool_name!r} schema $ref {pointer!r} not resolvable")
    return node


def _canonicalise_defs_root(
    parts: list[str],
    *,
    root: dict[str, Any],
    defs: dict[str, Any],
    pointer: str,
    tool_name: str,
) -> list[str]:
    """Rewrite the leading ``$defs`` / ``definitions`` token if needed.

    A pointer like ``#/$defs/A`` resolves cleanly when ``root`` has
    ``$defs``. If the schema author used the Draft-07 ``definitions``
    spelling instead, swap the first token so the path-walk
    succeeds. Supports the reverse swap as well.

    Args:
        parts: The split pointer tokens (RFC 6901 decoded) excluding
            the leading ``"#"`` sentinel.
        root: The top-level schema dict.
        defs: The merged definitions block from ``_collect_defs``.
        pointer: The original pointer string (for error messages).
        tool_name: For diagnostics in raised exceptions.

    Returns:
        The (possibly rewritten) token list with the leading defs key
        normalised to the spelling present in ``root``.

    Raises:
        MCPSchemaConversionError: If the defs-root token is one of
            the known spellings but neither exists in ``root``.
    """
    if len(parts) == 0:
        return parts
    head = parts[0]
    if head not in {"$defs", "definitions"}:
        return parts
    if head in root:
        return parts
    other = "definitions" if head == "$defs" else "$defs"
    if isinstance(root.get(other), dict) and parts[1:] and parts[1] in defs:
        return [other, *parts[1:]]
    raise MCPSchemaConversionError(f"MCP tool {tool_name!r} schema $ref {pointer!r} not resolvable")


def _collect_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Merge ``$defs`` and ``definitions`` blocks into a single dict.

    Args:
        schema: The top-level JSON Schema dict.

    Returns:
        A dict containing every definition from both ``$defs`` and
        ``definitions``. Values from ``definitions`` overwrite
        same-named values from ``$defs`` when both blocks define the
        same key.
    """
    defs: dict[str, Any] = {}
    block = schema.get("$defs")
    if isinstance(block, dict):
        defs.update(block)
    block = schema.get("definitions")
    if isinstance(block, dict):
        defs.update(block)
    return defs


def _flatten(node: Any) -> list[str]:
    """Return every string leaf in ``node`` for fast ``$ref`` detection.

    Args:
        node: Any JSON-compatible value (dict, list, str, or scalar).

    Returns:
        A flat list of all string keys and string values found anywhere
        in the nested structure.
    """
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.append(k)
            out.extend(_flatten(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(_flatten(item))
    elif isinstance(node, str):
        out.append(node)
    return out


def _deep_copy(node: Any) -> Any:
    """Recursively deep-copy a JSON-compatible value.

    Args:
        node: Any JSON-compatible value (dict, list, or scalar).

    Returns:
        A fully independent deep copy of ``node``.
    """
    if isinstance(node, dict):
        return {k: _deep_copy(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_deep_copy(item) for item in node]
    return node
