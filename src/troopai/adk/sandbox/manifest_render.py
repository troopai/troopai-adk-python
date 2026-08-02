"""Render a manifest as an ASCII filesystem tree for system prompts.

A ``SandboxAgent`` ships the workspace layout to the LLM by
rendering the manifest as a ``tree``-style listing. Entries are
emitted in dictionary order; descriptions hang off the right in
``# comment`` form so the model can grep for them. The result
gets truncated at ``MAX_MANIFEST_DESCRIPTION_CHARS`` to keep the
system prompt from runaway growth.
"""

from __future__ import annotations

import logging
import posixpath
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from troopai.adk.types.sandbox.entries import BaseEntry, Dir
from troopai.adk.types.sandbox.mounts import Mount

__all__ = [
    "MANIFEST_DESCRIPTION_TRUNCATION_MARKER_TEMPLATE",
    "MAX_MANIFEST_DESCRIPTION_CHARS",
    "render_manifest_description",
]

logger = logging.getLogger(__name__)

MAX_MANIFEST_DESCRIPTION_CHARS: int = 5000
"""Default ceiling on the rendered manifest description length."""

MANIFEST_DESCRIPTION_TRUNCATION_MARKER_TEMPLATE: str = "... (truncated {omitted_chars} chars)"
"""Format string used to render the truncation marker."""


def _coerce_posix_path(path: str | Path) -> PurePosixPath:
    """Return a POSIX-flavored path regardless of host OS."""
    if isinstance(path, Path):
        return PurePosixPath(path.as_posix())
    return PurePosixPath(path.replace("\\", "/"))


def _posix_path_as_path(path: PurePosixPath) -> Path:
    return Path(path.as_posix())


def _truncate_manifest_description(description: str, max_chars: int | None) -> str:
    if max_chars is None or len(description) <= max_chars:
        return description
    if max_chars <= 0:
        return ""

    omitted_chars = len(description) - max_chars
    keep_chars = 0
    marker = ""
    # Iteratively widen the omission count until the marker length stops
    # shifting — the marker text itself contains a digit count, so the
    # first estimate may be off by one or two characters. ``omitted_chars``
    # grows at most logarithmically (digit count of ``len(description)``),
    # so 8 iterations is far more than required; the cap defends against a
    # future format-template change introducing a wider variable-width
    # specifier (R2: bounded loops only).
    for _ in range(8):
        marker = (
            "\n"
            + MANIFEST_DESCRIPTION_TRUNCATION_MARKER_TEMPLATE.format(omitted_chars=omitted_chars)
            + "\n\nThe filesystem layout above was truncated. "
            "Use `ls` to explore specific directories before relying on omitted paths.\n"
        )
        keep_chars = max(0, max_chars - len(marker))
        actual_omitted_chars = len(description) - keep_chars
        if actual_omitted_chars == omitted_chars:
            break
        omitted_chars = actual_omitted_chars
    else:
        logger.warning("Manifest truncation marker did not converge in 8 iterations; using last estimate.")

    truncated = description[:keep_chars].rstrip() + marker
    if len(marker) >= max_chars:
        truncated = marker[:max_chars]
        logger.warning(
            "Manifest description exceeded %d characters and was truncated to %d characters.",
            max_chars,
            len(truncated),
        )
        return truncated
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    logger.warning(
        "Manifest description exceeded %d characters and was truncated to %d characters.",
        max_chars,
        len(truncated),
    )
    return truncated


def _path_parts(path: Path) -> tuple[str, ...]:
    parts = [part for part in _coerce_posix_path(path).parts if part not in {"", "."}]
    return tuple(parts)


class _Node:
    """One vertex of the tree the renderer assembles in-memory."""

    __slots__ = ("children", "description", "full_path", "is_dir")

    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.description: str | None = None
        self.is_dir: bool = False
        self.full_path: Path | None = None


def _mount_full_path(
    entry: str | Path,
    artifact: Mount,
    *,
    root_path: Path,
    coerce_rel_path: Callable[[str | Path], Path],
) -> Path:
    if artifact.mount_path is not None:
        mount_path = _coerce_posix_path(artifact.mount_path)
        return _posix_path_as_path(
            mount_path if mount_path.is_absolute() else _coerce_posix_path(root_path) / mount_path
        )
    return root_path / coerce_rel_path(entry)


def _insert_path(
    root_node: _Node,
    path: Path,
    *,
    description: str | None,
    is_dir: bool,
    full_path: Path | None = None,
    max_depth: int | None = None,
) -> None:
    parts = _path_parts(path)
    if len(parts) == 0:
        return
    node = root_node
    limit = len(parts) if max_depth is None else min(len(parts), max_depth)
    for index, part in enumerate(parts[:limit]):
        node = node.children.setdefault(part, _Node())
        if index < len(parts) - 1:
            node.is_dir = True
    if node.description is None and description is not None and limit == len(parts):
        node.description = description
    if full_path is not None and limit == len(parts):
        node.full_path = full_path
    if is_dir or limit < len(parts):
        node.is_dir = True


def _insert_entry_tree(
    root_node: _Node,
    path: Path,
    artifact: BaseEntry,
    *,
    full_path: Path | None,
    depth: int | None,
    coerce_rel_path: Callable[[str | Path], Path],
) -> None:
    stack: list[tuple[Path, BaseEntry, Path | None]] = [(path, artifact, full_path)]
    while len(stack) > 0:
        current_path, current_artifact, current_full_path = stack.pop()
        _insert_path(
            root_node,
            current_path,
            description=current_artifact.description,
            is_dir=current_artifact.permissions.directory,
            full_path=current_full_path,
            max_depth=depth,
        )
        if not isinstance(current_artifact, Dir):
            continue
        if depth is not None and len(_path_parts(current_path)) >= depth:
            continue

        for child_name, child_artifact in current_artifact.children.items():
            child_rel_path = coerce_rel_path(child_name)
            child_path = current_path / child_rel_path
            child_full_path = current_full_path / child_rel_path if current_full_path is not None else None
            stack.append((child_path, child_artifact, child_full_path))


def _collect(
    root_node: _Node,
    *,
    root_path: Path,
    initial_remaining: int | None,
) -> list[tuple[str, str, str, str | None]]:
    lines: list[tuple[str, str, str, str | None]] = []
    stack: list[tuple[str, _Node, str, int | None, tuple[str, ...]]] = [
        ("children", root_node, "", initial_remaining, ()),
    ]
    while len(stack) > 0:
        action, current_node, current_prefix, current_remaining, current_rel_parts = stack.pop()
        if action == "line":
            child = current_node
            name = current_rel_parts[-1]
            child_is_dir = child.is_dir or len(child.children) > 0
            display_name = f"{name}/" if child_is_dir else name
            if child.full_path is not None:
                full_path = child.full_path.as_posix()
            else:
                full_path = (_coerce_posix_path(root_path) / _coerce_posix_path("/".join(current_rel_parts))).as_posix()
            lines.append((current_prefix, display_name, full_path, child.description))
            continue

        if current_remaining is not None and current_remaining <= 0:
            continue

        names = sorted(current_node.children)
        next_remaining = None if current_remaining is None else current_remaining - 1
        for index in range(len(names) - 1, -1, -1):
            name = names[index]
            child = current_node.children[name]
            is_last = index == len(names) - 1
            connector = "└── " if is_last else "├── "
            child_parts = (*current_rel_parts, name)
            if next_remaining is None or next_remaining > 0:
                extension = "    " if is_last else "│   "
                stack.append(("children", child, current_prefix + extension, next_remaining, child_parts))
            stack.append(("line", child, current_prefix + connector, next_remaining, child_parts))
    return lines


def render_manifest_description(
    *,
    root: str,
    entries: dict[str | Path, BaseEntry],
    coerce_rel_path: Callable[[str | Path], Path],
    depth: int | None = 1,
    max_chars: int | None = MAX_MANIFEST_DESCRIPTION_CHARS,
) -> str:
    """Render ``entries`` as a tree rooted at ``root``.

    Args:
        root: Workspace root path (rendered as the tree's root label).
        entries: Manifest entries keyed by workspace-relative path.
        coerce_rel_path: Function turning an entry-key string/path into
            a relative ``Path`` (manifest validation provides this).
        depth: Maximum tree depth, ``None`` for unbounded. Must be
            strictly positive when provided.
        max_chars: Truncation ceiling, ``None`` for unbounded.

    Returns:
        Newline-terminated tree description, possibly with the
        truncation marker appended.
    """
    if depth is not None and depth <= 0:
        raise ValueError("render_manifest_description: depth must be a positive integer or None")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("render_manifest_description: max_chars must be a positive integer or None")

    root_str = root.rstrip("/") if len(root) > 1 else "/"
    if root_str == "":
        root_str = "/"
    root_path = _posix_path_as_path(_coerce_posix_path(root_str))

    root_node = _Node()
    for entry, artifact in entries.items():
        path = coerce_rel_path(entry)
        if path.is_absolute():
            path = path.relative_to(path.anchor)
        full_path: Path | None = (
            _mount_full_path(entry, artifact, root_path=root_path, coerce_rel_path=coerce_rel_path)
            if isinstance(artifact, Mount)
            else None
        )
        _insert_entry_tree(
            root_node,
            path,
            artifact,
            full_path=full_path,
            depth=depth,
            coerce_rel_path=coerce_rel_path,
        )

    lines: list[str] = [root_str]
    collected = _collect(root_node, root_path=root_path, initial_remaining=depth)
    if len(collected) > 0:
        max_width = max(len(prefix + name) for prefix, name, _, _ in collected)
        for prefix, name, full_path_str, description in collected:
            spacer = " " * (max_width - len(prefix + name) + 2)
            comment = f"# {full_path_str} — {description}" if description is not None else f"# {full_path_str}"
            lines.append(f"{prefix}{name}{spacer}{comment}")

    description = "\n".join(lines) + "\n"
    return _truncate_manifest_description(description, max_chars)


def coerce_rel_path(value: str | Path) -> Path:
    """Default relative-path coercion used when the caller has no manifest-specific one.

    Strips a leading ``"/"`` and a leading drive anchor so the result
    is always workspace-relative. Used by the runtime when the
    framework doesn't have access to ``Manifest._coerce_rel_path``.
    """
    raw = value.as_posix() if isinstance(value, Path) else value.replace("\\", "/")
    # Strip a Windows drive prefix ("C:/foo" -> "foo") before normalizing so
    # the drive anchor never survives as a leading path part.
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        raw = raw[2:].lstrip("/")
    normalized = posixpath.normpath(raw)
    if normalized == ".":
        return Path(".")
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    # Drop leading parent refs ("../../etc" -> "etc") so the result is always
    # workspace-relative; ``posixpath.normpath`` preserves them otherwise.
    parts = [part for part in normalized.split("/") if part != ".."]
    normalized = "/".join(parts)
    return Path(normalized) if len(normalized) > 0 else Path(".")
