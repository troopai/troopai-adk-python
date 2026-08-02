"""Internal helpers shared by the Mermaid and DOT emitters.

Module-private utilities: not exported from
:mod:`troopai.adk.visualization` (omitted from ``__init__.py``'s
``__all__``). Co-locating the shared logic here keeps both emitters
DRY without underscore-aliased imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from troopai.adk.flows.flow_wrappers import FlowStep

if TYPE_CHECKING:
    from troopai.adk.flows.flow import Flow


def build_step_lookup(flow: Flow) -> dict[str, FlowStep]:
    """Return a mapping from step name to its :class:`FlowStep` descriptor.

    Walks ``type(flow).__dict__`` ONLY — matches the contract of
    :class:`troopai.adk.flows.flow.FlowMeta`, which intentionally
    ignores inherited decorated methods. Looking past ``__dict__``
    would silently surface descriptions from parent classes whose
    decorated methods the registry does NOT consider valid steps,
    producing diagrams that disagree with the actual run topology.

    Args:
        flow: The flow instance whose class is being inspected.

    Returns:
        Mapping from method name to the unbound :class:`FlowStep`.
        Empty when no decorated methods are declared on the class
        body (the abstract :class:`Flow` base case).
    """
    lookup: dict[str, FlowStep] = {}
    for attr_name, attr_value in type(flow).__dict__.items():
        if isinstance(attr_value, FlowStep):
            lookup[attr_name] = attr_value
    return lookup


def assert_no_collision(
    forward: dict[str, str],
    original: str,
    sanitised: str,
    inverse: dict[str, str] | None = None,
) -> None:
    """Raise ``ValueError`` when sanitisation collapses two ids into one.

    The Mermaid / DOT emitters sanitise step / route / node names into
    a safe identifier subset. Two different originals (e.g. ``a-b``
    and ``a_b``) that both sanitise to ``a_b`` would silently merge
    into a single diagram node and retarget edges between them. This
    helper detects that case and raises with both colliding names so
    the developer can rename one of them.

    Also raises when the SAME original is re-registered with a
    DIFFERENT sanitised id — that would indicate a non-deterministic
    sanitiser, a contract bug rather than a user-facing issue, but
    the explicit raise still beats silent overwrite.

    Performance: when *inverse* is supplied (a mutable mapping from
    sanitised → original maintained alongside *forward* by the caller),
    collision detection is O(1).  When *inverse* is ``None`` the
    function falls back to a linear scan of *forward* (O(N)) — suitable
    for topologies under ~1k nodes.

    Args:
        forward: Mutable mapping from original → sanitised id. The
            caller passes an accumulator; this helper updates it on
            first sight and raises on second-sight collision.
        original: The original name before sanitisation.
        sanitised: The sanitised id.
        inverse: Optional mutable mapping from sanitised → original.
            When supplied it is kept in sync with *forward* and used
            for O(1) collision detection.  Pass the same dict on every
            call within a single diagram-emission pass.

    Raises:
        ValueError: When ``sanitised`` is already in the inverse view
            of ``forward`` under a different ``original``, OR when
            ``original`` was previously registered under a different
            ``sanitised`` (sanitiser non-determinism).
    """
    existing = forward.get(original)
    if existing is not None and existing != sanitised:
        raise ValueError(
            f"Sanitiser produced two different ids for {original!r}: "
            f"{existing!r} (previously) and {sanitised!r} (now). "
            f"This indicates a non-deterministic sanitiser — please report."
        )
    if existing == sanitised:
        return
    if inverse is not None:
        # O(1) path: consult the caller-maintained inverse index.
        prev_original = inverse.get(sanitised)
        if prev_original is not None and prev_original != original:
            raise ValueError(
                f"Sanitised identifier {sanitised!r} collides between "
                f"{prev_original!r} and {original!r}. Rename one of them to "
                f"avoid silently merging two diagram nodes."
            )
        forward[original] = sanitised
        inverse[sanitised] = original
    else:
        # O(N) fallback: walk the forward dict to find any collision.
        for prev_original, prev_sanitised in forward.items():
            if prev_sanitised == sanitised and prev_original != original:
                raise ValueError(
                    f"Sanitised identifier {sanitised!r} collides between "
                    f"{prev_original!r} and {original!r}. Rename one of them to "
                    f"avoid silently merging two diagram nodes."
                )
        forward[original] = sanitised


def safe(name: str) -> str:
    """Return a Mermaid / DOT-safe identifier derived from ``name``.

    Args:
        name: Arbitrary string (typically a method name or a route label).

    Returns:
        Alphanumeric-and-underscore identifier; never empty.
    """
    if len(name) == 0:
        return "_"
    safe_chars = [c if (c.isalnum() or c == "_") else "_" for c in name]
    return "".join(safe_chars)


def escape_label(label: str) -> str:
    """Escape characters that break a Mermaid / DOT quoted label.

    Replaces backslashes (first, so subsequent replacements don't double-escape),
    double quotes, and newlines so the surrounding ``"..."`` parses.

    Args:
        label: Raw label text.

    Returns:
        Escaped label safe for placement inside double quotes.
    """
    return label.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def escape_mermaid_label(label: str) -> str:
    """Escape characters that break a Mermaid quoted label.

    Unlike DOT (see :func:`escape_label`), Mermaid quoted strings do NOT
    honour backslash escapes — a literal ``\\"`` renders verbatim instead
    of producing a quote and can leave the string unterminated. Mermaid
    instead recognises HTML entity codes written with a leading ``#``
    (``#quot;`` for ``"``, ``#35;`` for ``#``). This rewrites the
    characters that would otherwise terminate the string or start a
    spurious entity, and turns newlines into Mermaid ``<br>`` line breaks.

    ``#`` is escaped first so the ``#`` introduced by the later
    replacements is not itself re-encoded.

    Args:
        label: Raw label text.

    Returns:
        Label text safe to place inside a Mermaid ``"..."`` string.
    """
    return label.replace("#", "#35;").replace('"', "#quot;").replace("\n", "<br>")


def node_label_from_desc(name: str, desc_lookup: dict[str, str | None]) -> str:
    """Return the display label for a step node from a description lookup.

    Uses an explicit ``is not None`` check so an empty-string description
    ``""`` is preserved as the label (matching the semantics of
    :func:`build_step_lookup`-based helpers that use
    ``step.description is not None``). Falls back to ``name`` only when the
    description is ``None``.

    Args:
        name: Step method name — used as the fallback label.
        desc_lookup: Mapping from step name to optional description string.
            Typically built from :class:`~troopai.adk.flows.definition.StepInfo`
            instances.

    Returns:
        The description string when explicitly set (even ``""``), or
        ``name`` when ``None``.
    """
    raw = desc_lookup.get(name)
    return raw if raw is not None else name


def gate_node_id(gate_id: str) -> str:
    """Convert a canonical gate id into an alphanumeric-and-underscore slug.

    :attr:`GateSpec.gate_id` is of the form ``"<listener>:<kind>:<csv>"``.
    The slug replaces ``:`` with ``__`` and ``,`` with ``_`` so the
    resulting id is safe as a Mermaid node id and a DOT id.

    Args:
        gate_id: Canonical id from :attr:`GateSpec.gate_id`.

    Returns:
        Mermaid / DOT-safe identifier prefixed with ``gate__``.
    """
    return "gate__" + gate_id.replace(":", "__").replace(",", "_")
