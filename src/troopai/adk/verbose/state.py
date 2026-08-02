"""Block-state tree for the stateful Panel renderer.

The stateless :class:`~troopai.adk.verbose.renderer.VerboseRenderer` emits
a line per event and is done. The upcoming Panel backend needs to
compose panels out of paired ``*_start`` / ``*_end`` events, accumulate
payload between them, and still close cleanly when an exception
interrupts a block mid-flight. All of that state lives in a
:class:`RunTree` owned by the renderer.

The tree has **no** behaviour that depends on Rich, the line renderer,
or any ``VerboseConfig`` field. It is a pure data structure so it is
trivially unit-testable: open blocks, close blocks, assert verdicts,
confirm interrupted-block cleanup.

Key concepts
------------

* **Block** — an interval in the run bounded by a ``*_start`` event and
  its matching ``*_end`` (or cleanup-synthesised end). Example blocks:
  an agent turn, an LLM call, a single tool invocation, a guardrail
  pass, a HITL approval gate.
* **Identity key** — a tuple that identifies the block uniquely across
  concurrent runs. E.g. ``("agent", run_id, agent_id)`` for an agent
  block; ``("tool", run_id, tool_call_id)`` for a tool call. The tuple
  shape is chosen by the caller — the tree does not impose one.
* **Verdict** — a short string describing the close state: ``"ok"``,
  ``"error"``, ``"timeout"``, ``"rejected"``, ``"interrupted"``. The
  renderer maps verdicts to panel border colours.

The tree uses a stack-of-open-blocks model: every ``open`` pushes and
every ``close`` removes the matching block from the stack. Mis-ordered
closes (close A before B when B was opened after A) are tolerated by
searching the stack and removing *only* the matched block — blocks
opened after it stay open, stay reachable via :meth:`RunTree.find_open`,
and are swept by :meth:`RunTree.close_all` if their own close never
arrives. This keeps the tree robust against runners that open blocks in
a parent-child chain but close them in a different order (a runner bug
we don't want to crash the renderer over) without silently losing the
intervening blocks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


BlockKey = tuple[object, ...]
"""Identity tuple for a block. Chosen by the caller."""


@dataclass
class BlockNode:
    """One open-or-closed interval in the run.

    Attributes:
        event: Dotted event name of the opening event
            (e.g. ``"agent.start"``, ``"tool.start"``).
        key: Identity tuple chosen by the caller.
        headline: Short single-line header shown at the top of the
            block's panel (e.g. ``"agent · coordinator"``).
        payload: Accumulated payload lines emitted while the block is
            open. Kept as a list so the renderer can stream them as
            they arrive without recomputing layout.
        parent: Parent block (``None`` only for the tree root).
        children: Blocks opened while this one was the top of the
            stack. Used by the renderer for nested-panel indentation.
        started_at: ``time.monotonic()`` at open. Used to compute
            elapsed time on close.
        closed_at: ``time.monotonic()`` at close. ``None`` while open.
        verdict: Close state (see module docstring). ``None`` while
            open.
    """

    event: str
    key: BlockKey
    headline: str = ""
    payload: list[str] = field(default_factory=list)
    parent: BlockNode | None = None
    children: list[BlockNode] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    closed_at: float | None = None
    verdict: str | None = None

    def is_open(self) -> bool:
        """Return ``True`` when the block has not yet been closed.

        Returns:
            ``True`` while :attr:`closed_at` is ``None``; ``False``
            after the block is closed.
        """
        return self.closed_at is None

    def elapsed(self) -> float:
        """Return seconds between open and close.

        If the block is still open, returns seconds since open (so
        operators can surface a "running for 4.3s…" label mid-flight).

        Returns:
            Elapsed wall-clock seconds as a ``float``. Always
            non-negative.
        """
        end = self.closed_at if self.closed_at is not None else time.monotonic()
        return end - self.started_at

    def append_payload(self, line: str) -> None:
        """Append *line* to the block's payload buffer.

        Args:
            line: A single text line (may contain Rich markup) to
                accumulate in :attr:`payload`.
        """
        self.payload.append(line)


class RunTree:
    """Tree of open and closed blocks for one run.

    One instance per :class:`~troopai.adk.verbose.config.VerboseConfig`
    (so concurrent runs stay isolated — the Panel renderer owns its
    own tree; nothing is shared at class level). The tree tracks a
    stack of currently-open blocks so nested opens slot under the
    right parent automatically.
    """

    def __init__(self) -> None:
        self._root = BlockNode(event="__root__", key=())
        self._stack: list[BlockNode] = [self._root]

    # ----- open / close primitives ----------------------------------

    def open(
        self,
        event: str,
        key: BlockKey,
        headline: str = "",
    ) -> BlockNode:
        """Open a new block as a child of the current top of stack.

        Returns the new :class:`BlockNode` so the caller can append
        payload to it without looking it up again.

        Args:
            event: Dotted event name of the opening event
                (e.g. ``"agent.start"``).
            key: Identity tuple that will be used to match this block
                on a later :meth:`close` call.
            headline: Short single-line header for the block's panel.
                Defaults to an empty string.

        Returns:
            The newly-created :class:`BlockNode`, already pushed onto
            the stack as a child of the previous top.
        """
        parent = self._stack[-1]
        node = BlockNode(
            event=event,
            key=key,
            headline=headline,
            parent=parent,
        )
        parent.children.append(node)
        self._stack.append(node)
        return node

    def close(
        self,
        key: BlockKey,
        verdict: str = "ok",
    ) -> BlockNode | None:
        """Close the most recent open block matching *key*.

        Removes the matching block from the open stack. If no matching
        open block is found, logs a ``DEBUG`` message and returns
        ``None`` — a missing close is a runner bug but must not crash
        the renderer.

        Blocks opened *after* the matched one (an out-of-order close)
        are left untouched: they stay on the stack, stay open, and can
        still be closed by their own close event or swept by
        :meth:`close_all`. Popping them here would strand them —
        removed from the stack yet never marked closed, so no
        follow-up close or cleanup sweep could ever flush their
        panels.

        Args:
            key: Identity tuple identifying the block to close.
            verdict: Close state string (e.g. ``"ok"``, ``"error"``,
                ``"interrupted"``). Stored on the node and used by the
                renderer to choose a panel border colour.

        Returns:
            The closed :class:`BlockNode`, or ``None`` when no open
            block with *key* was found.
        """
        node = self.find_open(key)
        if node is None:
            logger.debug(
                "RunTree.close: no open block for key=%r (stale close?)",
                key,
            )
            return None
        node.closed_at = time.monotonic()
        node.verdict = verdict
        # Remove only the matched node. Index 0 is the sentinel root and
        # is never removed.
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index] is node:
                del self._stack[index]
                break
        return node

    def close_all(self, verdict: str = "interrupted") -> list[BlockNode]:
        """Close every still-open block with *verdict*.

        Called from the renderer's cleanup path (e.g. a ``try/finally``
        in ``PanelRenderer.__exit__``) to ensure a run interrupted by
        an exception doesn't leave panels visually "open". Returns the
        list of newly-closed blocks in close order (deepest first),
        so the caller can flush each one to the console.

        Args:
            verdict: Close state applied to every remaining open block.
                Defaults to ``"interrupted"``.

        Returns:
            List of the :class:`BlockNode` instances that were just
            closed, in unwind order (deepest / most-recently-opened
            first). Empty when the stack was already clean.
        """
        closed: list[BlockNode] = []
        now = time.monotonic()
        # Walk the stack top-down so nested blocks close before their
        # parents — matches the natural "unwind" order of panel output.
        while len(self._stack) > 1:
            node = self._stack.pop()
            node.closed_at = now
            node.verdict = verdict
            closed.append(node)
        return closed

    # ----- queries --------------------------------------------------

    def current(self) -> BlockNode | None:
        """Return the innermost currently-open block, or ``None``.

        Returns:
            The :class:`BlockNode` at the top of the stack, or
            ``None`` when only the sentinel root is present (no blocks
            open).
        """
        if len(self._stack) <= 1:
            return None
        return self._stack[-1]

    def depth(self) -> int:
        """Return the number of currently-open blocks (0 when empty).

        Returns:
            Non-negative integer; ``0`` when no blocks are open.
        """
        return len(self._stack) - 1

    def root(self) -> BlockNode:
        """Return the sentinel root (for tests + renderer traversal).

        Returns:
            The synthetic ``"__root__"`` :class:`BlockNode` at the
            bottom of the stack. Never ``None``; never ``is_open()``
            in the ``closed_at`` sense (it is never explicitly closed).
        """
        return self._root

    # ----- public lookup --------------------------------------------

    def find_open(self, key: BlockKey) -> BlockNode | None:
        """Walk the stack top→bottom, return the newest open block
        whose ``key`` equals *key*.

        Public so cross-module callers (the Panel renderer's
        ``append_payload`` / ``_find_open`` paths) no longer need to
        reach into the private ``_stack`` list.

        Args:
            key: Identity tuple to search for.

        Returns:
            The most recently opened :class:`BlockNode` whose
            :attr:`~BlockNode.key` equals *key* and that is still
            open, or ``None`` when no match is found.
        """
        for node in reversed(self._stack):
            if node.key == key and node.is_open() is True:
                return node
        return None
