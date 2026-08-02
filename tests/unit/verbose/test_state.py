"""Tests for :mod:`troopai.adk.verbose.state`.

Covers the :class:`BlockNode` container and :class:`RunTree` stack-based
block tracker used by the Panel renderer. The tree has no Rich / render
dependencies, so these tests are pure data-structure assertions —
open / close / cleanup / stale-close tolerance.
"""

from __future__ import annotations

import logging
import time

import pytest

from troopai.adk.verbose.state import BlockNode, RunTree

# ---------------------------------------------------------------------------
# BlockNode
# ---------------------------------------------------------------------------


class TestBlockNode:
    def test_defaults_open(self) -> None:
        """A freshly-built node is open and has no verdict."""
        node = BlockNode(event="agent.start", key=("agent", "r1"))
        assert node.is_open() is True
        assert node.closed_at is None
        assert node.verdict is None
        assert len(node.payload) == 0
        assert len(node.children) == 0

    def test_is_open_false_after_close(self) -> None:
        node = BlockNode(event="agent.start", key=("agent", "r1"))
        node.closed_at = time.monotonic()
        assert node.is_open() is False

    def test_elapsed_while_open(self) -> None:
        """``elapsed()`` on an open block returns time since open."""
        node = BlockNode(event="tool.start", key=("tool", "t1"))
        node.started_at = time.monotonic() - 0.05
        elapsed = node.elapsed()
        assert elapsed >= 0.05

    def test_elapsed_after_close(self) -> None:
        """``elapsed()`` on a closed block returns the recorded interval."""
        node = BlockNode(event="tool.start", key=("tool", "t1"))
        node.started_at = 100.0
        node.closed_at = 100.5
        assert node.elapsed() == pytest.approx(0.5)

    def test_append_payload_preserves_order(self) -> None:
        node = BlockNode(event="llm.start", key=("llm", "c1"))
        node.append_payload("first")
        node.append_payload("second")
        node.append_payload("third")
        assert node.payload == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# RunTree — basic open/close
# ---------------------------------------------------------------------------


class TestRunTreeBasics:
    def test_empty_tree_has_no_current(self) -> None:
        tree = RunTree()
        assert tree.current() is None
        assert tree.depth() == 0

    def test_root_is_sentinel(self) -> None:
        tree = RunTree()
        root = tree.root()
        assert root.event == "__root__"
        assert root.key == ()
        assert root.parent is None

    def test_open_returns_new_node(self) -> None:
        tree = RunTree()
        node = tree.open(
            event="agent.start",
            key=("agent", "r1", "coord"),
            headline="agent · coord",
        )
        assert node.event == "agent.start"
        assert node.key == ("agent", "r1", "coord")
        assert node.headline == "agent · coord"
        assert node.is_open() is True

    def test_open_updates_depth_and_current(self) -> None:
        tree = RunTree()
        assert tree.depth() == 0
        node = tree.open("agent.start", ("agent", "r1"))
        assert tree.depth() == 1
        assert tree.current() is node

    def test_open_attaches_to_root(self) -> None:
        tree = RunTree()
        node = tree.open("agent.start", ("agent", "r1"))
        assert node.parent is tree.root()
        assert node in tree.root().children

    def test_close_matching_block_returns_node(self) -> None:
        tree = RunTree()
        opened = tree.open("agent.start", ("agent", "r1"))
        closed = tree.close(("agent", "r1"), verdict="ok")
        assert closed is opened
        assert closed is not None
        assert closed.is_open() is False
        assert closed.verdict == "ok"
        assert closed.closed_at is not None

    def test_close_pops_stack(self) -> None:
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        assert tree.depth() == 1
        tree.close(("agent", "r1"))
        assert tree.depth() == 0
        assert tree.current() is None

    def test_default_verdict_is_ok(self) -> None:
        tree = RunTree()
        tree.open("tool.start", ("tool", "t1"))
        node = tree.close(("tool", "t1"))
        assert node is not None
        assert node.verdict == "ok"

    def test_custom_verdicts(self) -> None:
        tree = RunTree()
        for verdict in ("error", "timeout", "rejected", "interrupted"):
            tree.open("tool.start", ("tool", verdict))
            node = tree.close(("tool", verdict), verdict=verdict)
            assert node is not None
            assert node.verdict == verdict


# ---------------------------------------------------------------------------
# RunTree — nested blocks
# ---------------------------------------------------------------------------


class TestRunTreeNesting:
    def test_nested_open_uses_current_as_parent(self) -> None:
        """An ``open`` while another block is on the stack attaches as child."""
        tree = RunTree()
        outer = tree.open("agent.start", ("agent", "r1"))
        inner = tree.open("tool.start", ("tool", "t1"))
        assert inner.parent is outer
        assert inner in outer.children

    def test_nested_depth(self) -> None:
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        tree.open("llm.start", ("llm", "c1"))
        tree.open("tool.start", ("tool", "t1"))
        assert tree.depth() == 3

    def test_nested_close_in_order(self) -> None:
        """LIFO close order: innermost first, outermost last."""
        tree = RunTree()
        outer = tree.open("agent.start", ("agent", "r1"))
        inner = tree.open("tool.start", ("tool", "t1"))
        assert tree.close(("tool", "t1")) is inner
        assert tree.depth() == 1
        assert tree.current() is outer
        assert tree.close(("agent", "r1")) is outer
        assert tree.depth() == 0


# ---------------------------------------------------------------------------
# RunTree — stale / missing / mis-ordered closes
# ---------------------------------------------------------------------------


class TestRunTreeTolerance:
    def test_close_missing_key_returns_none(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A close that matches no open block returns None + logs DEBUG."""
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        with caplog.at_level(logging.DEBUG, logger="troopai.adk.verbose.state"):
            result = tree.close(("tool", "nonexistent"))
        assert result is None
        assert tree.depth() == 1  # stack unchanged
        assert any("no open block for key" in record.message for record in caplog.records)

    def test_close_on_empty_tree_returns_none(self) -> None:
        tree = RunTree()
        assert tree.close(("anything",)) is None
        assert tree.depth() == 0

    def test_mis_ordered_close_keeps_intervening_blocks_open(self) -> None:
        """Closing outer before inner removes only outer from the stack.

        If the runner closes A before B when B was opened inside A, the
        tree must not strand B: B stays on the stack (still open, still
        findable) so its own close — or a ``close_all`` sweep — can
        still close and flush it. The old behaviour popped B off the
        stack without closing it, leaving a node that no follow-up
        close or cleanup sweep could ever reach.
        """
        tree = RunTree()
        outer = tree.open("agent.start", ("agent", "r1"))
        inner = tree.open("tool.start", ("tool", "t1"))
        # Close outer while inner is still on top of the stack.
        closed = tree.close(("agent", "r1"))
        assert closed is outer
        # inner survives: still open, still on the stack, still findable.
        assert tree.depth() == 1
        assert inner.is_open() is True
        assert tree.find_open(("tool", "t1")) is inner
        assert inner in outer.children
        # Its own close still works after the out-of-order outer close.
        assert tree.close(("tool", "t1"), verdict="ok") is inner
        assert tree.depth() == 0

    def test_mis_ordered_close_leftovers_swept_by_close_all(self) -> None:
        """A block skipped by an out-of-order close is swept by close_all."""
        tree = RunTree()
        tree.open("turn.start", ("turn", 1))
        inner = tree.open("hitl.approval.requested", ("hitl", "c1"))
        tree.close(("turn", 1), verdict="ok")
        swept = tree.close_all(verdict="interrupted")
        assert any(node is inner for node in swept)
        assert inner.is_open() is False
        assert inner.verdict == "interrupted"


# ---------------------------------------------------------------------------
# RunTree.close_all
# ---------------------------------------------------------------------------


class TestRunTreeCloseAll:
    def test_close_all_on_empty_tree(self) -> None:
        tree = RunTree()
        closed = tree.close_all()
        assert closed == []
        assert tree.depth() == 0

    def test_close_all_marks_interrupted_by_default(self) -> None:
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        tree.open("tool.start", ("tool", "t1"))
        closed = tree.close_all()
        assert len(closed) == 2
        for node in closed:
            assert node.verdict == "interrupted"
            assert node.closed_at is not None

    def test_close_all_custom_verdict(self) -> None:
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        tree.open("tool.start", ("tool", "t1"))
        closed = tree.close_all(verdict="error")
        assert all(node.verdict == "error" for node in closed)

    def test_close_all_returns_deepest_first(self) -> None:
        """Unwind order matches natural panel-output cleanup order."""
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        tree.open("llm.start", ("llm", "c1"))
        tree.open("tool.start", ("tool", "t1"))
        closed = tree.close_all()
        # Deepest (tool) first, then llm, then agent.
        assert [node.event for node in closed] == [
            "tool.start",
            "llm.start",
            "agent.start",
        ]

    def test_close_all_resets_depth(self) -> None:
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        tree.open("tool.start", ("tool", "t1"))
        tree.close_all()
        assert tree.depth() == 0
        assert tree.current() is None

    def test_close_all_assigns_single_timestamp(self) -> None:
        """All blocks closed in one cleanup share the same ``closed_at``."""
        tree = RunTree()
        tree.open("agent.start", ("agent", "r1"))
        tree.open("tool.start", ("tool", "t1"))
        closed = tree.close_all()
        timestamps = {node.closed_at for node in closed}
        assert len(timestamps) == 1

    def test_close_all_preserves_tree_structure(self) -> None:
        """Closed nodes stay in their parent's children after close_all."""
        tree = RunTree()
        outer = tree.open("agent.start", ("agent", "r1"))
        inner = tree.open("tool.start", ("tool", "t1"))
        tree.close_all()
        assert inner in outer.children
        assert outer in tree.root().children
