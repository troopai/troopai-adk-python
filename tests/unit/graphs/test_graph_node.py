"""Tests for ``GraphNode`` and ``GraphEdge`` structural primitives.

Covers:
- Node-id pattern acceptance and rejection.
- ``GraphEdge`` self-loop rejection.
- Default values for ``merge`` and ``join`` on ``GraphNode``.
"""

from __future__ import annotations

import pytest

from troopai.adk.graphs.join import JoinSemantics
from troopai.adk.graphs.merge import DEFAULT_MERGE
from troopai.adk.graphs.node import GraphEdge, GraphNode
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult

# ---------------------------------------------------------------------------
# Minimal executable stub for constructing GraphNode instances
# ---------------------------------------------------------------------------


class _StubExecutable:
    """Minimal sync-like stub — satisfies ``executable is not None`` check."""

    async def invoke(
        self,
        _input: ExecutableInput,
        _context: object,
        _config: object,
    ) -> NodeResult:
        return NodeResult(output="stub")


_STUB = _StubExecutable()


# ---------------------------------------------------------------------------
# GraphNode id validation
# ---------------------------------------------------------------------------


class TestGraphNodeId:
    def _make(self, node_id: str) -> GraphNode:
        return GraphNode(id=node_id, executable=_STUB)  # type: ignore[arg-type]

    # --- Accepted ids ---

    def test_single_alpha(self) -> None:
        n = self._make("a")
        assert n.id == "a"

    def test_alphanumeric(self) -> None:
        n = self._make("a1")
        assert n.id == "a1"

    def test_hyphenated(self) -> None:
        n = self._make("a-1")
        assert n.id == "a-1"

    def test_underscored(self) -> None:
        n = self._make("a_b")
        assert n.id == "a_b"

    def test_uppercase(self) -> None:
        n = self._make("X123")
        assert n.id == "X123"

    def test_long_alphanumeric(self) -> None:
        n = self._make("LegalReviewNode01")
        assert n.id == "LegalReviewNode01"

    # --- Rejected ids ---

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("")

    def test_leading_hyphen_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("-start")

    def test_leading_underscore_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("_start")

    def test_whitespace_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("a b")

    def test_dot_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("a.b")

    def test_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("a/b")

    def test_at_sign_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            self._make("@node")

    def test_numeric_first_char_rejected(self) -> None:
        """Node ids must start with a letter or digit but NOT a digit alone
        ... actually the regex is ``[a-zA-Z0-9][a-zA-Z0-9_-]*`` so leading
        digit is fine; test that leading non-alphanumeric is rejected."""
        with pytest.raises(ValueError, match="invalid"):
            self._make("-abc")


# ---------------------------------------------------------------------------
# GraphNode defaults
# ---------------------------------------------------------------------------


class TestGraphNodeDefaults:
    def _make(self) -> GraphNode:
        return GraphNode(id="n", executable=_STUB)  # type: ignore[arg-type]

    def test_merge_defaults_to_default_merge(self) -> None:
        n = self._make()
        assert n.merge is DEFAULT_MERGE

    def test_join_defaults_to_and(self) -> None:
        n = self._make()
        assert n.join == JoinSemantics.AND

    def test_description_defaults_to_none(self) -> None:
        n = self._make()
        assert n.description is None

    def test_metadata_defaults_to_empty_dict(self) -> None:
        n = self._make()
        assert len(n.metadata) == 0


# ---------------------------------------------------------------------------
# GraphEdge self-loop rejection
# ---------------------------------------------------------------------------


class TestGraphEdge:
    def test_normal_edge_ok(self) -> None:
        e = GraphEdge(source="a", target="b")
        assert e.source == "a"
        assert e.target == "b"

    def test_self_loop_rejected(self) -> None:
        with pytest.raises(ValueError, match="[Ss]elf.loop"):
            GraphEdge(source="x", target="x")

    def test_optional_fields_default_to_none(self) -> None:
        e = GraphEdge(source="a", target="b")
        assert e.when is None
        assert e.label is None
        assert e.priority == 0


# ---------------------------------------------------------------------------
# GraphNode reliability fields
# ---------------------------------------------------------------------------


def test_graph_node_reliability_fields_default_none() -> None:
    from troopai.adk.graphs.node import GraphNode

    n = GraphNode(id="x", executable=_STUB)  # type: ignore[arg-type]
    assert n.retry is None
    assert n.timeout is None


def test_graph_node_accepts_reliability_overrides() -> None:
    from troopai.adk.graphs.config import NodeRetryPolicy
    from troopai.adk.graphs.node import GraphNode

    pol = NodeRetryPolicy(max_attempts=3)
    n = GraphNode(id="x", executable=_STUB, retry=pol, timeout=5.0)  # type: ignore[arg-type]
    assert n.retry is pol
    assert n.timeout == 5.0
