"""Feature 5: Builder node defaults.

Tests that:
- set_node_defaults() sets retry, timeout, metadata, on_error defaults.
- Subsequently added nodes inherit defaults when not overridden.
- Explicit per-node arguments always win over defaults.
- Nodes added BEFORE set_node_defaults() are unaffected.
- Metadata merging: default provides the base, per-node keys shadow.
- Calling set_node_defaults() multiple times accumulates (later calls
  shadow earlier ones field-by-field, not wholesale).
- Chaining: set_node_defaults() returns self.
"""

from __future__ import annotations

from troopai.adk.graphs.config import NodeRetryPolicy
from troopai.adk.graphs.graph import Graph


def _noop():
    return "noop"


class TestSetNodeDefaultsBasic:
    def test_retry_default_applied_to_subsequent_nodes(self) -> None:
        policy = NodeRetryPolicy(max_attempts=3)
        g = (
            Graph.new("nd-test")
            .set_node_defaults(retry=policy)
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        assert g.get_node("a").retry is policy
        assert g.get_node("b").retry is policy

    def test_timeout_default_applied(self) -> None:
        g = (
            Graph.new("nd-test2")
            .set_node_defaults(timeout=5.0)
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        assert g.get_node("a").timeout == 5.0
        assert g.get_node("b").timeout == 5.0

    def test_metadata_default_applied(self) -> None:
        g = (
            Graph.new("nd-test3")
            .set_node_defaults(metadata={"owner": "team-a", "env": "prod"})
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        assert g.get_node("a").metadata["owner"] == "team-a"
        assert g.get_node("b").metadata["env"] == "prod"

    def test_on_error_default_applied(self) -> None:
        from troopai.adk.orchestration.executable import NodeResult

        def handler(_nid: str, _exc: BaseException) -> NodeResult | None:
            return None

        g = (
            Graph.new("nd-test4")
            .set_node_defaults(on_error=handler)
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        assert g.get_node("a").on_error is handler
        assert g.get_node("b").on_error is handler


class TestSetNodeDefaultsPrecedence:
    def test_explicit_retry_wins_over_default(self) -> None:
        default_policy = NodeRetryPolicy(max_attempts=3)
        override_policy = NodeRetryPolicy(max_attempts=7)
        g = (
            Graph.new("prec-test")
            .set_node_defaults(retry=default_policy)
            .node("a", _noop, retry=override_policy)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        assert g.get_node("a").retry is override_policy
        assert g.get_node("b").retry is default_policy

    def test_explicit_timeout_wins(self) -> None:
        g = (
            Graph.new("prec-test2")
            .set_node_defaults(timeout=10.0)
            .node("a", _noop, timeout=2.0)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        assert g.get_node("a").timeout == 2.0
        assert g.get_node("b").timeout == 10.0

    def test_per_node_metadata_shadows_default_keys(self) -> None:
        """Per-node metadata keys shadow the same keys from the default."""
        g = (
            Graph.new("meta-test")
            .set_node_defaults(metadata={"owner": "default-team", "env": "prod"})
            .node("a", _noop, metadata={"owner": "override-team"})
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        # Per-node shadows 'owner', but 'env' is still from default
        assert g.get_node("a").metadata["owner"] == "override-team"
        assert g.get_node("a").metadata["env"] == "prod"
        # Node b inherits all defaults unchanged
        assert g.get_node("b").metadata["owner"] == "default-team"


class TestSetNodeDefaultsOrdering:
    def test_nodes_before_set_defaults_unaffected(self) -> None:
        """Nodes registered BEFORE set_node_defaults() must NOT inherit the defaults."""
        policy = NodeRetryPolicy(max_attempts=5)
        g = (
            Graph.new("order-test")
            .node("before", _noop)  # added before defaults
            .set_node_defaults(retry=policy)
            .node("after", _noop)  # added after defaults
            .edge("before", "after")
            .entry("before")
            .terminal("after")
            .compile()
        )
        # 'before' must not have the default policy
        assert g.get_node("before").retry is None
        # 'after' must have it
        assert g.get_node("after").retry is policy

    def test_multiple_set_defaults_calls_accumulate(self) -> None:
        """Later set_node_defaults() calls shadow earlier ones field-by-field."""
        policy1 = NodeRetryPolicy(max_attempts=2)
        policy2 = NodeRetryPolicy(max_attempts=9)
        g = (
            Graph.new("multi-defaults")
            .set_node_defaults(retry=policy1, timeout=3.0)
            .set_node_defaults(retry=policy2)  # shadows retry only
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        # retry was shadowed by policy2
        assert g.get_node("a").retry is policy2
        # timeout was set by the first call and not overwritten
        assert g.get_node("a").timeout == 3.0

    def test_set_node_defaults_returns_self_for_chaining(self) -> None:
        from troopai.adk.graphs.builder import GraphBuilder

        b: GraphBuilder = GraphBuilder(id="chain-test")
        returned = b.set_node_defaults(timeout=1.0)
        assert returned is b
