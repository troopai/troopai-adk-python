"""Unit tests for :mod:`troopai.adk.visualization.helpers`.

Covers the collision detector, the empty-input edges of ``safe``, and
the escape function — the regression hooks the review-gate asked for.
"""

from __future__ import annotations

import pytest

from troopai.adk.visualization.helpers import (
    assert_no_collision,
    escape_label,
    escape_mermaid_label,
    gate_node_id,
    node_label_from_desc,
    safe,
)


class TestSafe:
    def test_alphanumeric_passes_through(self) -> None:
        assert safe("step_42") == "step_42"

    def test_empty_returns_underscore(self) -> None:
        assert safe("") == "_"

    def test_hyphen_replaced(self) -> None:
        assert safe("a-b") == "a_b"

    def test_dot_replaced(self) -> None:
        assert safe("module.attr") == "module_attr"


class TestEscapeLabel:
    def test_quote_escaped(self) -> None:
        assert '\\"' in escape_label('he said "hi"')

    def test_backslash_escaped_first(self) -> None:
        # Backslashes must be escaped first; otherwise the quote
        # escape's backslash would be double-escaped.
        out = escape_label("\\n")
        assert out == "\\\\n"

    def test_newline_replaced_with_space(self) -> None:
        assert escape_label("a\nb") == "a b"


class TestEscapeMermaidLabel:
    def test_quote_becomes_entity_code(self) -> None:
        # Mermaid quoted strings do not honour backslash escapes; a double
        # quote must become the '#quot;' entity code instead.
        assert escape_mermaid_label('say "hi"') == "say #quot;hi#quot;"

    def test_hash_escaped_first(self) -> None:
        # '#' is rewritten before the other replacements introduce a '#',
        # so a literal '#' cannot start a spurious entity code.
        assert escape_mermaid_label("C#") == "C#35;"

    def test_quote_after_hash_not_double_encoded(self) -> None:
        assert escape_mermaid_label('#"') == "#35;#quot;"

    def test_newline_becomes_br(self) -> None:
        assert escape_mermaid_label("a\nb") == "a<br>b"

    def test_backslash_left_literal(self) -> None:
        # Unlike DOT (escape_label), backslashes are not doubled.
        assert escape_mermaid_label("a\\b") == "a\\b"


class TestGateNodeId:
    def test_colons_replaced(self) -> None:
        assert gate_node_id("x:and:a,b") == "gate__x__and__a_b"

    def test_idempotent_when_already_safe(self) -> None:
        # No colons or commas → only the prefix is added.
        assert gate_node_id("plain") == "gate__plain"


class TestAssertNoCollision:
    def test_no_collision_records_mapping(self) -> None:
        forward: dict[str, str] = {}
        assert_no_collision(forward, "a-b", "a_b")
        assert forward["a-b"] == "a_b"

    def test_collision_raises(self) -> None:
        forward = {"a-b": "a_b"}
        with pytest.raises(ValueError, match="collides"):
            assert_no_collision(forward, "a.b", "a_b")

    def test_same_original_same_sanitised_is_noop(self) -> None:
        forward = {"x": "x"}
        # Re-registering the same mapping is fine.
        assert_no_collision(forward, "x", "x")
        assert forward["x"] == "x"

    def test_same_original_different_sanitised_raises(self) -> None:
        # Re-registering the same original under a different sanitised
        # id indicates a non-deterministic sanitiser (a contract bug);
        # we surface it loudly rather than silently overwriting.
        forward = {"x": "x"}
        with pytest.raises(ValueError, match="non-deterministic"):
            assert_no_collision(forward, "x", "y")


class TestNodeLabelFromDesc:
    def test_description_set_returns_description(self) -> None:
        assert node_label_from_desc("kickoff", {"kickoff": "Seed the run"}) == "Seed the run"

    def test_none_description_falls_back_to_name(self) -> None:
        assert node_label_from_desc("kickoff", {"kickoff": None}) == "kickoff"

    def test_missing_key_falls_back_to_name(self) -> None:
        assert node_label_from_desc("kickoff", {}) == "kickoff"

    def test_empty_string_description_preserved(self) -> None:
        # explicit empty-string description is not None — must NOT fall back to name
        assert node_label_from_desc("kickoff", {"kickoff": ""}) == ""


class TestAssertNoCollisionWithInverse:
    """O(1) path: same correctness guarantees with a caller-supplied inverse."""

    def test_inverse_records_mapping(self) -> None:
        forward: dict[str, str] = {}
        inverse: dict[str, str] = {}
        assert_no_collision(forward, "a-b", "a_b", inverse)
        assert forward["a-b"] == "a_b"
        assert inverse["a_b"] == "a-b"

    def test_inverse_detects_collision(self) -> None:
        forward = {"a-b": "a_b"}
        inverse = {"a_b": "a-b"}
        with pytest.raises(ValueError, match="collides"):
            assert_no_collision(forward, "a.b", "a_b", inverse)

    def test_inverse_noop_on_same_original_same_sanitised(self) -> None:
        forward = {"x": "x"}
        inverse = {"x": "x"}
        assert_no_collision(forward, "x", "x", inverse)
        assert forward == {"x": "x"}
        assert inverse == {"x": "x"}

    def test_inverse_raises_non_deterministic_sanitiser(self) -> None:
        forward = {"x": "x"}
        inverse = {"x": "x"}
        with pytest.raises(ValueError, match="non-deterministic"):
            assert_no_collision(forward, "x", "y", inverse)
