"""Tests for ``troopai.adk.sandbox.utils.deep_merge``."""

from __future__ import annotations

from troopai.adk.sandbox.utils.deep_merge import deep_merge


def test_overrides_win_on_leaf_collision() -> None:
    base = {"a": 1, "b": 2}
    out = deep_merge(base, {"b": 99})
    assert out == {"a": 1, "b": 99}
    assert base == {"a": 1, "b": 2}, "deep_merge must not mutate base"


def test_recursively_merges_nested_dicts() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    out = deep_merge(base, {"nested": {"y": 99, "z": 3}})
    assert out == {"a": 1, "nested": {"x": 1, "y": 99, "z": 3}}


def test_non_dict_value_overwrites_dict_value() -> None:
    base = {"a": {"x": 1}}
    out = deep_merge(base, {"a": "scalar"})
    assert out == {"a": "scalar"}


def test_dict_value_overwrites_non_dict_value() -> None:
    base = {"a": "scalar"}
    out = deep_merge(base, {"a": {"x": 1}})
    assert out == {"a": {"x": 1}}


def test_handles_empty_inputs() -> None:
    assert deep_merge({}, {}) == {}
    assert deep_merge({"a": 1}, {}) == {"a": 1}
    assert deep_merge({}, {"a": 1}) == {"a": 1}


def test_three_levels_deep_merge() -> None:
    base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
    out = deep_merge(base, {"l1": {"l2": {"l3": {"b": 99, "c": 3}}}})
    assert out == {"l1": {"l2": {"l3": {"a": 1, "b": 99, "c": 3}}}}


def test_non_string_keyed_dict_treated_as_scalar() -> None:
    base = {"x": {"a": 1}}
    overrides = {"x": {1: "int_key"}}
    out = deep_merge(base, overrides)
    assert out == {"x": {1: "int_key"}}
