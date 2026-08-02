"""Tests for State class — delta tracking and temp prefix."""

import pytest

from troopai.adk.session.state import State


class TestBasicAccess:
    def test_getitem_setitem(self):
        s = State.from_dict({"name": "Alice"})
        assert s["name"] == "Alice"
        s["score"] = 42
        assert s["score"] == 42

    def test_getitem_missing_raises(self):
        s = State()
        with pytest.raises(KeyError):
            _ = s["missing"]

    def test_contains(self):
        s = State.from_dict({"a": 1})
        assert "a" in s
        assert "b" not in s

    def test_get_with_default(self):
        s = State.from_dict({"a": 1})
        assert s.get("a") == 1
        assert s.get("b", 99) == 99

    def test_delitem(self):
        s = State.from_dict({"a": 1, "b": 2})
        del s["a"]
        assert "a" not in s
        assert s.get("a") is None

    def test_delitem_missing_raises(self):
        s = State()
        with pytest.raises(KeyError):
            del s["missing"]

    def test_len(self):
        s = State.from_dict({"a": 1, "b": 2})
        assert len(s) == 2
        s["c"] = 3
        assert len(s) == 3

    def test_iter(self):
        s = State.from_dict({"a": 1, "b": 2})
        assert set(s) == {"a", "b"}

    def test_keys_values_items(self):
        s = State.from_dict({"x": 10, "y": 20})
        assert set(s.keys()) == {"x", "y"}
        assert set(s.values()) == {10, 20}
        assert set(s.items()) == {("x", 10), ("y", 20)}


class TestDeltaTracking:
    def test_delta_tracks_new_keys(self):
        s = State.from_dict({"a": 1})
        s["b"] = 2
        assert s.delta() == {"b": 2}

    def test_delta_tracks_updates(self):
        s = State.from_dict({"a": 1})
        s["a"] = 99
        assert s.delta() == {"a": 99}

    def test_delta_tracks_deletes(self):
        s = State.from_dict({"a": 1})
        del s["a"]
        assert s.delta() == {"a": None}

    def test_has_changes(self):
        s = State.from_dict({"a": 1})
        assert not s.has_changes()
        s["b"] = 2
        assert s.has_changes()

    def test_commit_applies_delta(self):
        s = State.from_dict({"a": 1})
        s["b"] = 2
        s["a"] = 99
        s.commit()
        assert s["a"] == 99
        assert s["b"] == 2
        assert not s.has_changes()

    def test_commit_applies_deletes(self):
        s = State.from_dict({"a": 1, "b": 2})
        del s["a"]
        s.commit()
        assert "a" not in s
        assert "b" in s

    def test_delta_cleared_after_commit(self):
        s = State()
        s["x"] = 1
        s.commit()
        assert s.delta() == {}

    def test_delta_overrides_data(self):
        s = State.from_dict({"a": 1})
        s["a"] = 2
        assert s["a"] == 2  # delta wins


class TestToPersist:
    def test_to_persist_merges_data_and_delta(self):
        s = State.from_dict({"a": 1})
        s["b"] = 2
        result = s.to_persist()
        assert result == {"a": 1, "b": 2}

    def test_to_persist_excludes_deleted(self):
        s = State.from_dict({"a": 1, "b": 2})
        del s["a"]
        result = s.to_persist()
        assert result == {"b": 2}

    def test_to_persist_excludes_temp(self):
        s = State.from_dict({"a": 1})
        s["temp:scratch"] = "wip"
        result = s.to_persist()
        assert result == {"a": 1}
        assert "temp:scratch" not in result


class TestTempPrefix:
    def test_temp_key_stored_in_memory(self):
        s = State()
        s["temp:scratch"] = "wip"
        assert s["temp:scratch"] == "wip"
        assert "temp:scratch" in s

    def test_temp_key_not_in_delta(self):
        s = State()
        s["temp:scratch"] = "wip"
        assert s.delta() == {}

    def test_temp_key_not_in_to_persist(self):
        s = State()
        s["temp:scratch"] = "wip"
        assert s.to_persist() == {}

    def test_temp_key_in_iter(self):
        s = State()
        s["temp:x"] = 1
        s["y"] = 2
        assert set(s) == {"temp:x", "y"}

    def test_del_temp_key(self):
        s = State()
        s["temp:x"] = 1
        del s["temp:x"]
        assert "temp:x" not in s

    def test_del_missing_temp_raises(self):
        s = State()
        with pytest.raises(KeyError):
            del s["temp:missing"]


class TestAppPrefix:
    def test_app_prefix_keys_stored_normally(self):
        s = State.from_dict({"app:config": "v1"})
        assert s["app:config"] == "v1"

    def test_app_prefix_tracked_in_delta(self):
        s = State()
        s["app:config"] = "v2"
        assert s.delta() == {"app:config": "v2"}

    def test_app_prefix_excluded_from_to_persist(self):
        # app:-prefixed keys are owned by the app-state table, not the
        # sessions column; to_persist() must exclude them to prevent stale
        # app-state values from shadowing the canonical table on reload.
        s = State()
        s["app:config"] = "v2"
        s["normal"] = "keep"
        result = s.to_persist()
        assert "app:config" not in result
        assert result.get("normal") == "keep"

    def test_app_prefix_in_data_excluded_from_to_persist(self):
        # app: keys already in the data layer (loaded from DB) must also be
        # excluded — the session column must never contain app: entries.
        s = State.from_dict({"app:tenant": "acme", "other": "yes"})
        result = s.to_persist()
        assert "app:tenant" not in result
        assert result.get("other") == "yes"


class TestFromDict:
    def test_from_dict_creates_clean_state(self):
        s = State.from_dict({"a": 1, "b": 2})
        assert s["a"] == 1
        assert not s.has_changes()

    def test_from_dict_copies_data(self):
        original = {"a": 1}
        s = State.from_dict(original)
        s["a"] = 99
        assert original["a"] == 1  # not mutated

    def test_empty_state(self):
        s = State()
        assert len(s) == 0
        assert not s.has_changes()
        assert s.to_persist() == {}
