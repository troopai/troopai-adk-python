"""Tests for shared ADK exceptions."""

from __future__ import annotations

from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.exceptions.exceptions import TroopAIError


class TestCheckpointConflictError:
    def test_thread_id_attribute(self) -> None:
        err = CheckpointConflictError("run-42")
        assert err.thread_id == "run-42"

    def test_str_contains_thread_id(self) -> None:
        err = CheckpointConflictError("run-42")
        assert "run-42" in str(err)

    def test_is_exception(self) -> None:
        err = CheckpointConflictError("t1")
        assert isinstance(err, Exception)

    def test_is_opus_ai_error(self) -> None:
        err = CheckpointConflictError("t1")
        assert isinstance(err, TroopAIError)


def test_tool_not_permitted_for_tenant() -> None:
    from troopai.adk.exceptions import ToolNotPermittedForTenant, TroopAIError

    err = ToolNotPermittedForTenant(tenant_id="t1", tool_name="search", agent_name="a")
    assert isinstance(err, TroopAIError)
    assert err.tenant_id == "t1"
    assert err.tool_name == "search"
    assert err.agent_name == "a"
    assert "search" in str(err)
    assert "t1" in str(err)
    assert "on agent 'a'" in str(err)


def test_tool_not_permitted_custom_message() -> None:
    from troopai.adk.exceptions import ToolNotPermittedForTenant

    err = ToolNotPermittedForTenant(tenant_id="t1", tool_name="x", agent_name="a", message="nope")
    assert str(err) == "nope"
