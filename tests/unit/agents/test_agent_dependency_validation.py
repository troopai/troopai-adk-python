"""Unit tests for Agent.__post_init__ dependency validation.

When tools declare ``requires_env`` / ``requires_packages``, missing
items must surface at agent construction with a single
``ToolDependencyError`` listing every offender.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.agents import Agent
from troopai.adk.exceptions import ToolDependencyError
from troopai.adk.tools import ShellTool
from troopai.adk.tools.function_tool import FunctionTool

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _make_tool(name: str, **overrides: Any) -> FunctionTool:
    defaults: dict[str, Any] = {"name": name, "schema": MINIMAL_SCHEMA}
    defaults.update(overrides)
    return FunctionTool(**defaults)


class TestAgentDependencyValidation:
    def test_no_dependencies_passes(self) -> None:
        Agent(name="ok", system_prompt="x", tools=[_make_tool("t")])

    def test_satisfied_env_passes(self) -> None:
        with patch.dict("os.environ", {"TROOPAI_INT_TEST_VAR": "v"}, clear=False):
            Agent(
                name="ok",
                system_prompt="x",
                tools=[_make_tool("t", requires_env=("TROOPAI_INT_TEST_VAR",))],
            )

    def test_missing_env_raises(self) -> None:
        tool = _make_tool("slack", requires_env=("TROOPAI_INT_TEST_MISSING",))
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ToolDependencyError) as excinfo:
            Agent(name="ok", system_prompt="x", tools=[tool])
        assert excinfo.value.agent_name == "ok"
        assert "slack" in excinfo.value.missing
        assert excinfo.value.missing["slack"] == ["env:TROOPAI_INT_TEST_MISSING"]

    def test_missing_package_raises(self) -> None:
        tool = _make_tool(
            "x",
            requires_packages=("troopai-int-test-no-such-pkg-13579",),
        )
        with pytest.raises(ToolDependencyError) as excinfo:
            Agent(name="ok", system_prompt="x", tools=[tool])
        assert excinfo.value.missing["x"] == ["package:troopai-int-test-no-such-pkg-13579"]

    def test_aggregates_across_tools(self) -> None:
        tool_a = _make_tool("a", requires_env=("MISSING_A",))
        tool_b = _make_tool("b", requires_env=("MISSING_B",))
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ToolDependencyError) as excinfo:
            Agent(
                name="ok",
                system_prompt="x",
                tools=[tool_a, tool_b],
            )
        assert "a" in excinfo.value.missing
        assert "b" in excinfo.value.missing
        assert excinfo.value.missing["a"] == ["env:MISSING_A"]
        assert excinfo.value.missing["b"] == ["env:MISSING_B"]

    def test_error_message_lists_every_missing_item(self) -> None:
        tool = _make_tool(
            "deploy",
            requires_env=("MISSING_TOKEN",),
            requires_packages=("troopai-no-such-pkg-24680",),
        )
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ToolDependencyError) as excinfo:
            Agent(name="ok", system_prompt="x", tools=[tool])
        text = str(excinfo.value)
        assert "ok" in text
        assert "deploy" in text
        assert "env:MISSING_TOKEN" in text
        assert "package:troopai-no-such-pkg-24680" in text

    def test_non_function_tool_skipped(self) -> None:
        # A non-FunctionTool entry in agent.tools (here a ShellTool)
        # must NOT trip the validator. The isinstance(tool, FunctionTool)
        # guard is what keeps these entries out of the dependency walk.
        # Pair the ShellTool with a FunctionTool whose env var is set so
        # the loop actually iterates past the builtin.
        with patch.dict("os.environ", {"TROOPAI_BUILTIN_TEST_VAR": "v"}, clear=False):
            Agent(
                name="ok",
                system_prompt="x",
                tools=[
                    ShellTool(),
                    _make_tool("regular", requires_env=("TROOPAI_BUILTIN_TEST_VAR",)),
                ],
            )
