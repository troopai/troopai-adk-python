"""Tests for ``load_skill_tool`` — args validation + loader dispatch.

The factory ``make_load_skill_tool`` runs when ``SkillsCapability``
builds its tools, but the tool's invocation path (``LoadSkillArgs``
validation + ``_on_invoke`` → ``loader.load_skill``) was previously
exercised by no test. These pin that path.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from troopai.adk.sandbox.tools.load_skill_tool import (
    LoadSkillArgs,
    make_load_skill_tool,
)


class TestLoadSkillArgs:
    def test_valid(self) -> None:
        assert LoadSkillArgs(skill_name="weather").skill_name == "weather"

    def test_skill_name_is_required(self) -> None:
        with pytest.raises(ValidationError):
            LoadSkillArgs()  # type: ignore[call-arg]


def _loader_returning(payload: dict[str, str]) -> Any:
    loader = AsyncMock()
    loader.load_skill = AsyncMock(return_value=payload)
    return loader


class TestMakeLoadSkillTool:
    def test_tool_shape(self) -> None:
        tool = make_load_skill_tool(loader=_loader_returning({}))
        assert tool.name == "load_skill"
        assert tool.schema is LoadSkillArgs
        assert tool.on_invoke is not None

    async def test_invoke_dispatches_to_loader(self) -> None:
        payload = {"SKILL.md": "# Weather\nUse the API."}
        loader = _loader_returning(payload)
        tool = make_load_skill_tool(loader=loader)
        raw = json.dumps({"skill_name": "weather"})
        result = await tool.on_invoke(None, raw)  # type: ignore[arg-type]
        assert result == payload
        loader.load_skill.assert_awaited_once_with("weather")

    async def test_invoke_bad_json_raises(self) -> None:
        tool = make_load_skill_tool(loader=_loader_returning({}))
        with pytest.raises(ValidationError):
            await tool.on_invoke(None, "not json at all")  # type: ignore[arg-type]

    async def test_invoke_skill_not_found_propagates(self) -> None:
        loader = AsyncMock()
        loader.load_skill = AsyncMock(side_effect=KeyError("no such skill: ghost"))
        tool = make_load_skill_tool(loader=loader)
        raw = json.dumps({"skill_name": "ghost"})
        with pytest.raises(KeyError, match="ghost"):
            await tool.on_invoke(None, raw)  # type: ignore[arg-type]
