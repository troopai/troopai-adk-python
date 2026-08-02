"""Tests for MemoryTool subclasses and MemoryTool.all()."""

import json

import pytest

from troopai.adk.memory.in_memory import TemporaryMemory
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
from troopai.adk.tools.builtin.memory_tool import (
    ForgetMemoryTool,
    RecallMemoryTool,
    RememberMemoryTool,
)


@pytest.fixture
def memory():
    return TemporaryMemory()


class TestMemoryToolSubclasses:
    def test_remember_is_executable_builtin(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        assert isinstance(tool, ExecutableBuiltinTool)
        assert tool.name == "remember"

    def test_recall_is_executable_builtin(self, memory):
        tool = RecallMemoryTool(memory=memory, namespace="user:1")
        assert isinstance(tool, ExecutableBuiltinTool)
        assert tool.name == "recall"

    def test_forget_is_executable_builtin(self, memory):
        tool = ForgetMemoryTool(memory=memory, namespace="user:1")
        assert isinstance(tool, ExecutableBuiltinTool)
        assert tool.name == "forget"

    def test_on_invoke_set_by_default(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        assert tool.on_invoke is not None


class TestRememberMemoryTool:
    @pytest.mark.asyncio
    async def test_stores_entry(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        result = await tool.on_invoke(
            None,
            json.dumps(
                {
                    "content": "User prefers dark mode",
                    "importance": 4,
                    "categories": ["preference"],
                }
            ),
        )
        assert "Remembered" in result

        search_results = await memory.search("dark mode", namespace="user:1")
        assert len(search_results) == 1
        assert search_results[0].entry.metadata.importance == 4

    @pytest.mark.asyncio
    async def test_default_importance(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        await tool.on_invoke(None, json.dumps({"content": "Some fact"}))

        search_results = await memory.search("fact", namespace="user:1")
        assert len(search_results) == 1
        assert search_results[0].entry.metadata.importance == 3


class TestRecallMemoryTool:
    @pytest.mark.asyncio
    async def test_finds_memories(self, memory):
        await memory.add("User prefers dark mode", namespace="user:1")

        tool = RecallMemoryTool(memory=memory, namespace="user:1")
        result = await tool.on_invoke(None, json.dumps({"query": "dark mode"}))
        assert "dark mode" in result

    @pytest.mark.asyncio
    async def test_no_results(self, memory):
        tool = RecallMemoryTool(memory=memory, namespace="user:1")
        result = await tool.on_invoke(None, json.dumps({"query": "nonexistent"}))
        assert "No memories found" in result


class TestForgetMemoryTool:
    @pytest.mark.asyncio
    async def test_deletes_entry(self, memory):
        entry = await memory.add("Temporary fact", namespace="user:1")

        tool = ForgetMemoryTool(memory=memory, namespace="user:1")
        result = await tool.on_invoke(None, json.dumps({"memory_id": entry.id}))
        assert "deleted" in result
        assert await memory.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_nonexistent(self, memory):
        tool = ForgetMemoryTool(memory=memory, namespace="user:1")
        result = await tool.on_invoke(None, json.dumps({"memory_id": "fake-id"}))
        assert "not found" in result


class TestRememberImportanceCoercion:
    """`importance` is `int | None`; the executor calls `on_invoke` without
    Pydantic validation, so an explicit null (or an omitted key) must fall
    back to the documented default of 3 rather than raising `TypeError`
    inside `max(1, min(5, ...))`.
    """

    @pytest.mark.asyncio
    async def test_null_importance_falls_back_to_default(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        assert tool.on_invoke is not None
        result = await tool.on_invoke(None, json.dumps({"content": "fact", "importance": None}))
        assert "Importance: 3" in result

    @pytest.mark.asyncio
    async def test_missing_importance_falls_back_to_default(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        assert tool.on_invoke is not None
        result = await tool.on_invoke(None, json.dumps({"content": "fact"}))
        assert "Importance: 3" in result

    @pytest.mark.asyncio
    async def test_explicit_importance_is_respected(self, memory):
        tool = RememberMemoryTool(memory=memory, namespace="user:1")
        assert tool.on_invoke is not None
        result = await tool.on_invoke(None, json.dumps({"content": "fact", "importance": 5}))
        assert "Importance: 5" in result
