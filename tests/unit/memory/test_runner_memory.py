"""Tests for Runner memory integration (inject + extract)."""

from unittest.mock import patch

import pytest

from troopai.adk.memory.extractor import ExtractionResult
from troopai.adk.memory.in_memory import TemporaryMemory
from troopai.adk.memory.memory_config import MemoryConfig, MemoryInjectionPosition
from troopai.adk.memory.memory_types import MemorySource
from troopai.adk.run.runner import _extract_query, _inject_memories


class TestExtractQuery:
    def test_string_input(self):
        assert _extract_query("hello world") == "hello world"

    def test_message_list(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is Python?"},
        ]
        assert _extract_query(messages) == "What is Python?"

    def test_message_list_last_user(self):
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        assert _extract_query(messages) == "Follow-up"

    def test_empty_list(self):
        assert _extract_query([]) == ""


class TestInjectMemories:
    @pytest.fixture
    def memory(self):
        return TemporaryMemory()

    @pytest.mark.asyncio
    async def test_inject_as_developer_message(self, memory):
        await memory.add("User prefers dark mode", namespace="ns")

        config = MemoryConfig(
            memory=memory,
            namespace="ns",
            inject=True,
            inject_limit=5,
            inject_position=MemoryInjectionPosition.DEVELOPER_MESSAGE,
        )

        result = await _inject_memories("Tell me about dark mode preferences", config)

        # Should convert to list with developer message + user message
        assert isinstance(result, list)
        assert result[0]["role"] == "developer"
        assert "dark mode" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Tell me about dark mode preferences"

    @pytest.mark.asyncio
    async def test_inject_as_system_suffix_new_message(self, memory):
        await memory.add("User prefers dark mode", namespace="ns")

        config = MemoryConfig(
            memory=memory,
            namespace="ns",
            inject=True,
            inject_position=MemoryInjectionPosition.SYSTEM_SUFFIX,
        )

        result = await _inject_memories("What dark mode setting?", config)

        assert isinstance(result, list)
        assert result[0]["role"] == "system"
        assert "dark mode" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_inject_as_system_suffix_existing_system(self, memory):
        await memory.add("User prefers dark mode", namespace="ns")

        config = MemoryConfig(
            memory=memory,
            namespace="ns",
            inject=True,
            inject_position=MemoryInjectionPosition.SYSTEM_SUFFIX,
        )

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "dark mode preferences"},
        ]
        result = await _inject_memories(messages, config)

        assert isinstance(result, list)
        assert result[0]["role"] == "system"
        assert "You are helpful" in result[0]["content"]
        assert "dark mode" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_inject_no_matches_returns_unchanged(self, memory):
        config = MemoryConfig(
            memory=memory,
            namespace="ns",
            inject=True,
        )

        result = await _inject_memories("hello", config)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_inject_empty_query_returns_unchanged(self, memory):
        config = MemoryConfig(
            memory=memory,
            namespace="ns",
            inject=True,
        )

        result = await _inject_memories("", config)
        assert result == ""


class TestRunnerMemoryIntegration:
    """Integration test: Runner.arun() with memory config."""

    @pytest.mark.asyncio
    async def test_memory_injection_in_arun(self):
        """Verify that memory injection modifies input before agent loop."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.runner import Runner

        memory = TemporaryMemory()
        await memory.add("User prefers concise answers", namespace="user:1")

        agent = Agent(name="test", system_prompt="You are helpful.")

        memory_config = MemoryConfig(
            memory=memory,
            namespace="user:1",
            inject=True,
            inject_limit=3,
        )

        # Mock the agent loop to capture what input it receives
        captured_input = None

        async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
            nonlocal captured_input
            captured_input = user_prompt
            # Return a minimal result
            from troopai.adk.types.run import RunResult

            return RunResult(
                final_output="Done",
                user_prompt=user_prompt,
                new_items=[],
                context=kwargs["context"],
                last_agent=agent,
            )

        with (
            patch("troopai.adk.run.runner.run_agent_loop", side_effect=mock_run_agent_loop),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_output_guardrails", return_value=[]),
        ):
            _ = await Runner.arun(
                agent,
                "What does the user prefer for answers?",
                memory=memory_config,
            )

        # The input to agent loop should include injected memories
        assert captured_input is not None
        assert isinstance(captured_input, list)
        # Developer message with memories should be first
        assert captured_input[0]["role"] == "developer"
        assert "concise" in captured_input[0]["content"]

    @pytest.mark.asyncio
    async def test_memory_extraction_after_run(self):
        """Verify that memory extraction happens after session save."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.runner import Runner

        memory = TemporaryMemory()

        class FakeExtractor:
            extract_called = False

            async def extract(self, _messages, *, namespace):
                self.extract_called = True
                return [
                    ExtractionResult(content="Extracted fact", importance=3),
                ]

        extractor = FakeExtractor()

        agent = Agent(name="test", system_prompt="You are helpful.")

        memory_config = MemoryConfig(
            memory=memory,
            namespace="user:1",
            inject=False,
            auto_extract=True,
            extractor=extractor,
        )

        async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
            from troopai.adk.types.run import RunResult

            return RunResult(
                final_output="Done",
                user_prompt=user_prompt,
                new_items=[],
                context=kwargs["context"],
                last_agent=agent,
            )

        with (
            patch("troopai.adk.run.runner.run_agent_loop", side_effect=mock_run_agent_loop),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_output_guardrails", return_value=[]),
        ):
            await Runner.arun(
                agent,
                "Tell me about Python",
                memory=memory_config,
            )

        assert extractor.extract_called
        # Verify the extracted fact was stored
        results = await memory.search("Extracted", namespace="user:1")
        assert len(results) == 1
        assert results[0].entry.metadata.source == MemorySource.EXTRACTION
