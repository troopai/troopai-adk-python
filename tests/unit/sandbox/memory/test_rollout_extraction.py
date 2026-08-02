"""Tests for ``troopai.adk.sandbox.memory.rollout_extraction``."""

from __future__ import annotations

import json

import pytest

from troopai.adk.sandbox.memory.interface import (
    RolloutExtractionArtifacts,
    RolloutTerminalMetadata,
)
from troopai.adk.sandbox.memory.rollout_extraction import (
    ROLLOUT_CONTENT_BYTE_CAP,
    run_rollout_extraction,
)


def _metadata() -> RolloutTerminalMetadata:
    return RolloutTerminalMetadata(terminal_state="completed", has_final_output=True)


@pytest.mark.asyncio
async def test_run_rollout_extraction_parses_valid_payload() -> None:
    async def fake_llm(_system: str, _user: str) -> str:
        return json.dumps(
            {
                "rollout_slug": "task-slug",
                "rollout_summary": "User asked X; agent did Y.",
                "raw_memory": "rollout_id: r1\n\n## Reusable knowledge\n- thing",
            }
        )

    artifacts = await run_rollout_extraction(
        rollout_id="r1",
        rollout_contents='{"role":"user","content":"hi"}',
        terminal_metadata=_metadata(),
        llm=fake_llm,
    )
    assert isinstance(artifacts, RolloutExtractionArtifacts)
    assert artifacts.rollout_slug == "task-slug"


@pytest.mark.asyncio
async def test_run_rollout_extraction_returns_none_for_noop_payload() -> None:
    async def fake_llm(_system: str, _user: str) -> str:
        return json.dumps({"rollout_slug": "", "rollout_summary": "", "raw_memory": ""})

    artifacts = await run_rollout_extraction(
        rollout_id="r1",
        rollout_contents='{"role":"user","content":"hi"}',
        terminal_metadata=_metadata(),
        llm=fake_llm,
    )
    assert artifacts is None


@pytest.mark.asyncio
async def test_run_rollout_extraction_rejects_non_json() -> None:
    async def fake_llm(_system: str, _user: str) -> str:
        return "this is not JSON"

    with pytest.raises(ValueError, match="non-JSON"):
        await run_rollout_extraction(
            rollout_id="r1",
            rollout_contents="x",
            terminal_metadata=_metadata(),
            llm=fake_llm,
        )


@pytest.mark.asyncio
async def test_run_rollout_extraction_rejects_invalid_schema() -> None:
    async def fake_llm(_system: str, _user: str) -> str:
        return json.dumps({"rollout_slug": "ok"})  # missing fields

    with pytest.raises(ValueError, match="schema validation"):
        await run_rollout_extraction(
            rollout_id="r1",
            rollout_contents="x",
            terminal_metadata=_metadata(),
            llm=fake_llm,
        )


@pytest.mark.asyncio
async def test_run_rollout_extraction_truncates_oversized_content() -> None:
    received_user_prompt: list[str] = []

    async def fake_llm(_system: str, user: str) -> str:
        received_user_prompt.append(user)
        return json.dumps({"rollout_slug": "s", "rollout_summary": "S", "raw_memory": "R"})

    huge = "x" * (ROLLOUT_CONTENT_BYTE_CAP + 1000)
    await run_rollout_extraction(
        rollout_id="r1",
        rollout_contents=huge,
        terminal_metadata=_metadata(),
        llm=fake_llm,
    )
    assert "rollout truncated for prompt budget" in received_user_prompt[0]


@pytest.mark.asyncio
async def test_run_rollout_extraction_passes_extra_prompt() -> None:
    captured_system: list[str] = []

    async def fake_llm(system: str, _user: str) -> str:
        captured_system.append(system)
        return json.dumps({"rollout_slug": "s", "rollout_summary": "S", "raw_memory": "R"})

    await run_rollout_extraction(
        rollout_id="r1",
        rollout_contents="x",
        terminal_metadata=_metadata(),
        llm=fake_llm,
        extra_prompt="prefer Python idioms",
    )
    assert "prefer Python idioms" in captured_system[0]
