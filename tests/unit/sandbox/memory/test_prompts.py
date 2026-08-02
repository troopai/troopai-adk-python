"""Tests for ``troopai.adk.sandbox.memory.prompts``."""

from __future__ import annotations

from troopai.adk.sandbox.memory.prompts import (
    render_memory_consolidation_prompt,
    render_memory_read_prompt,
    render_rollout_extraction_prompt,
    render_rollout_extraction_user_prompt,
)
from troopai.adk.sandbox.memory.storage import ConsolidationInputSelection, ConsolidationSelectionItem


def test_render_rollout_extraction_prompt_includes_required_sections() -> None:
    out = render_rollout_extraction_prompt()
    assert "rollout_slug" in out
    assert "rollout_summary" in out
    assert "raw_memory" in out
    assert "GATING RULES" in out


def test_render_rollout_extraction_prompt_with_extra() -> None:
    out = render_rollout_extraction_prompt(extra_prompt="Always be terse.")
    assert "Always be terse." in out


def test_render_rollout_extraction_prompt_drops_empty_extra() -> None:
    out_a = render_rollout_extraction_prompt(extra_prompt=None)
    out_b = render_rollout_extraction_prompt(extra_prompt="")
    out_c = render_rollout_extraction_prompt(extra_prompt="   ")
    assert "ADDITIONAL GUIDANCE" not in out_a
    assert "ADDITIONAL GUIDANCE" not in out_b
    assert "ADDITIONAL GUIDANCE" not in out_c


def test_render_rollout_extraction_user_prompt() -> None:
    out = render_rollout_extraction_user_prompt(
        terminal_metadata_json='{"terminal_state":"completed"}',
        rollout_contents='{"role":"user","content":"hi"}\n{"role":"assistant","content":"hello"}',
    )
    assert "Terminal metadata" in out
    assert '"terminal_state":"completed"' in out
    assert "hi" in out
    assert "hello" in out


def test_render_memory_consolidation_prompt_no_removed() -> None:
    selection = ConsolidationInputSelection(selected=[], retained_rollout_ids=set(), removed=[])
    out = render_memory_consolidation_prompt(memory_root="memories", selection=selection)
    assert "memories/MEMORY.md" in out
    assert "memories/memory_summary.md" in out
    assert "Retained (still in scope): 0" in out
    assert "Removed (evicted from scope): 0" in out
    assert "Removed entries:" not in out


def test_render_memory_consolidation_prompt_with_removed() -> None:
    removed = [
        ConsolidationSelectionItem(
            rollout_id="abc",
            updated_at="2025-01-01T00:00:00Z",
            rollout_path="sessions/abc.jsonl",
            rollout_summary_file="abc_slug.md",
            terminal_state="completed",
        ),
    ]
    selection = ConsolidationInputSelection(
        selected=[],
        retained_rollout_ids=set(),
        removed=removed,
    )
    out = render_memory_consolidation_prompt(memory_root="memories", selection=selection)
    assert "Removed entries:" in out
    assert "abc" in out
    assert "abc_slug.md" in out


def test_render_memory_consolidation_prompt_with_extra() -> None:
    selection = ConsolidationInputSelection(selected=[], retained_rollout_ids=set(), removed=[])
    out = render_memory_consolidation_prompt(
        memory_root="memories",
        selection=selection,
        extra_prompt="Be conservative when evicting.",
    )
    assert "Be conservative when evicting." in out


def test_render_memory_read_prompt() -> None:
    out = render_memory_read_prompt(memory_dir="memories", memory_summary="# Memory\n\nUser likes brevity.")
    assert "memories" in out
    assert "User likes brevity." in out
    assert "live_update" in out
