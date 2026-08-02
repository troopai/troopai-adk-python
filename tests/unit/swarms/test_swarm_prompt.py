"""Tests for the opt-in swarm prompt helper.

The framework never auto-injects swarm prompts; this helper is the
supported opt-in path, so its contract (prefix prepended, original
prompt preserved verbatim) is pinned here.
"""

from __future__ import annotations

from troopai.adk.swarms.swarm_prompt import (
    RECOMMENDED_SWARM_PROMPT_PREFIX,
    prompt_with_swarm_instructions,
)


class TestPromptWithSwarmInstructions:
    def test_prefix_is_prepended(self) -> None:
        result = prompt_with_swarm_instructions("You review code.")
        assert result.startswith(RECOMMENDED_SWARM_PROMPT_PREFIX)

    def test_original_prompt_preserved_verbatim(self) -> None:
        original = "You review code for correctness and style."
        result = prompt_with_swarm_instructions(original)
        assert result.endswith(original)

    def test_prefix_documents_both_injected_tools(self) -> None:
        # The prefix exists so the LLM knows to call transfer_to_<name>
        # and swarm_done rather than terminating implicitly — if either
        # mention is dropped, routing silently degrades.
        assert "transfer_to_" in RECOMMENDED_SWARM_PROMPT_PREFIX
        assert "swarm_done" in RECOMMENDED_SWARM_PROMPT_PREFIX

    def test_empty_prompt_returns_prefix(self) -> None:
        result = prompt_with_swarm_instructions("")
        assert result.strip() == RECOMMENDED_SWARM_PROMPT_PREFIX.strip()
