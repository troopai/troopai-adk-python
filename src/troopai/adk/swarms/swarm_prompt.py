"""Opt-in system prompt prefix for swarm members.

Mirrors ``troopai.adk.handoffs.handoff_prompt.prompt_with_handoff_instructions``:
opt-in only, never auto-injected. Developers call this explicitly if
they want the LLM to understand the swarm routing mechanism.

The framework never auto-injects swarm-related prompts. Both Microsoft
AutoGen (``AssistantAgent`` default ``"Reply with TERMINATE"``) and
Strands (``_build_node_input`` multi-paragraph preamble) do so
implicitly; TroopAI does not. The helper exists so a developer who
wants the swarm-awareness prompt doesn't have to hand-write it, and so
the phrasing stays consistent across policies.
"""

from __future__ import annotations

RECOMMENDED_SWARM_PROMPT_PREFIX: str = (
    "# System context\n"
    "You are part of a multi-agent swarm. Agents can hand off the "
    "conversation to a peer by calling a function tool generally named "
    "`transfer_to_<agent_name>`. When your contribution is complete and "
    "no further peer work is needed, call the `swarm_done` tool with a "
    "concise `reason` to terminate the swarm. Transfers are handled in "
    "the background — do not mention them to the user.\n"
)
"""Recommended prefix for swarm-member system prompts.

Describes the two tools every ``SwarmPolicy`` may inject
(``transfer_to_<name>``, ``swarm_done``) so the LLM knows to call
them rather than trying to terminate implicitly.
"""


def prompt_with_swarm_instructions(prompt: str) -> str:
    """Prepend the recommended swarm instructions to a system prompt.

    Example::

        agent = Agent(
            name="Reviewer",
            system_prompt=prompt_with_swarm_instructions(
                "You review code for correctness and style."
            ),
            ...
        )

    Args:
        prompt: The original agent system prompt.

    Returns:
        The prompt with the swarm-awareness prefix prepended.
    """
    return f"{RECOMMENDED_SWARM_PROMPT_PREFIX}\n\n{prompt}"
