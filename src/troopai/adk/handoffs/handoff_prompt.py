"""System prompt prefix for LLM-orchestrated multi-agent handoffs.

Provides context to the LLM about the handoff mechanism so it can
properly use ``transfer_to_<agent_name>`` function tools.

Inspired by OpenAI's ``extensions/handoff_prompt.py``.
"""

RECOMMENDED_PROMPT_PREFIX = (
    "# System context\n"
    "You are part of a multi-agent system. Agents can hand off a conversation "
    "to another agent when appropriate. Handoffs are achieved by calling a "
    "handoff function, generally named `transfer_to_<agent_name>`. "
    "Transfers between agents are handled seamlessly in the background; "
    "do not mention or draw attention to these transfers in your conversation "
    "with the user.\n"
)


def prompt_with_handoff_instructions(prompt: str) -> str:
    """Prepend recommended handoff instructions to a prompt string.

    Args:
        prompt: The original system prompt string.

    Returns:
        The prompt with handoff instructions prepended.
    """
    return f"{RECOMMENDED_PROMPT_PREFIX}\n\n{prompt}"
