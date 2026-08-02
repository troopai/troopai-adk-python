"""the extraction step — extract per-rollout artifacts via an LLM call.

Reads a single rollout's JSONL contents from the sandbox session,
templates the extraction prompt, and calls the configured LLM.
Returns the parsed ``RolloutExtractionArtifacts``; the caller
(usually ``SandboxMemoryManager``) persists the result through
``SandboxMemoryStorage``.

The LLM caller is injected (any async callable with the
``Callable[[str, str], Awaitable[str]]`` signature — system_prompt,
user_prompt → raw JSON response). This keeps the module
testable without spinning up a real provider client.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from troopai.adk.sandbox.memory.interface import (
    RolloutExtractionArtifacts,
    RolloutTerminalMetadata,
)
from troopai.adk.sandbox.memory.prompts import (
    render_rollout_extraction_prompt,
    render_rollout_extraction_user_prompt,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ROLLOUT_CONTENT_BYTE_CAP",
    "ROLLOUT_CONTENT_TOKEN_CAP",
    "RolloutExtractionLLMCaller",
    "run_rollout_extraction",
]

RolloutExtractionLLMCaller = Callable[[str, str], Awaitable[str]]
"""Callback signature: ``async (system_prompt, user_prompt) -> raw_json_response``."""

ROLLOUT_CONTENT_TOKEN_CAP: int = 150_000
"""Soft ceiling on the rollout-JSONL token budget for the extraction step prompts."""

ROLLOUT_CONTENT_BYTE_CAP: int = ROLLOUT_CONTENT_TOKEN_CAP * 4
"""Bytes-budget approximation of the token cap (4 bytes / token)."""


async def run_rollout_extraction(
    *,
    rollout_id: str,
    rollout_contents: str,
    terminal_metadata: RolloutTerminalMetadata,
    llm: RolloutExtractionLLMCaller,
    extra_prompt: str | None = None,
) -> RolloutExtractionArtifacts | None:
    """Drive the extraction LLM call and parse the response.

    Args:
        rollout_id: Stable identifier for the rollout (used for log lines).
        rollout_contents: Full JSONL body of the rollout — truncated
            to ``ROLLOUT_CONTENT_BYTE_CAP`` bytes before prompting.
        terminal_metadata: How the rollout ended.
        llm: Async callable returning the LLM's raw response text.
        extra_prompt: Optional appendix concatenated to the system
            prompt — useful for per-deployment guidance.

    Returns:
        ``RolloutExtractionArtifacts`` on signal, ``None`` when the
        LLM emitted the empty-no-op response (per the prompt's
        gating rule).
    """
    truncated = _truncate_for_prompt(rollout_contents)
    system_prompt = render_rollout_extraction_prompt(extra_prompt=extra_prompt)
    user_prompt = render_rollout_extraction_user_prompt(
        terminal_metadata_json=json.dumps(terminal_metadata.model_dump(), sort_keys=True),
        rollout_contents=truncated,
    )

    logger.debug(
        "extract[%s]: prompt %d chars, content %d chars (truncated from %d)",
        rollout_id,
        len(system_prompt) + len(user_prompt),
        len(truncated),
        len(rollout_contents),
    )

    raw_response = await llm(system_prompt, user_prompt)
    return _parse_response(raw_response, rollout_id=rollout_id)


def _parse_response(raw: str, *, rollout_id: str) -> RolloutExtractionArtifacts | None:
    """Decode ``raw`` to ``RolloutExtractionArtifacts`` or ``None`` for no-op."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extract[{rollout_id}]: LLM returned non-JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"extract[{rollout_id}]: LLM JSON must be an object, got {type(payload).__name__}")

    if _is_noop_payload(payload):
        logger.info("extract[%s]: no-op response, skipping persistence", rollout_id)
        return None

    try:
        return RolloutExtractionArtifacts.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"extract[{rollout_id}]: response failed schema validation: {exc}") from exc


def _is_noop_payload(payload: dict[str, object]) -> bool:
    """The prompt's gating rule emits all-empty strings to signal 'skip me'."""
    slug = payload.get("rollout_slug")
    summary = payload.get("rollout_summary")
    raw_memory = payload.get("raw_memory")
    if not isinstance(slug, str) or not isinstance(summary, str) or not isinstance(raw_memory, str):
        return False
    return len(slug.strip()) == 0 and len(summary.strip()) == 0 and len(raw_memory.strip()) == 0


def _truncate_for_prompt(content: str) -> str:
    """Soft-cap the rollout body so the prompt fits a typical context window."""
    encoded = content.encode("utf-8")
    if len(encoded) <= ROLLOUT_CONTENT_BYTE_CAP:
        return content
    head = ROLLOUT_CONTENT_BYTE_CAP // 2
    tail = ROLLOUT_CONTENT_BYTE_CAP - head
    head_text = encoded[:head].decode("utf-8", errors="ignore")
    tail_text = encoded[-tail:].decode("utf-8", errors="ignore")
    return f"{head_text}\n…(rollout truncated for prompt budget; {len(encoded) - ROLLOUT_CONTENT_BYTE_CAP} bytes elided)…\n{tail_text}"
