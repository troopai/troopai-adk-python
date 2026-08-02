"""Serialize a sandbox-agent run into a rollout JSONL segment.

A rollout is the JSON-encoded transcript of one agent run.
``build_rollout_payload`` turns a ``RunResult`` (or a partial
result + exception) into a dict that's safe to round-trip through
``json.dumps``; ``dump_rollout_json`` adds the newline framing
expected by the JSONL file format.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from troopai.adk.sandbox.memory.interface import RolloutTerminalMetadata, TerminalState

logger = logging.getLogger(__name__)

__all__ = [
    "build_rollout_payload",
    "dump_rollout_json",
    "terminal_metadata_for_exception",
    "terminal_metadata_for_result",
]


def dump_rollout_json(payload: dict[str, object]) -> str:
    """Return one JSONL-framed line representing ``payload``."""
    return json.dumps(payload, separators=(",", ":"), default=_json_fallback) + "\n"


def build_rollout_payload(
    *,
    rollout_id: str,
    inputs: object,
    new_items: object,
    final_output: object,
    interruptions: object = None,
    terminal_metadata: RolloutTerminalMetadata | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble a JSON-safe dict representing one rollout segment.

    Args:
        rollout_id: Stable identifier scoping this rollout.
        inputs: Inputs the run started with (often the initial user
            prompt). Stringified via ``_json_fallback`` if not
            directly JSON-encodable.
        new_items: Items the run produced (model outputs, tool
            calls, handoffs). Same encoding rule.
        final_output: The run's final output payload, or ``None``.
        interruptions: Optional approval / interruption payloads.
        terminal_metadata: How the run ended; defaults to a "completed"
            stub when omitted.
        extras: Free-form additional fields merged into the result.
    """
    metadata = terminal_metadata or RolloutTerminalMetadata(
        terminal_state="completed",
        has_final_output=final_output is not None,
    )
    payload: dict[str, object] = {
        "rollout_id": rollout_id,
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "input": inputs,
        "generated_items": new_items,
        "final_output": final_output,
        "interruptions": interruptions,
        "terminal_metadata": metadata.model_dump(),
    }
    if extras is not None:
        for key, value in extras.items():
            if key not in payload:
                payload[key] = value
    return payload


def terminal_metadata_for_result(
    *,
    has_final_output: bool,
    interrupted: bool = False,
    max_turns_exceeded: bool = False,
) -> RolloutTerminalMetadata:
    """Classify the terminal state of a successful run (no exception)."""
    state: TerminalState
    if max_turns_exceeded:
        state = "max_turns_exceeded"
    elif interrupted:
        state = "interrupted"
    else:
        state = "completed"
    return RolloutTerminalMetadata(
        terminal_state=state,
        has_final_output=has_final_output,
    )


def terminal_metadata_for_exception(exc: BaseException) -> RolloutTerminalMetadata:
    """Classify the terminal state implied by ``exc``."""
    state: TerminalState
    exc_name = type(exc).__name__
    if exc_name in {"InputGuardrailTripwireTriggered", "OutputGuardrailTripwireTriggered"}:
        state = "guardrail_tripped"
    elif exc_name == "MaxTurnsExceeded":
        state = "max_turns_exceeded"
    elif isinstance(exc, _asyncio_cancelled_error_type()):
        state = "cancelled"
    else:
        state = "failed"
    return RolloutTerminalMetadata(
        terminal_state=state,
        exception_type=_qualified_name(exc),
        exception_message=_first_message_line(exc),
        has_final_output=False,
    )


def _asyncio_cancelled_error_type() -> type[BaseException]:
    """Indirection to avoid eager asyncio import in module top-level annotations."""
    import asyncio

    return asyncio.CancelledError


def _qualified_name(exc: BaseException) -> str:
    cls = type(exc)
    module = cls.__module__
    qualname = cls.__qualname__
    if module in ("builtins", "__main__"):
        return qualname
    return f"{module}.{qualname}"


def _first_message_line(exc: BaseException) -> str:
    text = str(exc).strip()
    if len(text) == 0:
        return _qualified_name(exc)
    return text.splitlines()[0]


def _json_fallback(value: Any) -> object:
    """Best-effort encoder for non-JSON-native objects in rollouts.

    Returns a structured sentinel ``{"__unserializable_type__": ...}``
    when no encoder works — leaking ``repr()`` strings straight into
    durable memory would poison the LLM-visible audit trail.
    """
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump()
        except (TypeError, ValueError) as exc:
            logger.warning("rollouts: model_dump failed for %r: %s", value, exc)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except (TypeError, ValueError) as exc:
            logger.warning("rollouts: to_dict failed for %r: %s", value, exc)
    logger.warning(
        "rollouts: no serializer for %r — emitting unserializable sentinel into JSONL",
        type(value).__name__,
    )
    return {
        "__unserializable_type__": type(value).__name__,
        "repr": repr(value)[:200],
    }
