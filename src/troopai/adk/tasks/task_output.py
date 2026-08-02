"""TaskOutput — typed result of a single :class:`Task` execution.

Frozen dataclass surfacing the agent's final output, the
:class:`RunItem` trail, usage, and an explicit ``skipped`` flag for
pipeline-conditional execution. Skipped tasks return a
``TaskOutput(skipped=True, final_output=None, …)`` so a pipeline's
positional indexing stays stable — the slot is never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from troopai.adk.types.items import RunItem
    from troopai.adk.types.tokens.llm_usage import LLMUsage

_OUTPUT_PREVIEW_CHARS = 60
"""Max characters of a value shown in tasks-module ``__repr__`` previews."""


def _output_preview(output: Any) -> str:
    """Render a one-line, length-capped preview of a final output for reprs."""
    if output is None:
        return "None"
    text = output if isinstance(output, str) else repr(output)
    text = text.replace("\n", " ")
    if len(text) > _OUTPUT_PREVIEW_CHARS:
        text = text[: _OUTPUT_PREVIEW_CHARS - 1] + "…"
    return repr(text) if isinstance(output, str) else text


@dataclass(frozen=True, kw_only=True)
class TaskOutput:
    """Typed result of a single :class:`Task` execution.

    Attributes:
        task_id: The task's identity for this run. When ``Task.task_id``
            is set, this equals it verbatim; otherwise it is the fresh
            ``str(uuid.uuid4())`` (full 36-char canonical UUID) that
            the Runner generated. The verbose Task panel truncates
            the display to 8 chars, but the full UUID is what flows
            through hooks, tracing, and session events.
        task_name: The display name surfaced in verbose / hooks /
            tracing. Equals ``Task.name`` when set, else a truncated
            form of ``Task.description``.
        final_output: The agent's final output for this task. ``str``
            when no output schema is in effect; otherwise the parsed
            instance of the resolved ``AgentOutputSchemaBase`` type.
            ``None`` when :attr:`skipped` is ``True`` or
            :attr:`error` is set.
        new_items: The Layer 3 :class:`RunItem` trail produced by this
            task — messages, tool calls, tool outputs, handoffs.
            Empty tuple when :attr:`skipped` is ``True``.
        usage: The :class:`LLMUsage` accumulated **by this task's
            agent loop**. In a pipeline, summing
            ``output.usage.total_tokens`` over all outputs reproduces
            ``TaskPipelineResult.context.usage.total_tokens`` (the pipeline
            harness sums each completed task's usage into the pipeline
            ``RunContext`` after the task returns).
        skipped: ``True`` when :attr:`Task.skip_if` returned ``True``
            in a pipeline; the agent was never invoked. ``False`` for
            executed tasks, even on error.
        error: Stringified exception when the task raised. ``None`` on
            success. Mutually exclusive with :attr:`skipped` —
            skip-paths never set ``error``.
        streaming_placeholder: ``True`` when this slot was constructed
            by :meth:`Runner.arun_task_pipeline_streamed` as a
            placeholder for a task whose inner stream may not have
            drained by the time the next task's ``skip_if`` runs. The
            placeholder carries identity + metadata only —
            ``final_output``, ``new_items``, ``usage``, and ``error``
            are unset. ``skip_if`` callables that need real prior
            outputs must check this flag and either defer or treat the
            slot as unknown. ``False`` for every non-streamed path and
            for fully-resolved streamed slots.
        metadata: Verbatim copy of :attr:`Task.metadata`. Carried
            through so downstream consumers (audit, dashboards,
            tracing) can correlate this result with the originating
            request.
    """

    task_id: str
    """The task's identity for this run."""

    task_name: str
    """Display name surfaced in verbose / hooks / tracing."""

    final_output: Any = None
    """Agent's final output, parsed per ``output_schema`` when set."""

    new_items: tuple[RunItem, ...] = ()
    """Layer 3 RunItem trail produced by this task."""

    usage: LLMUsage | None = None
    """LLMUsage accumulated by this task's agent loop."""

    skipped: bool = False
    """``True`` when ``Task.skip_if`` returned ``True``."""

    error: str | None = None
    """Stringified exception when the task raised; ``None`` on success."""

    streaming_placeholder: bool = False
    """``True`` for the placeholder slot the streamed pipeline records
    when the consumer-driven inner stream may not have drained yet."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Verbatim copy of ``Task.metadata`` for downstream correlation."""

    @override
    def __repr__(self) -> str:
        """One-line task-result summary for humans.

        The full dataclass repr dumps every RunItem in ``new_items``
        and the whole ``metadata`` dict — unreadable in a REPL or log
        line. This shows what a human checks first: which task, whether
        it was skipped or failed, and a capped preview of the output.
        ``error`` renders as the exception class name (the stringified
        form is ``"ClassName: message"``).
        """
        parts: list[str] = [f"task_id={self.task_id!r}", f"task_name={self.task_name!r}"]
        if self.skipped:
            parts.append("skipped=True")
        if self.error is not None:
            error_label = self.error.split(":", 1)[0]
            if len(error_label) > _OUTPUT_PREVIEW_CHARS:
                error_label = error_label[: _OUTPUT_PREVIEW_CHARS - 1] + "…"
            parts.append(f"error={error_label}")
        if self.final_output is not None:
            parts.append(f"final_output={_output_preview(self.final_output)}")
        return f"TaskOutput({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for :class:`TaskPipelineState`.

        Trades fidelity for portability:

        * :attr:`final_output` is kept as-is when it is genuinely
          JSON-serializable (checked with a real ``json.dumps`` probe, so a
          ``list`` / ``dict`` containing Pydantic or dataclass values does
          NOT slip through), and serialized via ``str(final_output)``
          otherwise. On resume, type recovery is the developer's job — the
          framework does not reverse-engineer Pydantic / dataclass shapes.
        * :attr:`new_items` is NOT serialized — the audit trail for
          completed tasks lives on the
          :class:`~troopai.adk.session.session.Session` if attached.
          Resume rebuilds it from the session when needed; in-memory
          replays start with an empty trail.
        * :attr:`usage` is serialized to the four scalar token counts
          plus ``requests``. Detail breakdowns are dropped — the
          aggregated total is what pipelines care about for budget
          accounting on resume.

        Returns:
            A JSON-compatible ``dict`` whose values are ``str``,
            ``int``, ``float``, ``bool``, ``None``, ``list``, or
            ``dict``.  Load via :meth:`from_dict`.
        """
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        usage_dict: dict[str, int] | None = None
        if isinstance(self.usage, LLMUsage):
            usage_dict = {
                "requests": self.usage.requests,
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
            }
        # A shallow ``isinstance(x, (list, dict))`` check is not enough: a
        # ``list[BaseModel]`` (or a dict holding non-JSON values) passes it yet
        # crashes the ``json.dumps`` that :class:`TaskPipelineState` runs over
        # this dict. Probe with an actual dump — keep the value only when it is
        # genuinely JSON-serializable, else fall back to ``str()``.
        try:
            json.dumps(self.final_output)
            final_output_serialized: Any = self.final_output
        except (TypeError, ValueError):
            final_output_serialized = str(self.final_output)
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "final_output": final_output_serialized,
            "usage": usage_dict,
            "skipped": self.skipped,
            "error": self.error,
            "streaming_placeholder": self.streaming_placeholder,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskOutput:
        """Rehydrate a :class:`TaskOutput` from :meth:`to_dict` output.

        ``new_items`` is always reset to ``()`` (the trail is not
        serialized; replay it from the session if needed). ``usage``
        is rebuilt from the four scalar token counts when present.
        Other fields pass through unchanged.

        Args:
            data: A ``dict`` previously produced by :meth:`to_dict`.
                Must contain the ``task_id`` and ``task_name`` keys.

        Returns:
            A new :class:`TaskOutput` rehydrated from ``data``.

        Raises:
            ValueError: When ``task_id`` or ``task_name`` is absent from
                ``data``.
        """
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        missing = [k for k in ("task_id", "task_name") if k not in data]
        if len(missing) > 0:
            raise ValueError(
                f"TaskOutput dict missing required field(s): {missing!r}",
            )
        usage_dict = data.get("usage")
        usage: LLMUsage | None = None
        if isinstance(usage_dict, dict):
            usage = LLMUsage(
                requests=usage_dict.get("requests", 0),
                input_tokens=usage_dict.get("input_tokens", 0),
                output_tokens=usage_dict.get("output_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
            )
        return cls(
            task_id=data["task_id"],
            task_name=data["task_name"],
            final_output=data.get("final_output"),
            new_items=(),
            usage=usage,
            skipped=data.get("skipped", False),
            error=data.get("error"),
            streaming_placeholder=data.get("streaming_placeholder", False),
            metadata=dict(data.get("metadata", {})),
        )
