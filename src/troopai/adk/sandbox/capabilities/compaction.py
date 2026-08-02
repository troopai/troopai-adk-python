"""``CompactionCapability`` — automatic context-window truncation.

Mirrors OpenAI's Compaction. Two policies:

- ``StaticCompactionPolicy(threshold)`` — fixed token threshold,
  e.g. ``240_000``.
- ``DynamicCompactionPolicy(model_info, threshold=0.9)`` — picks the
  threshold as a fraction of the model's context window.

When ``policy`` is None, ``sampling_params`` auto-selects: if the
sampling params name a model with a known context window, it returns
a DynamicCompactionPolicy keyed to that model; otherwise it falls
back to a StaticCompactionPolicy at 240k tokens.

``process_context`` looks for a ``"compaction"`` marker item in the
input context (placed by an upstream compaction pass) and discards
everything before it, keeping only the suffix from the marker on.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, Literal, override

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from troopai.adk.sandbox.capabilities.base import SandboxCapability

__all__ = [
    "CompactionCapability",
    "CompactionModelInfo",
    "CompactionPolicy",
    "DynamicCompactionPolicy",
    "StaticCompactionPolicy",
]


_DEFAULT_COMPACT_THRESHOLD: int = 240_000


# Model name → context window in tokens. Names are normalized: lowercased,
# "openai/" prefix stripped, "-" and "." removed before lookup.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic Claude 4.x / 3.7 / 3 Opus — 200k context.
    "claudeopus47": 200_000,
    "claudeopus46": 200_000,
    "claudesonnet46": 200_000,
    "claudehaiku45": 200_000,
    "claudehaiku4520251001": 200_000,
    "claudesonnet45": 200_000,
    "claudeopus41": 200_000,
    "claudeopus4": 200_000,
    "claudesonnet4": 200_000,
    "claude37sonnet": 200_000,
    "claude3opus": 200_000,
    "claude3sonnet": 200_000,
    "claude3haiku": 200_000,
    "claude35sonnet": 200_000,
    "claude35haiku": 200_000,
    # OpenAI GPT family.
    "gpt41": 1_047_576,
    "gpt41mini": 1_047_576,
    "gpt41nano": 1_047_576,
    "gpt4o": 128_000,
    "gpt4omini": 128_000,
    "gpt5": 400_000,
    "gpt51": 400_000,
    "gpt52": 400_000,
    "gpt5mini": 400_000,
    "gpt5nano": 400_000,
    "gpt5chat": 128_000,
    # OpenAI o-series / codex.
    "o1": 200_000,
    "o1mini": 200_000,
    "o3": 200_000,
    "o3mini": 200_000,
    "o4": 200_000,
    "codexmini": 200_000,
    # Google Gemini.
    "gemini15pro": 1_000_000,
    "gemini15flash": 1_000_000,
    "gemini20flash": 1_000_000,
    "gemini25flash": 1_000_000,
    "gemini25pro": 1_000_000,
}


def _normalize_model_key(model: str) -> str:
    """Normalize a model identifier for context-window lookup.

    Lowercases, strips a ``"openai/"`` prefix, removes ``-`` and ``.``
    characters. Keeps the framework's window dict small and tolerant
    of minor naming variations across provider SDKs.
    """
    norm = model.lower()
    if norm.startswith("openai/"):
        norm = norm[len("openai/") :]
    if norm.startswith("anthropic/"):
        norm = norm[len("anthropic/") :]
    return norm.replace("-", "").replace(".", "").strip()


class CompactionModelInfo(BaseModel):
    """Per-model context-window record.

    Attributes:
        context_window: Maximum input tokens the model accepts.
    """

    model_config = ConfigDict(frozen=True)

    context_window: int
    """Maximum input tokens the model accepts."""

    @classmethod
    def maybe_for_model(cls, model: str) -> CompactionModelInfo | None:
        """Return info for ``model`` if known; otherwise ``None``."""
        window = _MODEL_CONTEXT_WINDOWS.get(_normalize_model_key(model))
        if window is None:
            return None
        return cls(context_window=window)

    @classmethod
    def for_model(cls, model: str) -> CompactionModelInfo:
        """Return info for ``model`` or raise ``ValueError`` if unknown."""
        info = cls.maybe_for_model(model)
        if info is None:
            raise ValueError(f"Unknown context window for model: {model!r}")
        return info


class CompactionPolicy(BaseModel, abc.ABC):
    """Abstract policy returning a per-call compaction threshold."""

    model_config = ConfigDict(frozen=True)

    type: str
    """Discriminator string."""

    @abc.abstractmethod
    def compaction_threshold(self, sampling_params: dict[str, Any]) -> int:
        """Return the token count at which compaction should trigger."""


class StaticCompactionPolicy(CompactionPolicy):
    """Fixed-threshold policy.

    Attributes:
        threshold: Compaction trigger in tokens.
    """

    type: Literal["static"] = "static"
    """Discriminator. Always ``"static"``."""

    threshold: int = _DEFAULT_COMPACT_THRESHOLD
    """Compaction trigger in tokens (default 240_000)."""

    @override
    def compaction_threshold(self, sampling_params: dict[str, Any]) -> int:
        _ = sampling_params
        return self.threshold


class DynamicCompactionPolicy(CompactionPolicy):
    """Threshold proportional to a known model's context window.

    Attributes:
        model_info: Per-model context-window record.
        threshold: Fraction of the context window (0-1). Default 0.9.
    """

    type: Literal["dynamic"] = "dynamic"
    """Discriminator. Always ``"dynamic"``."""

    model_info: CompactionModelInfo
    """Per-model context-window record."""

    threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    """Fraction of the context window (0-1). Default 0.9."""

    @override
    def compaction_threshold(self, sampling_params: dict[str, Any]) -> int:
        _ = sampling_params
        return int(self.model_info.context_window * self.threshold)


class CompactionCapability(SandboxCapability):
    """Capability that drives automatic context compaction.

    Cost-conservative default: when no policy is set, auto-selects
    a DynamicCompactionPolicy if the run's sampling params name a
    known model, otherwise falls back to a StaticCompactionPolicy.

    ``process_context`` truncates the input context before the most
    recent ``type=="compaction"`` marker item (placed by an upstream
    compaction pass).
    """

    type: Literal["compaction"] = "compaction"
    """Discriminator. Always ``"compaction"``."""

    policy: CompactionPolicy | None = None
    """Explicit compaction policy, or ``None`` to auto-select."""

    @field_validator("policy", mode="before")
    @classmethod
    def _validate_policy(cls, value: object) -> object | None:
        if value is None:
            return None
        if isinstance(value, CompactionPolicy):
            return value
        if isinstance(value, Mapping):
            policy_type = value.get("type")
            if policy_type == "static":
                return StaticCompactionPolicy.model_validate(dict(value))
            if policy_type == "dynamic":
                return DynamicCompactionPolicy.model_validate(dict(value))
            raise ValueError(f"Unsupported compaction policy type: {policy_type!r}")
        return value

    @field_serializer("policy", when_used="always", return_type=dict[str, Any] | None)
    def _serialize_policy(self, policy: CompactionPolicy | None) -> dict[str, Any] | None:
        if policy is None:
            return None
        return policy.model_dump()

    @override
    def sampling_params(self, params: dict[str, Any]) -> dict[str, Any]:
        policy = self.policy
        if policy is None:
            model = params.get("model")
            if isinstance(model, str) and len(model) > 0:
                info = CompactionModelInfo.maybe_for_model(model)
                policy = DynamicCompactionPolicy(model_info=info) if info is not None else StaticCompactionPolicy()
            else:
                policy = StaticCompactionPolicy()
        return {
            "context_management": [
                {
                    "type": "compaction",
                    "compact_threshold": policy.compaction_threshold(params),
                }
            ]
        }

    @override
    def process_context(self, context: list[Any]) -> list[Any]:
        """Truncate context to suffix-from-last-compaction-marker.

        Scans backwards for an item whose ``type`` is ``"compaction"``;
        returns the slice from that index onward. Returns the context
        unchanged when no marker is found.
        """
        last_index: int | None = None
        for index in range(len(context) - 1, -1, -1):
            item = context[index]
            item_type = item.get("type") if isinstance(item, Mapping) else getattr(item, "type", None)
            if item_type == "compaction":
                last_index = index
                break
        if last_index is not None:
            return context[last_index:]
        return context
