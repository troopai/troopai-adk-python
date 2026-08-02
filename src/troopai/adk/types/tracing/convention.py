"""Tracing attribute-convention selector."""

from __future__ import annotations

import enum


class TracingConvention(enum.Enum):
    """Selects the span-attribute vocabulary the OTel bridge emits."""

    DEFAULT = "default"
    """GenAI semconv (``gen_ai.*``) plus framework (``troopai.*``) attributes."""

    OPENINFERENCE = "openinference"
    """OpenInference conventions (``openinference.span.kind``, ``input.value``,
    ``output.value``, ``llm.token_count.*``) read natively by Phoenix/Arize."""
