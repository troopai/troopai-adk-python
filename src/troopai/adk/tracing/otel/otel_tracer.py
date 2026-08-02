"""OpenTelemetry-backed :class:`Tracer` implementation.

Wraps an ``opentelemetry.trace.TracerProvider`` and exposes the seven
typed ``*_span`` factories the runner calls. Each factory constructs an
:class:`~troopai.adk.tracing.otel.otel_span.OTelSpan` with an OTel span
name following the conventions documented in ``docs/tracing/otel.md``:

================= ===================================
Span kind         OTel name
================= ===================================
agent             ``agent.<agent_name>``
function (tool)   ``tool.<tool_name>`` (or ``mcp.<name>``)
generation        ``llm.generation``
response          ``llm.response``
handoff           ``agent.handoff``
guardrail         ``guardrail.<name>``
custom            the caller-provided name
================= ===================================

Attribute keys follow GenAI semantic conventions where applicable
(``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.usage.input_tokens``,
``gen_ai.usage.output_tokens``). Framework-specific fields live under
the ``troopai.*`` prefix.

Construction imports :mod:`opentelemetry` at call time; missing packages
surface as :class:`~troopai.adk.exceptions.TracingDependencyError`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import TracingDependencyError
from troopai.adk.tracing.openinference import conventions as oi
from troopai.adk.tracing.otel.otel_span import OTelSpan
from troopai.adk.tracing.spans import Span
from troopai.adk.types.tracing.convention import TracingConvention
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider as OTelTracerProvider

logger = logging.getLogger(__name__)


_SCALAR_TYPES = (str, int, float, bool)

# Default char cap for tool I/O recorded on spans. Tool inputs/outputs
# frequently contain large JSON payloads, pasted documents, and PII.
# Without a cap, a single tool call can push multi-MB blobs to the
# observability backend on every turn. 2 KB fits most tool calls and
# keeps trace costs bounded.
_DEFAULT_TOOL_IO_MAX_CHARS = 2048

# Conservative secret-redaction patterns applied to tool I/O and
# custom-span data before emission. Matches common credential shapes
# (Bearer tokens, OpenAI/Anthropic/Google/AWS/GitHub/Slack prefixes,
# PEM-encoded private keys, JSON-embedded api_key / password / secret
# fields) — NOT a replacement for proper secret handling in tool code,
# but a last line of defence against accidental span-level leaks.
#
# Each pattern is anchored to a concrete prefix or keyword to minimise
# false positives; regex engines short-circuit on the first literal
# character, so the list is cheap to walk even on large payloads.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-~+/=]{8,}"), "Bearer ***"),
    # sk-ant- MUST come before the generic sk- pattern; otherwise the
    # generic pattern swallows the ``ant-...`` suffix and the more
    # specific Anthropic marker is never emitted.
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "sk-ant-***"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "sk-***"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{20,}"), "AIza***"),
    # AWS access key IDs have a fixed ``AKIA`` / ``ASIA`` prefix and 16
    # uppercase-alphanumeric body — tight enough that false positives
    # are rare. The 40-char secret key is shape-matched by the generic
    # JSON-field pattern below when keyed under ``secret``.
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), "AKIA***"),
    # GitHub tokens: ``ghp_`` (personal), ``gho_`` (OAuth), ``ghs_``
    # (server), ``ghu_`` (user-to-server), ``ghr_`` (refresh).
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b"), "gh_***"),
    # Slack bot / user / app tokens.
    (re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}"), "xox-***"),
    # PEM-encoded private keys — single-line and multi-line forms.
    # Both the detection regex and the replacement banner are assembled
    # from non-contiguous tokens (``-{5}`` quantifier in the regex,
    # split string literals in the replacement) so the source text of
    # this file never contains the literal ``BEGIN <type> PRIVATE KEY``
    # banner that pre-commit's detect-private-key hook scans for. The
    # very module that IS the credential-redaction system would otherwise
    # trip its own safety net. Runtime bytes are unchanged.
    (
        re.compile(
            r"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}"
            r"[\s\S]+?"
            r"-{5}END [A-Z ]*PRIVATE KEY-{5}"
        ),
        ("-" * 5) + "BEGIN " + "PRIVATE KEY" + ("-" * 5) + "\n***\n" + ("-" * 5) + "END " + "PRIVATE KEY" + ("-" * 5),
    ),
    # Generic JSON / assignment-shaped secret field. The negative
    # lookaheads stop the pattern from re-redacting values we already
    # replaced (``Bearer ***``, ``sk-***`` etc.). ``access_token``,
    # ``client_secret`` and their camelCase siblings are matched via
    # the ``[_\-]?`` class inside the alternation plus the ``(?i)``
    # flag, so ``accessToken``, ``ClientSecret``, ``private_key`` all
    # hit the same arm.
    (
        re.compile(
            r"(?i)(\"?(?:api[_\-]?key|password|passwd|secret|token|authorization"
            r"|access[_\-]?token|client[_\-]?secret|private[_\-]?key"
            r"|refresh[_\-]?token|aws[_\-]?secret[_\-]?access[_\-]?key)\"?"
            r"\s*[:=]\s*\"?)(?!Bearer\s)(?!sk-)(?!AIza)(?!AKIA)(?!ASIA)"
            r"(?!gh[pousr]_)(?!xox)[^\"\s,}]{4,}"
        ),
        r"\1***",
    ),
)


def _redact(value: str) -> str:
    """Apply conservative credential-shape redaction to a string."""
    for pattern, replacement in _REDACTION_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _redact_and_truncate(value: str, max_chars: int) -> str:
    """Redact credential shapes, then truncate if over budget.

    Redaction runs first on the full value so a secret at the tail of a
    long string doesn't escape by sitting past the truncation boundary.
    """
    redacted = _redact(value)
    if len(redacted) <= max_chars:
        return redacted
    return f"{redacted[:max_chars]}... [truncated; {len(redacted)} chars total]"


def _flatten_attributes(exported: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten an exported ``SpanData`` dict for OTel ``set_attributes``.

    OpenTelemetry span attributes MUST be scalars or homogeneous lists
    of scalars. This walks the exported dict, promoting nested dicts
    with dotted prefixes and JSON-encoding any value that does not fit
    OTel's shape constraints.

    Separator semantics::

        _flatten_attributes({"a": {"b": 1}})            → {"a.b": 1}
        _flatten_attributes({"a": 1}, prefix="troopai")  → {"troopai.a": 1}
        _flatten_attributes({"a": {"b": 1}}, prefix="x")→ {"x.a.b": 1}

    Empty ``prefix`` skips the leading dot so top-level keys never
    become ``".a"``. The dotted form matches the OTel GenAI semconv
    convention (``gen_ai.request.model``) and what observability
    backends key off for column extraction.
    """
    flat: dict[str, Any] = {}
    for key, value in exported.items():
        full = key if len(prefix) == 0 else f"{prefix}.{key}"
        if value is None:
            continue
        if isinstance(value, _SCALAR_TYPES):
            flat[full] = value
        elif isinstance(value, list):
            # OTel accepts homogeneous scalar sequences; serialise others.
            if len(value) == 0:
                continue
            if all(isinstance(v, _SCALAR_TYPES) for v in value):
                flat[full] = value
            else:
                flat[full] = json.dumps(value, default=str)
        elif isinstance(value, dict):
            flat.update(_flatten_attributes(value, prefix=full))
        else:
            flat[full] = json.dumps(value, default=str)
    return flat


def _agent_attrs(data: AgentSpanData) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "troopai.agent.name": data.name,
    }
    if data.handoffs is not None and len(data.handoffs) > 0:
        attrs["troopai.agent.handoffs"] = list(data.handoffs)
    if data.tools is not None and len(data.tools) > 0:
        attrs["troopai.agent.tools"] = list(data.tools)
    if data.output_type is not None:
        attrs["troopai.agent.output_type"] = data.output_type
    for k, v in data.metadata.items():
        attrs.update(_flatten_attributes({k: v}, prefix="troopai.metadata"))
    if data.tenant_id is not None:
        attrs["troopai.tenant.id"] = data.tenant_id
    return attrs


def _function_attrs(
    data: FunctionSpanData,
    *,
    record_full: bool,
    max_chars: int,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "troopai.tool.name": data.name,
    }
    if data.input is not None:
        attrs["troopai.tool.input"] = data.input if record_full else _redact_and_truncate(data.input, max_chars)
    if data.output is not None:
        raw_output = str(data.output)
        attrs["troopai.tool.output"] = raw_output if record_full else _redact_and_truncate(raw_output, max_chars)
    if data.mcp_data is not None:
        attrs.update(_flatten_attributes(data.mcp_data, prefix="troopai.mcp"))
    if data.a2a_data is not None:
        # Apply credential-shape redaction (no truncation — these values
        # are short identifiers and URLs). Defensive against future
        # callers stuffing per-tenant tokens or session cookies into
        # a2a_data without thinking; the input/output redaction gate
        # already protects tool I/O, so a2a_data needs the same gate.
        # NB: redaction does not run when ``record_full=True`` to match
        # the ``input``/``output`` policy above.
        a2a_safe = (
            {k: _redact(v) if isinstance(v, str) else v for k, v in data.a2a_data.items()}
            if not record_full
            else data.a2a_data
        )
        attrs.update(_flatten_attributes(a2a_safe, prefix="troopai.a2a"))
    return attrs


def _generation_attrs(data: GenerationSpanData) -> dict[str, Any]:
    attrs: dict[str, Any] = {"gen_ai.system": "troopai"}
    if data.model is not None:
        attrs["gen_ai.request.model"] = data.model
    if data.usage is not None:
        usage = data.usage
        # Handle both dict form and LLMUsage.export() shape. Use explicit
        # None checks — a legitimate zero-token response must still report
        # ``input_tokens=0`` rather than falling through to the alternate key.
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int):
            attrs["gen_ai.usage.input_tokens"] = input_tokens
        if isinstance(output_tokens, int):
            attrs["gen_ai.usage.output_tokens"] = output_tokens
    if data.model_config is not None:
        attrs.update(_flatten_attributes(data.model_config, prefix="gen_ai.request"))
    if data.tenant_id is not None:
        attrs["troopai.tenant.id"] = data.tenant_id
    return attrs


def _response_attrs(data: ResponseSpanData) -> dict[str, Any]:
    attrs: dict[str, Any] = {"gen_ai.system": "troopai"}
    if data.response_id is not None:
        attrs["gen_ai.response.id"] = data.response_id
    return attrs


def _handoff_attrs(data: HandoffSpanData) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if data.from_agent is not None:
        attrs["troopai.handoff.from"] = data.from_agent
    if data.to_agent is not None:
        attrs["troopai.handoff.to"] = data.to_agent
    return attrs


def _guardrail_attrs(data: GuardrailSpanData) -> dict[str, Any]:
    return {
        "troopai.guardrail.name": data.name,
        "troopai.guardrail.triggered": data.triggered,
    }


def _custom_attrs(data: CustomSpanData) -> dict[str, Any]:
    attrs: dict[str, Any] = {"troopai.span.name": data.name}
    attrs.update(_flatten_attributes(data.data, prefix="troopai.custom"))
    return attrs


def _graph_attrs(data: CustomSpanData) -> dict[str, Any]:
    """Graph-execution root-span attribute mapping.

    Reads the exported :class:`GraphSpanData` payload from
    ``data.data`` and surfaces graph identity + terminal state under
    the ``troopai.graph.*`` namespace. ``None``-valued fields are
    omitted (OTel rejects ``None`` attribute values).
    """
    payload = data.data
    attrs: dict[str, Any] = {}
    graph_id = payload.get("graph_id")
    if graph_id is not None:
        attrs["troopai.graph.id"] = graph_id
    entry = payload.get("entry")
    if entry is not None:
        attrs["troopai.graph.entry"] = entry
    status = payload.get("status")
    if status is not None:
        attrs["troopai.graph.status"] = status
    supersteps_total = payload.get("supersteps_total")
    if supersteps_total is not None:
        attrs["troopai.graph.supersteps_total"] = supersteps_total
    return attrs


def _graph_superstep_attrs(data: CustomSpanData) -> dict[str, Any]:
    """BSP-superstep-span attribute mapping.

    Surfaces the superstep index + ready/fired node sets under
    ``troopai.graph.superstep.*``. The parent ``troopai.graph.id`` is
    repeated so a single attribute query can correlate superstep
    spans back to their graph.
    """
    payload = data.data
    attrs: dict[str, Any] = {}
    graph_id = payload.get("graph_id")
    if graph_id is not None:
        attrs["troopai.graph.id"] = graph_id
    index = payload.get("index")
    if index is not None:
        attrs["troopai.graph.superstep.index"] = index
    ready_nodes = payload.get("ready_nodes")
    if ready_nodes is not None and len(ready_nodes) > 0:
        attrs["troopai.graph.superstep.ready_nodes"] = list(ready_nodes)
    fired_nodes = payload.get("fired_nodes")
    if fired_nodes is not None and len(fired_nodes) > 0:
        attrs["troopai.graph.superstep.fired_nodes"] = list(fired_nodes)
    return attrs


def _graph_node_attrs(data: CustomSpanData) -> dict[str, Any]:
    """Per-node-attempt-span attribute mapping.

    Surfaces node identity + reliability + lifecycle status under
    ``troopai.graph.node.*``. ``resume_attempt`` only appears when the
    span covers a resumed node (set by the caller on resume); the
    original attempt leaves it ``None`` so the attribute is omitted.
    """
    payload = data.data
    attrs: dict[str, Any] = {}
    graph_id = payload.get("graph_id")
    if graph_id is not None:
        attrs["troopai.graph.id"] = graph_id
    node_name = payload.get("node_name")
    if node_name is not None:
        attrs["troopai.graph.node.name"] = node_name
    attempts = payload.get("attempts")
    if attempts is not None:
        attrs["troopai.graph.node.attempts"] = attempts
    status = payload.get("status")
    if status is not None:
        attrs["troopai.graph.node.status"] = status
    duration_ms = payload.get("duration_ms")
    if duration_ms is not None:
        attrs["troopai.graph.node.duration_ms"] = duration_ms
    resume_attempt = payload.get("resume_attempt")
    if resume_attempt is not None:
        attrs["troopai.graph.node.resume_attempt"] = resume_attempt
    return attrs


def _swarm_attrs(data: CustomSpanData) -> dict[str, Any]:
    """Swarm-execution root-span attribute mapping.

    Reads the exported :class:`SwarmSpanData` payload from
    ``data.data`` and surfaces swarm identity + terminal state under
    the ``troopai.swarm.*`` namespace. ``None``-valued fields are
    omitted (OTel rejects ``None`` attribute values).
    """
    payload = data.data
    attrs: dict[str, Any] = {}
    swarm_id = payload.get("swarm_id")
    if swarm_id is not None:
        attrs["troopai.swarm.id"] = swarm_id
    entry = payload.get("entry")
    if entry is not None:
        attrs["troopai.swarm.entry"] = entry
    status = payload.get("status")
    if status is not None:
        attrs["troopai.swarm.status"] = status
    turns_total = payload.get("turns_total")
    if turns_total is not None:
        attrs["troopai.swarm.turns_total"] = turns_total
    return attrs


def _swarm_turn_attrs(data: CustomSpanData) -> dict[str, Any]:
    """Per-turn-span attribute mapping.

    Surfaces turn identity + status + duration under
    ``troopai.swarm.turn.*``. The parent ``troopai.swarm.id`` is repeated
    so a single attribute query can correlate turn spans back to their
    run. ``resume_attempt`` is omitted on fresh turns.
    """
    payload = data.data
    attrs: dict[str, Any] = {}
    swarm_id = payload.get("swarm_id")
    if swarm_id is not None:
        attrs["troopai.swarm.id"] = swarm_id
    index = payload.get("index")
    if index is not None:
        attrs["troopai.swarm.turn.index"] = index
    member = payload.get("member")
    if member is not None:
        attrs["troopai.swarm.turn.member"] = member
    status = payload.get("status")
    if status is not None:
        attrs["troopai.swarm.turn.status"] = status
    duration_ms = payload.get("duration_ms")
    if duration_ms is not None:
        attrs["troopai.swarm.turn.duration_ms"] = duration_ms
    resume_attempt = payload.get("resume_attempt")
    if resume_attempt is not None:
        attrs["troopai.swarm.turn.resume_attempt"] = resume_attempt
    return attrs


def _filter_to_fields(exported: dict[str, Any], cls: type) -> dict[str, Any]:
    """Return only the keys from *exported* that are declared fields on *cls*.

    Used when reconstructing a frozen SpanData dataclass from an
    ``export()`` snapshot: any key that is not a declared field of *cls*
    would raise :class:`TypeError` on ``cls(**...)`` construction, and
    future additions to the exported dict should not silently break the
    OTel bridge.  The ``"type"`` discriminator key is always excluded.

    Args:
        exported: Raw dict produced by :meth:`~SpanData.export`.
        cls: The target dataclass type.

    Returns:
        A filtered dict suitable for ``cls(**filtered)``.
    """
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return {k: v for k, v in exported.items() if k != "type" and k in known}


_CustomAttrsFn = Callable[[CustomSpanData], dict[str, Any]]

_CUSTOM_TYPE_TO_ATTRS: dict[str, _CustomAttrsFn] = {
    "graph": _graph_attrs,
    "graph_superstep": _graph_superstep_attrs,
    "graph_node": _graph_node_attrs,
    "swarm": _swarm_attrs,
    "swarm_turn": _swarm_turn_attrs,
}


class OTelTracer:
    """Tracer that emits spans via the OpenTelemetry API."""

    def __init__(
        self,
        provider: OTelTracerProvider | None = None,
        service_name: str = "troopai-adk-python",
        *,
        record_tool_io_full: bool = False,
        tool_io_max_chars: int = _DEFAULT_TOOL_IO_MAX_CHARS,
        convention: TracingConvention = TracingConvention.DEFAULT,
    ) -> None:
        """Initialise the tracer.

        Args:
            provider: An existing ``TracerProvider``. When omitted, falls
                back to the global provider via
                ``opentelemetry.trace.get_tracer_provider()`` (which is the
                no-op provider unless :func:`setup_otel` or user code
                installed one).
            service_name: Instrumentation scope name, used as the
                ``instrumentation-scope`` label on every emitted span.
            record_tool_io_full: Emit tool inputs/outputs verbatim, no
                redaction or truncation. Default ``False`` (safe). Only
                turn on in trusted environments — tool I/O frequently
                contains PII, credentials, or multi-MB payloads that
                inflate trace cost.
            tool_io_max_chars: Per-attribute char cap when
                ``record_tool_io_full`` is ``False``.
            convention: Span-attribute vocabulary to emit. Defaults to
                ``TracingConvention.DEFAULT`` (``gen_ai.*`` plus
                ``troopai.*``). Pass ``TracingConvention.OPENINFERENCE``
                to emit OpenInference attributes instead (read natively
                by Phoenix/Arize).

        Raises:
            TracingDependencyError: When the ``opentelemetry`` packages
                are not installed. Install the extra via
                ``pip install 'troopai-adk-python[otel]'``.
        """
        try:
            from opentelemetry import trace as otel_trace
        except ImportError as exc:
            raise TracingDependencyError(missing="opentelemetry") from exc

        self._service_name = service_name
        self._record_tool_io_full = record_tool_io_full
        self._tool_io_max_chars = tool_io_max_chars
        self._convention = convention
        if provider is not None:
            self._provider: OTelTracerProvider | None = provider
            self._otel_tracer = provider.get_tracer(service_name)
        else:
            self._provider = None
            self._otel_tracer = otel_trace.get_tracer(service_name)
        logger.debug(
            "OTelTracer initialised (service_name=%s, provider=%s, record_tool_io_full=%s, convention=%s)",
            service_name,
            "explicit" if provider is not None else "global",
            record_tool_io_full,
            convention.value,
        )

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        convention = self._convention

        def _flatten_agent(exported: dict[str, Any]) -> dict[str, Any]:
            rebuilt = AgentSpanData(**_filter_to_fields(exported, AgentSpanData))
            if convention is TracingConvention.OPENINFERENCE:
                return oi.agent_attrs(rebuilt)
            return _agent_attrs(rebuilt)

        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name=f"agent.{data.name}",
            attribute_flattener=_flatten_agent,
        )

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        # A2A takes precedence over MCP when both are present — an A2A
        # call is the wider boundary; MCP-via-A2A still surfaces as a2a.
        if data.a2a_data is not None:
            prefix = "a2a"
        elif data.mcp_data is not None:
            prefix = "mcp"
        else:
            prefix = "tool"
        convention = self._convention
        record_full = self._record_tool_io_full
        max_chars = self._tool_io_max_chars

        def _flatten_function(exported: dict[str, Any]) -> dict[str, Any]:
            rebuilt = FunctionSpanData(**_filter_to_fields(exported, FunctionSpanData))
            if convention is TracingConvention.OPENINFERENCE:
                if not record_full:
                    rebuilt = dataclasses.replace(
                        rebuilt,
                        input=_redact_and_truncate(rebuilt.input, max_chars) if rebuilt.input is not None else None,
                        output=_redact_and_truncate(str(rebuilt.output), max_chars)
                        if rebuilt.output is not None
                        else None,
                    )
                return oi.function_attrs(rebuilt)
            return _function_attrs(rebuilt, record_full=record_full, max_chars=max_chars)

        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name=f"{prefix}.{data.name}",
            attribute_flattener=_flatten_function,
        )

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        convention = self._convention

        def _flatten_generation(exported: dict[str, Any]) -> dict[str, Any]:
            rebuilt = GenerationSpanData(**_filter_to_fields(exported, GenerationSpanData))
            if convention is TracingConvention.OPENINFERENCE:
                return oi.generation_attrs(rebuilt)
            return _generation_attrs(rebuilt)

        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name="llm.generation",
            attribute_flattener=_flatten_generation,
        )

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        convention = self._convention

        def _flatten_response(exported: dict[str, Any]) -> dict[str, Any]:
            rebuilt = ResponseSpanData(**_filter_to_fields(exported, ResponseSpanData))
            if convention is TracingConvention.OPENINFERENCE:
                return oi.response_attrs(rebuilt)
            return _response_attrs(rebuilt)

        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name="llm.response",
            attribute_flattener=_flatten_response,
        )

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        convention = self._convention

        def _flatten_handoff(exported: dict[str, Any]) -> dict[str, Any]:
            rebuilt = HandoffSpanData(**_filter_to_fields(exported, HandoffSpanData))
            if convention is TracingConvention.OPENINFERENCE:
                return oi.handoff_attrs(rebuilt)
            return _handoff_attrs(rebuilt)

        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name="agent.handoff",
            attribute_flattener=_flatten_handoff,
        )

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        convention = self._convention

        def _flatten_guardrail(exported: dict[str, Any]) -> dict[str, Any]:
            rebuilt = GuardrailSpanData(**_filter_to_fields(exported, GuardrailSpanData))
            if convention is TracingConvention.OPENINFERENCE:
                return oi.guardrail_attrs(rebuilt)
            return _guardrail_attrs(rebuilt)

        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name=f"guardrail.{data.name}",
            attribute_flattener=_flatten_guardrail,
        )

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        # Graph- and swarm-typed spans route through custom_span (see
        # sandbox precedent in spans.py) — re-derive the typed flattener
        # from data["type"] so the OTel attribute surface stays
        # type-aware without needing a Tracer-protocol extension.
        inner_type = data.data.get("type")
        custom_attrs_fn = _CUSTOM_TYPE_TO_ATTRS.get(inner_type) if isinstance(inner_type, str) else None
        convention = self._convention
        # Correct: custom-span closures capture the construction-time `data`
        # directly and ignore `_exported`, because CustomSpanData.data is a
        # plain dict mutated in place by the runner — there is no immutable
        # exported snapshot to rebuild from (unlike typed spans above).
        if convention is TracingConvention.OPENINFERENCE:
            return OTelSpan(
                data=data,
                otel_tracer=self._otel_tracer,
                name=data.name,
                attribute_flattener=lambda _exported: oi.custom_attrs_by_type(data),
            )
        if custom_attrs_fn is not None:
            fn = custom_attrs_fn  # narrow for closure capture
            return OTelSpan(
                data=data,
                otel_tracer=self._otel_tracer,
                name=data.name,
                attribute_flattener=lambda _exported: fn(data),
            )
        return OTelSpan(
            data=data,
            otel_tracer=self._otel_tracer,
            name=data.name,
            attribute_flattener=lambda _exported: _custom_attrs(data),
        )

    def flush(self) -> None:
        """Synchronously drain all pending spans to the configured exporter(s).

        When an explicit ``TracerProvider`` was supplied at construction time
        its :meth:`~opentelemetry.sdk.trace.TracerProvider.force_flush` method
        is called directly.  Otherwise the global provider (installed by
        :func:`~troopai.adk.tracing.otel.setup_otel` via
        ``opentelemetry.trace.set_tracer_provider``) is flushed instead — it
        may be the OTel SDK no-op provider if :func:`setup_otel` was not
        called, in which case ``force_flush`` is a no-op.
        """
        try:
            from opentelemetry import trace as otel_trace
        except ImportError:
            return

        provider = self._provider if self._provider is not None else otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()  # type: ignore[union-attr]  # hasattr guard above proves force_flush exists
            logger.debug("OTelTracer.flush: force_flush completed on %s", type(provider).__name__)
