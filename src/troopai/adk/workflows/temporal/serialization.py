"""Temporal DataConverter factory for TroopAI ADK.

Attempts to use the Pydantic-aware converter from
``temporalio.contrib.pydantic`` so that Pydantic models survive
workflow replay without extra encoding steps.  Falls back to the
default Temporal DataConverter when the contrib package is absent.

Install the ``temporal`` optional extra to get both converters::

    pip install "troopai-adk-python[temporal]"
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any, override

from troopai.adk.schemas.agent_output_schema import AgentOutputSchemaBase

if TYPE_CHECKING:
    from temporalio.converter import DataConverter

    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.tools import FunctionTool, Tool

logger = logging.getLogger(__name__)

# Sentinel key marking a serialized ``httpx.Timeout`` (vs a plain float timeout).
_HTTPX_TIMEOUT_KEY = "__httpx_timeout__"


def build_troopai_data_converter() -> DataConverter:
    """Return a Temporal :class:`~temporalio.converter.DataConverter` for TroopAI workflows.

    Tries to import the Pydantic-aware converter supplied by
    ``temporalio.contrib.pydantic``.  If that module is unavailable a
    shallow copy of :attr:`DataConverter.default <temporalio.converter.DataConverter.default>`
    is returned instead so callers always get a fully-configured converter.

    Returns:
        A :class:`~temporalio.converter.DataConverter` instance ready to
        pass to a Temporal ``Worker`` or ``Client``.
    """
    try:
        from temporalio.contrib.pydantic import pydantic_data_converter

        logger.debug("Using temporalio.contrib.pydantic data converter")
        return pydantic_data_converter
    except ImportError:
        from temporalio.converter import DataConverter

        logger.debug("temporalio.contrib.pydantic unavailable; falling back to DataConverter.default")
        return dataclasses.replace(DataConverter.default)


def config_to_json_dict(config: LLMConfig) -> dict[str, Any]:
    """Serialize an :class:`LLMConfig` to a JSON-safe dict for activity transport.

    Most fields are JSON-native, but two are not and would make
    ``json.dumps`` raise (crashing the durable run at its first LLM turn):

    - ``retry_policy`` is an ``LLMRetryPolicy`` dataclass whose ``retry_on``
      is a ``frozenset`` — flattened to a plain dict with ``retry_on`` as a
      sorted list.
    - ``timeout`` may be an ``httpx.Timeout`` — flattened to its four
      components under a sentinel key. A plain float timeout passes through.

    :func:`config_from_json_dict` is the inverse for every non-``None`` field
    (``to_json_dict`` drops ``None`` fields, which all default back to ``None``).
    """
    out = config.to_json_dict()
    retry_policy = out.get("retry_policy")
    if retry_policy is not None:
        rp = dataclasses.asdict(retry_policy)
        if rp.get("retry_on") is not None:
            rp["retry_on"] = sorted(rp["retry_on"])
        out["retry_policy"] = rp
    timeout = out.get("timeout")
    if timeout is not None and not isinstance(timeout, (int, float)):
        out["timeout"] = {
            _HTTPX_TIMEOUT_KEY: {
                "connect": timeout.connect,
                "read": timeout.read,
                "write": timeout.write,
                "pool": timeout.pool,
            }
        }
    return out


def config_from_json_dict(data: dict[str, Any]) -> LLMConfig:
    """Reconstruct an :class:`LLMConfig` from :func:`config_to_json_dict` output.

    Rebuilds the ``retry_policy`` dataclass (``retry_on`` back to a frozenset)
    and the ``httpx.Timeout`` from its components, so the durable activity
    reconstructs the exact config the workflow sent.
    """
    from troopai.adk.llms.llm_config import LLMConfig

    fields = dict(data)
    retry_policy = fields.get("retry_policy")
    if isinstance(retry_policy, dict):
        from troopai.adk.types.llms.retry_policy import LLMRetryPolicy

        rp = dict(retry_policy)
        if rp.get("retry_on") is not None:
            rp["retry_on"] = frozenset(rp["retry_on"])
        fields["retry_policy"] = LLMRetryPolicy(**rp)
    timeout = fields.get("timeout")
    if isinstance(timeout, dict) and _HTTPX_TIMEOUT_KEY in timeout:
        import httpx

        comp = timeout[_HTTPX_TIMEOUT_KEY]
        fields["timeout"] = httpx.Timeout(
            connect=comp["connect"],
            read=comp["read"],
            write=comp["write"],
            pool=comp["pool"],
        )
    return LLMConfig(**fields)


# ---------------------------------------------------------------------------
# Tool + output-schema transport across the activity boundary
# ---------------------------------------------------------------------------
#
# The durable activity needs the LLM-facing *definitions* of the agent's tools
# (so the model can choose which to call) and its structured-output schema (so
# the model is constrained). The executables themselves never cross the
# boundary — each tool call is journaled as its own activity back in the
# workflow — so only name/description/parameter-schema are serialized.


def tool_to_json_dict(tool: Tool) -> dict[str, Any] | None:
    """Serialize a tool to the JSON-safe definition the model needs to see.

    Only the LLM-facing fields survive: name, description, parameter JSON
    schema, and the schema-enforcement level. The executable (``on_invoke``)
    is intentionally dropped — the activity never runs tools.

    Args:
        tool: The tool to serialize.

    Returns:
        A JSON-safe dict, or ``None`` for a tool type whose definition cannot
        yet be reconstructed across the activity boundary (provider-hosted
        tools). The caller logs and skips ``None`` rather than crashing.
    """
    from troopai.adk.schemas.utils import SchemaEnforcement, normalize_schema
    from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
    from troopai.adk.tools.function_tool import FunctionTool

    if isinstance(tool, FunctionTool):
        return {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.get_json_schema(),
            "schema_enforcement": tool.schema_enforcement.value,
        }
    if isinstance(tool, ExecutableBuiltinTool):
        from pydantic import BaseModel

        raw = (
            tool.schema.model_json_schema()
            if isinstance(tool.schema, type) and issubclass(tool.schema, BaseModel)
            else tool.schema
        )
        return {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": normalize_schema(raw),
            "schema_enforcement": SchemaEnforcement.NORMALIZED.value,
        }
    return None


def tool_from_json_dict(data: dict[str, Any]) -> FunctionTool:
    """Reconstruct a definition-only :class:`FunctionTool` from :func:`tool_to_json_dict`.

    The result carries no ``on_invoke`` — it exists solely so the LLM call in
    the activity emits the correct tool definitions. Both ``FunctionTool`` and
    ``ExecutableBuiltinTool`` round-trip here as function-type definitions,
    which is exactly what the model sees for either.

    Args:
        data: One entry produced by :func:`tool_to_json_dict`.

    Returns:
        A ``FunctionTool`` with name, description, parameter schema, and
        enforcement populated.
    """
    from troopai.adk.schemas.utils import SchemaEnforcement
    from troopai.adk.tools.function_tool import FunctionTool

    return FunctionTool(
        name=data["name"],
        description=data.get("description") or None,
        schema=data.get("parameters", {}),
        schema_enforcement=SchemaEnforcement(data.get("schema_enforcement", SchemaEnforcement.NORMALIZED.value)),
    )


@dataclasses.dataclass(frozen=True)
class ForwardedOutputSchema(AgentOutputSchemaBase):
    """An ``AgentOutputSchema`` reconstructed from a serialized JSON schema.

    Carries enough for the durable activity to constrain the model (the JSON
    schema + strict flag). Validation degrades to a plain JSON parse because
    the originating Python type does not cross the activity boundary — the
    provider still enforces the schema, and any typed validation happens back
    in the workflow which holds the real output type.

    Attributes:
        schema_dict: The JSON schema the model is constrained to.
        strict: Whether the schema uses strict mode.
        type_name: Human-readable name of the original output type.
    """

    schema_dict: dict[str, Any]
    strict: bool
    type_name: str

    @override
    def is_plain_text(self) -> bool:
        """Always ``False`` — a schema is only forwarded for structured output."""
        return False

    @override
    def json_schema(self) -> dict[str, Any]:
        """Return the forwarded JSON schema."""
        return self.schema_dict

    @override
    def validate_json(self, json_str: str) -> Any:
        """Parse the model's JSON output (provider already enforced the schema)."""
        return json.loads(json_str)

    @override
    def is_strict_json_schema(self) -> bool:
        """Return whether the forwarded schema uses strict mode."""
        return self.strict

    @override
    def name(self) -> str:
        """Return the original output type's name."""
        return self.type_name


def output_schema_to_json_dict(output_schema: AgentOutputSchemaBase) -> dict[str, Any]:
    """Serialize a structured-output schema for activity transport.

    Args:
        output_schema: A non-plain-text output schema.

    Returns:
        A JSON-safe dict with the schema, strict flag, and type name.
    """
    return {
        "json_schema": output_schema.json_schema(),
        "is_strict": output_schema.is_strict_json_schema(),
        "name": output_schema.name(),
    }


def output_schema_from_json_dict(data: dict[str, Any]) -> AgentOutputSchemaBase:
    """Reconstruct a :class:`ForwardedOutputSchema` from :func:`output_schema_to_json_dict`."""
    return ForwardedOutputSchema(
        schema_dict=data["json_schema"],
        strict=data.get("is_strict", False),
        type_name=data.get("name", "output"),
    )
