"""Agent output schema for structured output validation.

Provides ``AgentOutputSchema`` — a wrapper that accepts any Python type
(Pydantic BaseModel, TypedDict, dataclass, plain types like ``int`` or
``list[str]``) and generates JSON schemas suitable for LLM structured
output, validates responses, and handles type wrapping/unwrapping.

The class supports three usage patterns:

1. **Plain text** — ``None`` or ``str`` output: no schema, model returns
   free-form text.
2. **Pydantic/dict** — ``BaseModel`` or ``dict`` subclasses: JSON schema
   generated directly from the type.
3. **Other types** — ``int``, ``list[str]``, dataclasses, TypedDict, etc.:
   automatically wrapped in a ``{"response": <value>}`` envelope so a
   valid JSON Schema can be generated.  The wrapper is transparent to
   callers — ``validate_json()`` unwraps automatically.

Example::

    from pydantic import BaseModel
    from troopai.adk.schemas import AgentOutputSchema


    class Analysis(BaseModel):
        sentiment: str
        confidence: float


    # From a BaseModel
    schema = AgentOutputSchema(Analysis)
    schema.json_schema()  # -> {"type": "object", "properties": {...}}
    result = schema.validate_json('{"sentiment": "positive", "confidence": 0.95}')
    assert isinstance(result, Analysis)

    # From a plain type (auto-wrapped)
    schema = AgentOutputSchema(list[str])
    schema.json_schema()  # -> wraps in {"response": [...]}
    result = schema.validate_json('{"response": ["a", "b"]}')
    assert result == ["a", "b"]

    # Plain text (no schema)
    schema = AgentOutputSchema(str)
    assert schema.is_plain_text()

References:
    - OpenAI Agents SDK ``AgentOutputSchema``:
      https://github.com/openai/openai-agents-python/blob/main/src/agents/agent_output.py
    - OpenAI structured output guide:
      https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

import copy
import json
import logging
import re
import types
from abc import ABC, abstractmethod
from typing import Any, Literal, Union, get_args, get_origin, override

from pydantic import BaseModel, TypeAdapter, ValidationError

from troopai.adk.schemas.utils import SchemaEnforcement, enforce_schema

# Upper bound on the raw JSON payload we are willing to hand to
# ``json.loads``. A cooperating-but-pathological LLM could emit a
# deeply-nested or huge JSON blob that stackoverflows or OOMs the
# validator. 256 KiB is generous for typical structured outputs
# (Claude's max output window at ~32 KiB today) while still bounded.
# The guard runs BEFORE ``json.loads`` so the attacker pays the cost
# of the size check, not the deserialization.
MAX_SCHEMA_BYTES: int = 256 * 1024

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AgentOutputSchemaBase(ABC):
    """Abstract base class for agent output schema implementations.

    Subclass this to provide custom JSON schema generation and
    validation logic.  The default implementation is
    :class:`AgentOutputSchema`.

    Methods:
        is_plain_text: Whether the output is unstructured text.
        json_schema: Returns the JSON schema for structured output.
        validate_json: Parses and validates a JSON string.
        is_strict_json_schema: Whether strict mode is enabled.
        name: String representation of the output type.
    """

    @abstractmethod
    def is_plain_text(self) -> bool:
        """Whether the output type represents plain text (no schema).

        Returns:
            ``True`` if the output is ``None`` or ``str`` — the model
            should return free-form text rather than structured JSON.
        """

    @abstractmethod
    def json_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the structured output type.

        Returns:
            A JSON Schema dict suitable for use in a ``response_format``
            parameter when calling an LLM API.

        Raises:
            ValueError: If called when :meth:`is_plain_text` is ``True``.
        """

    @abstractmethod
    def validate_json(self, json_str: str) -> Any:
        """Parse and validate a JSON string against the output schema.

        Args:
            json_str: The raw JSON string from the LLM response.

        Returns:
            The validated Python object matching the output type.

        Raises:
            ValueError: If the JSON is invalid or does not match the schema.
        """

    @abstractmethod
    def is_strict_json_schema(self) -> bool:
        """Whether the schema uses strict mode.

        Returns:
            ``True`` if the schema enforces strict JSON Schema
            constraints (``additionalProperties: false``, all properties
            required, etc.).
        """

    @abstractmethod
    def name(self) -> str:
        """String representation of the output type.

        Returns:
            A human-readable name for the output type, useful for
            logging and error messages.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_union(output_type: object) -> bool:
    """Check whether *output_type* is a union type.

    Recognizes both spellings: ``typing.Union[A, B]`` and the PEP 604
    form ``A | B`` (a ``types.UnionType``). ``get_origin`` reports a
    distinct origin for each, so both must be checked.

    Args:
        output_type: The type to inspect.

    Returns:
        ``True`` if *output_type* is a union of two or more types.
    """
    return get_origin(output_type) in (Union, types.UnionType)


def _is_subclass_of_base_model_or_dict(output_type: type[Any]) -> bool:
    """Check whether *output_type* is a BaseModel or dict subclass.

    These types can produce valid JSON schemas directly without needing
    to be wrapped in an envelope.

    Args:
        output_type: The type to inspect.

    Returns:
        ``True`` if *output_type* is a subclass of ``BaseModel`` or
        ``dict``, or if its ``__origin__`` is ``dict``.
    """
    if isinstance(output_type, type) and issubclass(output_type, (BaseModel, dict)):
        return True
    # Handle generic dict types like dict[str, Any]
    return get_origin(output_type) is dict


def _is_typed_dict(output_type: type[Any]) -> bool:
    """Check whether *output_type* is a TypedDict.

    Args:
        output_type: The type to inspect.

    Returns:
        ``True`` if *output_type* is a TypedDict class.
    """
    return (
        isinstance(output_type, type)
        and issubclass(output_type, dict)
        and hasattr(output_type, "__annotations__")
        and hasattr(output_type, "__required_keys__")
    )


def _needs_wrapping(output_type: type[Any]) -> bool:
    """Determine whether *output_type* needs to be wrapped.

    Types that are NOT BaseModel, dict, or TypedDict subclasses need
    to be wrapped in a ``{"response": <value>}`` envelope to produce
    a valid JSON Schema.

    Args:
        output_type: The type to inspect.

    Returns:
        ``True`` if the type needs wrapping.
    """
    if _is_subclass_of_base_model_or_dict(output_type):
        return False
    return not _is_typed_dict(output_type)


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class AgentOutputSchema(AgentOutputSchemaBase):
    """Manages JSON schema generation and validation for agent outputs.

    Accepts any Python type and provides:

    - JSON schema generation via Pydantic's ``TypeAdapter``
    - Configurable schema enforcement (none, normalized, or strict)
    - Automatic wrapping of non-BaseModel types in an envelope
    - Transparent unwrapping during validation

    Attributes:
        output_schema: The original output type as provided to the
            constructor.

    Example::

        from pydantic import BaseModel
        from troopai.adk.schemas import SchemaEnforcement


        class Result(BaseModel):
            answer: str
            confidence: float


        # Direct usage (strict by default)
        schema = AgentOutputSchema(Result)
        validated = schema.validate_json('{"answer": "42", "confidence": 0.99}')

        # Normalized mode (sensible defaults without strict constraints)
        schema = AgentOutputSchema(Result, schema_enforcement=SchemaEnforcement.NORMALIZED)

        # No enforcement
        schema = AgentOutputSchema(Result, schema_enforcement=SchemaEnforcement.NONE)

        # With Agent
        agent = Agent(output_schema=Result)  # auto-wrapped, strict by default
        agent = Agent(
            output_schema=AgentOutputSchema(
                Result,
                schema_enforcement=SchemaEnforcement.NORMALIZED,
            )
        )

    References:
        - OpenAI Agents SDK ``AgentOutputSchema``:
          https://github.com/openai/openai-agents-python/blob/main/src/agents/agent_output.py
    """

    def __init__(
        self,
        output_schema: type[Any] | None,
        schema_enforcement: SchemaEnforcement = SchemaEnforcement.STRICT,
    ) -> None:
        """Initialize the output schema wrapper.

        Args:
            output_schema: The Python type for structured output.

                - ``None`` or ``str``: Plain text output (no schema).
                - ``BaseModel`` subclass: Direct JSON schema generation.
                - ``dict`` / ``TypedDict``: Direct JSON schema generation.
                - Any other type (``int``, ``list[str]``, dataclass, etc.):
                  Automatically wrapped in ``{"response": <value>}``.

            schema_enforcement: Controls how the generated JSON schema is
                processed before being sent to the LLM.  Defaults to
                ``SchemaEnforcement.STRICT``.

                - ``STRICT``: Full OpenAI strict-mode compliance via
                  :func:`ensure_strict_schema`.
                - ``COMPACT``: Strict-mode compliance plus metadata
                  stripping for cost optimization.
                - ``NORMALIZED``: Provider-agnostic normalization via
                  :func:`normalize_schema`.
                - ``NONE``: Raw schema as-is, no transformation.
        """
        self.output_schema: type[Any] | None = output_schema
        self._schema_enforcement: SchemaEnforcement = schema_enforcement

        # --- Plain text: no schema needed ---
        if output_schema is None or output_schema is str:
            self._type_adapter: TypeAdapter[Any] | None = None
            self._is_wrapped: bool = False
            self._json_schema_dict: dict[str, Any] = {}
            return

        # --- Determine if wrapping is needed ---
        if _needs_wrapping(output_schema):
            from typing import TypedDict as _TypedDict

            # Create a wrapper TypedDict with a "response" key.
            # Functional form is required: the value type is a runtime variable.
            _WrappedOutput = _TypedDict(  # noqa: UP013
                "_WrappedOutput",
                {"response": output_schema},  # type: ignore[valid-type]
            )
            self._is_wrapped = True
            self._type_adapter = TypeAdapter(_WrappedOutput)
        else:
            self._is_wrapped = False
            self._type_adapter = TypeAdapter(output_schema)

        # --- Generate and enforce JSON schema ---
        raw_schema = self._type_adapter.json_schema()
        self._json_schema_dict = enforce_schema(
            copy.deepcopy(raw_schema),
            schema_enforcement,
        )

    @override
    def is_plain_text(self) -> bool:
        """Whether the output type represents plain text.

        Returns:
            ``True`` if ``output_schema`` is ``None`` or ``str``.
        """
        return self.output_schema is None or self.output_schema is str

    @override
    def json_schema(self) -> dict[str, Any]:
        """Return the JSON schema for structured output.

        Returns a deep copy so a caller that mutates the schema in place
        (e.g. a provider converter applying its own transforms) cannot
        corrupt the cached dict for every subsequent call.

        Returns:
            A JSON Schema dict.  For wrapped types, this includes the
            ``{"response": ...}`` envelope.

        Raises:
            ValueError: If the output type is plain text.
        """
        if self.is_plain_text():
            raise ValueError(f"Cannot get JSON schema for plain text output type: {self.output_schema}")
        return copy.deepcopy(self._json_schema_dict)

    @override
    def validate_json(self, json_str: str) -> Any:
        """Parse and validate a JSON string against this schema.

        For wrapped types, the ``"response"`` key is automatically
        unwrapped so callers receive the inner value directly.

        Args:
            json_str: The raw JSON string from the LLM response.

        Returns:
            The validated Python object.  For ``BaseModel`` types, this
            is a model instance.  For wrapped types (``int``, ``list``,
            etc.), this is the unwrapped value.

        Raises:
            ValueError: If the JSON is invalid, does not match, or
                exceeds :data:`MAX_SCHEMA_BYTES` (256 KiB). The
                size guard runs BEFORE ``json.loads`` so a
                pathological payload cannot trigger deserialization
                cost; it is a defence-in-depth bound on top of any
                provider-side output-token limit.
        """
        if self._type_adapter is None:
            raise ValueError("Cannot validate JSON for plain text output type")

        # Encode to bytes for an accurate byte count — ``len(str)``
        # counts codepoints, not bytes, and a UTF-8 payload can be up
        # to 4× larger than its string-length. We measure the wire
        # size, not the logical size.
        raw_bytes = len(json_str.encode("utf-8"))
        if raw_bytes > MAX_SCHEMA_BYTES:
            raise ValueError(
                f"LLM response exceeds validate_json size cap: "
                f"{raw_bytes} bytes > {MAX_SCHEMA_BYTES} bytes ({MAX_SCHEMA_BYTES // 1024} KiB). "
                "Reject the response rather than attempting deserialization."
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}") from e

        try:
            validated = self._type_adapter.validate_python(data)
        except ValidationError as first_error:
            # Attempt repair: some providers (e.g. Gemini) produce
            # "Refund" instead of "refund" for Literal discriminators.
            repaired = self._try_repair_discriminators(data)
            if repaired is not data:
                try:
                    validated = self._type_adapter.validate_python(repaired)
                    logger.debug(
                        "Repaired discriminator values for schema %s",
                        self.name(),
                    )
                except Exception as repair_error:
                    raise ValueError(
                        f"LLM response does not match output schema {self.name()}: {first_error}"
                    ) from repair_error
            else:
                raise ValueError(
                    f"LLM response does not match output schema {self.name()}: {first_error}"
                ) from first_error

        # Unwrap if we wrapped the type
        if self._is_wrapped:
            if isinstance(validated, dict) and "response" in validated:
                return validated["response"]
            raise ValueError(
                f"Expected wrapped response with 'response' key for schema {self.name()}, "
                f"got {type(validated).__name__}"
            )

        return validated

    @property
    def schema_enforcement(self) -> SchemaEnforcement:
        """The configured enforcement level (``none``/``normalized``/``strict``/``compact``).

        Distinct from :meth:`is_strict_json_schema`, which collapses
        ``STRICT`` and ``COMPACT`` to a single boolean; this preserves the
        exact level so it can be round-tripped (e.g. by ``dump_agent``).
        """
        return self._schema_enforcement

    @override
    def is_strict_json_schema(self) -> bool:
        """Whether strict JSON Schema mode is enabled.

        Returns:
            ``True`` if ``schema_enforcement`` is
            ``SchemaEnforcement.STRICT`` or ``SchemaEnforcement.COMPACT``.
        """
        return self._schema_enforcement in (
            SchemaEnforcement.STRICT,
            SchemaEnforcement.COMPACT,
        )

    @override
    def name(self) -> str:
        """String representation of the output type.

        For Union types, produces a sanitized short name like
        ``RefundIntent_BillingIntent_Respond`` instead of the raw
        ``typing.Union[module.RefundIntent, ...]`` repr.

        Returns:
            The type's ``__name__`` if available, a sanitized Union
            name for Union types, or ``str()`` as fallback.
        """
        if self.output_schema is None:
            return "str"
        if _is_union(self.output_schema):
            args = get_args(self.output_schema)
            parts = [getattr(a, "__name__", str(a)) for a in args]
            joined = "_".join(parts)
            # Sanitize: keep only alphanumeric + underscores
            return re.sub(r"[^a-zA-Z0-9_]", "", joined)
        if hasattr(self.output_schema, "__name__"):
            return self.output_schema.__name__
        return str(self.output_schema)

    def _try_repair_discriminators(self, data: Any) -> Any:
        """Attempt to repair Literal discriminator values in LLM output.

        Some providers (e.g. Gemini) don't enforce ``Literal``/``const``
        values strictly, producing ``"Refund"`` instead of ``"refund"``.
        This method normalizes such values by matching case-insensitively
        against the expected Literal values from the Union members.

        Args:
            data: The parsed JSON data (dict or wrapped dict).

        Returns:
            The repaired data if changes were made, or the original
            *data* reference unchanged (identity check: ``repaired is data``
            means no repair was possible).
        """
        if self.output_schema is None:
            return data

        # Handle wrapped types — repair inside the "response" envelope
        if self._is_wrapped:
            if isinstance(data, dict) and "response" in data:
                inner = data["response"]
                repaired_inner = self._repair_union_values(inner)
                if repaired_inner is not inner:
                    return {**data, "response": repaired_inner}
            return data

        return self._repair_union_values(data)

    def _repair_union_values(self, data: Any) -> Any:
        """Normalize discriminator field values against Union member Literals.

        Walks the Union members, collects all ``Literal`` field values,
        and replaces any case-insensitive matches in *data* with the
        correct casing.

        Args:
            data: The inner data dict to repair.

        Returns:
            A new dict with corrected values, or the original reference
            if no repairs were needed.
        """
        if not isinstance(data, dict):
            return data

        output_type = self.output_schema
        if not _is_union(output_type):
            return data

        members = get_args(output_type)

        # Build map: field_name → {lowercase_value: correct_value}
        const_map: dict[str, dict[str, str]] = {}
        for member in members:
            if not hasattr(member, "model_fields"):
                continue
            for field_name, field_info in member.model_fields.items():
                annotation = field_info.annotation
                if get_origin(annotation) is Literal:
                    for lit_val in get_args(annotation):
                        if isinstance(lit_val, str):
                            const_map.setdefault(field_name, {})
                            const_map[field_name][lit_val.lower()] = lit_val

        if len(const_map) == 0:
            return data

        repaired = dict(data)
        changed = False
        for field_name, lower_map in const_map.items():
            if field_name in repaired:
                val = repaired[field_name]
                if isinstance(val, str) and val.lower() in lower_map:
                    correct = lower_map[val.lower()]
                    if correct != val:
                        repaired[field_name] = correct
                        changed = True

        return repaired if changed else data

    @override
    def __repr__(self) -> str:
        return (
            f"AgentOutputSchema("
            f"output_schema={self.name()}, "
            f"enforcement={self._schema_enforcement.value}, "
            f"wrapped={self._is_wrapped})"
        )
