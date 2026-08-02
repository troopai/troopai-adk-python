"""Restate handler for durable LLM invocation.

Provides :func:`invoke_model_handler` — an async function intended to be
registered as a Restate handler that executes an LLM call inside a durable
context.

Reuses :func:`~troopai.adk.workflows.temporal.activity.get_model` from the
Temporal activity module so that both engines share a single model registry.
Register models once with :func:`~troopai.adk.workflows.temporal.activity.register_model`
and they are available to both the Temporal and Restate backends::

    from troopai.adk.workflows.temporal.activity import register_model
    from troopai.adk.llms import LiteLLM

    register_model("gpt-4o", LiteLLM(model="gpt-4o"))

References:
    Restate Python SDK handler docs:
    https://docs.restate.dev/develop/python/overview#handlers
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.types.responses.llm_response import LLMResponse
from troopai.adk.workflows.temporal.activity import get_model

if TYPE_CHECKING:
    from troopai.adk.schemas.agent_output_schema import AgentOutputSchemaBase
    from troopai.adk.tools import Tool

logger = logging.getLogger(__name__)


async def invoke_model_handler(
    ctx: Any,  # noqa: ARG001  # Restate handler convention: ctx is always the first param
    model_name: str,
    messages: list[Any],
    config: dict[str, Any] | None,
    tools_json: str = "",
    output_schema_json: str = "",
) -> dict[str, Any]:
    """Restate handler that executes a durable LLM call.

    Looks up the LLM from the shared model registry, builds an
    :class:`~troopai.adk.llms.llm_config.LLMConfig` from *config* when
    provided, delegates to :meth:`~troopai.adk.llms.llm.LLM.acomplete`,
    and returns the response serialized via :func:`dataclasses.asdict`.

    The handler is intentionally not decorated with ``@restate.handler``
    here so that callers control registration (service class, decorator
    chain, or factory pattern).

    Args:
        ctx: The active Restate handler context.  Not used directly in
            this function but required by the Restate handler signature
            convention.
        model_name: Registry key to look up the
            :class:`~troopai.adk.llms.llm.LLM` via
            :func:`~troopai.adk.workflows.temporal.activity.get_model`.
        messages: Conversation messages as a list of provider-agnostic
            content items (already deserialized from JSON by the caller
            or Restate's serde layer).
        config: Optional LLM configuration fields dict.  When not
            ``None``, passed as keyword arguments to
            :class:`~troopai.adk.llms.llm_config.LLMConfig`.
        tools_json: JSON-serialized list of tool schema dicts produced by
            the Temporal serialization helpers.  Empty string means no
            tools — the LLM receives an unconstrained turn.  Forwarding
            tools is essential: without them a ``tool_choice="required"``
            turn loops forever and no tools are ever called.
        output_schema_json: JSON-serialized structured-output schema dict.
            Empty string means no schema — the model is unconstrained.

    Returns:
        The LLM response serialized via :func:`dataclasses.asdict`,
        suitable for JSON transport through the Restate journal.

    Raises:
        KeyError: When *model_name* is not in the registry.
        TypeError: When the LLM returns a type other than
            :class:`~troopai.adk.types.responses.llm_response.LLMResponse`
            (e.g. a streaming iterator).

    References:
        Restate Python SDK:
        https://docs.restate.dev/develop/python/overview
    """
    logger.info(
        "invoke_model_handler started: model=%r",
        model_name,
    )

    llm = get_model(model_name)
    llm_config = LLMConfig(**config) if config is not None else None

    # Reconstruct tool definitions and output schema from their JSON forms.
    # Using the same Temporal serialization helpers keeps the two backends
    # consistent and avoids duplicating the reconstruction logic.
    tools: list[Tool] | None = None
    if tools_json:
        from troopai.adk.workflows.temporal.serialization import tool_from_json_dict

        tool_dicts: list[dict[str, Any]] = json.loads(tools_json)
        if tool_dicts:
            tools = [tool_from_json_dict(d) for d in tool_dicts]

    output_schema: AgentOutputSchemaBase | None = None
    if output_schema_json:
        from troopai.adk.workflows.temporal.serialization import output_schema_from_json_dict

        output_schema = output_schema_from_json_dict(json.loads(output_schema_json))

    response = await llm.acomplete(
        messages=messages,
        llm_config=llm_config,
        tools=tools,
        output_schema=output_schema,
    )

    if not isinstance(response, LLMResponse):
        raise TypeError(
            f"invoke_model_handler expected LLMResponse from model={model_name!r} but got {type(response).__name__}"
        )

    logger.debug(
        "invoke_model_handler completed: model=%r finish_reason=%r",
        model_name,
        response.finish_reason,
    )
    # Nullify timestamp before serialization: datetime is not JSON-serializable
    # with Restate's default serde, and the response is reconstructed with
    # timestamp=None on the caller side anyway.
    response_for_wire = dataclasses.replace(response, timestamp=None)
    return dataclasses.asdict(response_for_wire)
