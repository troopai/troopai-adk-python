"""Temporal activity for durable LLM invocation.

Provides :func:`invoke_model_activity` — a Temporal
:func:`~temporalio.activity.defn`-decorated coroutine that executes an LLM
call inside a durable activity boundary with automatic heartbeating.

Also exposes a module-level model registry so that worker processes can
register :class:`~troopai.adk.llms.llm.LLM` instances by name before
activities execute::

    from troopai.adk.workflows.temporal.activity import register_model
    from troopai.adk.llms import LiteLLM

    register_model("gpt-4o", LiteLLM(model="gpt-4o"))

The registry and :class:`ModelActivityInput` are importable without
``temporalio`` installed.  The :func:`invoke_model_activity` function is
registered only when the ``temporalio`` package is present.

References:
    Temporal Python SDK activity docs:
    https://docs.temporal.io/develop/python/core-application#develop-activities
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.llms.llm import LLM

if TYPE_CHECKING:
    from troopai.adk.schemas.agent_output_schema import AgentOutputSchemaBase
    from troopai.adk.tools import Tool

logger = logging.getLogger(__name__)

# ==================================================================
# Module-level model registry
# ==================================================================

_MODEL_REGISTRY: dict[str, LLM] = {}
"""Registry mapping model names to :class:`~troopai.adk.llms.llm.LLM` instances.

Populated by :func:`register_model` before workers start.  Private to this
module — consumers use :func:`register_model` and :func:`get_model`.
"""


def register_model(name: str, llm: LLM) -> None:
    """Register an :class:`~troopai.adk.llms.llm.LLM` instance by name.

    The name is used as the registry key in :class:`ModelActivityInput`.
    Registering the same name twice overwrites the previous entry.

    Args:
        name: Registry key — matches :attr:`ModelActivityInput.model_name`.
        llm: The :class:`~troopai.adk.llms.llm.LLM` instance to register.
    """
    _MODEL_REGISTRY[name] = llm
    logger.info("Registered model %r in activity registry", name)


def get_model(name: str) -> LLM:
    """Return the :class:`~troopai.adk.llms.llm.LLM` registered under *name*.

    Args:
        name: Registry key to look up.

    Returns:
        The registered :class:`~troopai.adk.llms.llm.LLM` instance.

    Raises:
        KeyError: When *name* is not in the registry.  The error message
            lists all currently registered model names.
    """
    llm = _MODEL_REGISTRY.get(name)
    if llm is None:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(
            f"No model registered under {name!r}. "
            f"Available models: [{available}]. "
            "Call register_model() before starting the worker."
        )
    return llm


# ==================================================================
# Activity input dataclass
# ==================================================================


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelActivityInput:
    """Serialized input for :func:`invoke_model_activity`.

    All fields that carry complex types (messages, tools, config) are
    pre-serialized to JSON strings so that Temporal's DataConverter can
    round-trip them without a custom payload codec.

    Attributes:
        model_name: Registry key used to look up the
            :class:`~troopai.adk.llms.llm.LLM` via :func:`get_model`.
        messages_json: JSON-serialized conversation messages
            (``list[LLMInputContentItem]`` serialized with
            ``json.dumps``).
        tools_json: JSON-serialized tool schemas
            (``list[dict]`` serialized with ``json.dumps``).
        config_json: JSON-serialized
            :class:`~troopai.adk.llms.llm_config.LLMConfig` fields
            produced by :meth:`~troopai.adk.llms.llm_config.LLMConfig.to_json_dict`.
        output_schema_json: Optional JSON-serialized output schema.
            Empty string means no structured output.
    """

    model_name: str
    """Registry key for the LLM instance to invoke."""

    messages_json: str
    """JSON-serialized list of conversation message items."""

    tools_json: str = ""
    """JSON-serialized list of tool schema dicts.  Empty string means no tools.

    Defaulted so an activity input serialized before this field existed (an
    in-flight durable history) still deserializes on replay — the activity
    treats an empty string as "no tools".
    """

    config_json: str
    """JSON-serialized LLMConfig fields dict."""

    output_schema_json: str = ""
    """JSON-serialized output schema.  Empty string means no schema."""


# ==================================================================
# Temporal activity (registered only when temporalio is available)
# ==================================================================

try:
    from temporalio import activity

    @activity.defn
    async def invoke_model_activity(inp: ModelActivityInput) -> dict[str, Any]:
        """Temporal activity that executes a durable LLM call.

        Looks up the LLM from the module registry, deserializes all inputs from
        JSON, fires a background heartbeat task at half the configured
        ``heartbeat_timeout`` interval, then delegates to
        :meth:`~troopai.adk.llms.llm.LLM.acomplete`.

        Heartbeating keeps the Temporal server informed that the worker is still
        alive during long-running LLM calls.  The task is always cancelled in
        the ``finally`` block to avoid resource leaks.

        Args:
            inp: Serialized activity input containing model name, messages,
                tools, and config.

        Returns:
            The LLM response serialized via
            :func:`dataclasses.asdict`, suitable for JSON transport.

        References:
            Temporal heartbeat docs:
            https://docs.temporal.io/develop/python/core-application#heartbeat-an-activity
        """
        logger.info(
            "invoke_model_activity started: model=%r attempt=%d",
            inp.model_name,
            activity.info().attempt,
        )

        llm = get_model(inp.model_name)

        from troopai.adk.workflows.temporal.serialization import (
            config_from_json_dict,
            output_schema_from_json_dict,
            tool_from_json_dict,
        )

        messages: list[Any] = json.loads(inp.messages_json)
        config_fields: dict[str, Any] = json.loads(inp.config_json)
        # Mirror of config_to_json_dict on the workflow side: rebuilds the
        # retry_policy frozenset + httpx.Timeout the JSON form flattened.
        llm_config = config_from_json_dict(config_fields)

        # Reconstruct the LLM-facing tool definitions + structured-output schema
        # the workflow serialized. Forwarding these is essential: without tools a
        # tool_choice="required" turn loops forever, and without the schema the
        # model is unconstrained.
        tools: list[Tool] | None = None
        tool_dicts: list[dict[str, Any]] = json.loads(inp.tools_json) if len(inp.tools_json) > 0 else []
        if len(tool_dicts) > 0:
            tools = [tool_from_json_dict(d) for d in tool_dicts]

        output_schema: AgentOutputSchemaBase | None = None
        if len(inp.output_schema_json) > 0:
            output_schema = output_schema_from_json_dict(json.loads(inp.output_schema_json))

        info = activity.info()
        heartbeat_task: asyncio.Task[None] | None = None

        stop = asyncio.Event()

        if info.heartbeat_timeout is not None:
            interval_seconds = info.heartbeat_timeout.total_seconds() / 2

            async def _heartbeat_loop() -> None:
                while not stop.is_set():
                    await asyncio.sleep(interval_seconds)
                    if not stop.is_set():
                        activity.heartbeat()
                        logger.debug(
                            "invoke_model_activity heartbeat sent: model=%r",
                            inp.model_name,
                        )

            heartbeat_task = asyncio.create_task(_heartbeat_loop())

        try:
            response = await llm.acomplete(
                messages=messages,
                llm_config=llm_config,
                tools=tools,
                output_schema=output_schema,
            )
            from troopai.adk.types.responses.llm_response import LLMResponse

            if not isinstance(response, LLMResponse):
                raise TypeError(
                    f"invoke_model_activity expected LLMResponse from model={inp.model_name!r} "
                    f"but got {type(response).__name__}"
                )
            # Nullify timestamp before serialization: datetime is not JSON-serializable
            # with Temporal's default DataConverter, and _dict_to_llm_response always
            # reconstructs timestamp=None anyway.
            response_for_wire = dataclasses.replace(response, timestamp=None)
            return dataclasses.asdict(response_for_wire)
        finally:
            stop.set()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task

except ImportError as exc:
    if "temporalio" not in str(exc):
        raise
