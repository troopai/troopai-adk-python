"""TemporalLLM — routes LLM calls through Temporal activities when inside a workflow.

Wraps any :class:`~troopai.adk.llms.llm.LLM` implementation and transparently
routes :meth:`acomplete` through :func:`~temporalio.workflow.execute_activity`
when called from inside a Temporal workflow.  Outside a workflow the call is
forwarded directly to the wrapped LLM, making ``TemporalLLM`` safe to use in
both durable and non-durable contexts.

Usage::

    from troopai.adk.llms import LiteLLM
    from troopai.adk.workflows.engine import ModelActivityConfig
    from troopai.adk.workflows.temporal.llm import TemporalLLM

    llm = TemporalLLM(
        wrapped=LiteLLM(model="gpt-4o"),
        activity_config=ModelActivityConfig(),
        model_name="gpt-4o",
    )

    # Or use the class method to install on an existing agent:
    TemporalLLM.install(agent, activity_config=ModelActivityConfig())

References:
    Temporal Python SDK workflow API:
    https://docs.temporal.io/develop/python/core-application#develop-workflows
    Temporal execute_activity docs:
    https://python.temporal.io/temporalio.workflow.html#execute_activity
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import TYPE_CHECKING, Any, override

from troopai.adk.llms.llm import LLM
from troopai.adk.workflows.engine import ModelActivityConfig

if TYPE_CHECKING:
    from troopai.adk.llms.cost import CostEstimate
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMStreamEvent
    from troopai.adk.types.tokens.llm_usage import LLMUsage

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TemporalLLM(LLM):
    """LLM bridge that routes ``acomplete`` through a Temporal activity.

    When called from inside a Temporal workflow (detected via
    ``workflow.in_workflow()``), each LLM call is serialized and dispatched
    through ``workflow.execute_activity()``.  This makes the call durable:
    if the worker crashes mid-call, Temporal replays the workflow and the
    activity retries from the last heartbeat.

    Outside a workflow the wrapped LLM is called directly — no overhead
    is added in non-durable paths.

    Attributes:
        wrapped: The real :class:`~troopai.adk.llms.llm.LLM` instance that
            handles provider communication.
        activity_config: Timeout and retry policy applied to every activity
            execution.
        model_name: Registry key used by the worker to look up the wrapped
            LLM via :func:`~troopai.adk.workflows.temporal.activity.get_model`.
            When empty, ``__post_init__`` derives it from the wrapped LLM's
            ``model`` identifier (falling back to ``str(wrapped)`` only when no
            ``model`` is exposed).

    References:
        Temporal retry policy:
        https://docs.temporal.io/retry-policies
        Temporal activity options:
        https://python.temporal.io/temporalio.workflow.html#execute_activity
    """

    wrapped: LLM
    """The underlying LLM implementation that handles provider communication."""

    activity_config: ModelActivityConfig
    """Timeout and retry policy for each Temporal activity execution."""

    model_name: str = ""
    """Registry key for worker-side model lookup.

    When empty at construction time, ``__post_init__`` derives it from the
    wrapped LLM's ``model`` identifier (the stable key that matches
    ``register_model()`` on the worker), falling back to ``str(wrapped)``
    only when no ``model`` attribute is exposed.
    """

    def __post_init__(self) -> None:
        if len(self.model_name) == 0:
            self.model_name = _derive_registry_key(self.wrapped)

    # ------------------------------------------------------------------
    # Cost delegation
    # ------------------------------------------------------------------

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        """Delegate USD costing to the wrapped LLM.

        Without this override the base :meth:`~troopai.adk.llms.llm.LLM.cost`
        returns ``None`` for every call, so a durable run's tenant dollar
        budget would silently never be charged.  Forwarding to the wrapped
        provider preserves the underlying cost table.

        Args:
            model: The model identifier that served the call.
            usage: Token usage for the call.

        Returns:
            The wrapped LLM's cost estimate, or ``None`` when it cannot price
            the call.
        """
        return self.wrapped.cost(model, usage)

    @override
    def estimate_cost(
        self,
        messages: list[LLMInputContentItem],
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        """Delegate the pre-call cost estimate to the wrapped LLM.

        Symmetric with :meth:`cost`: the wrapped provider owns token counting
        and pricing, so pre-call dollar gating stays correct under durable
        execution.

        Args:
            messages: The input messages to count tokens for.
            model: The model identifier.
            max_output_tokens: Optional output-side bound for the estimate.

        Returns:
            The wrapped LLM's :class:`~troopai.adk.llms.cost.CostEstimate`.
        """
        return self.wrapped.estimate_cost(messages, model, max_output_tokens=max_output_tokens)

    # ------------------------------------------------------------------
    # LLM ABC implementation
    # ------------------------------------------------------------------

    async def acomplete(  # type: ignore[override]  # overload narrowing lives on ABC; impl signature is wider
        self,
        messages: str | list[Any],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        """Dispatch an LLM call, routing through Temporal when inside a workflow.

        Args:
            messages: Conversation input — plain string or a list of
                provider-agnostic content items.
            llm_config: Optional LLM parameters (temperature, max tokens, etc.).
            tools: Optional pre-filtered tool list.
            output_schema: Optional structured output schema.
            stream: When ``True``, returns an ``AsyncIterator[LLMStreamEvent]``.
                Streaming is not supported inside a Temporal workflow (activity
                results are non-streaming); requesting ``stream=True`` inside a
                workflow raises ``NotImplementedError``.

        Returns:
            ``LLMResponse`` when ``stream=False``; ``AsyncIterator[LLMStreamEvent]``
            when ``stream=True`` and called outside a workflow.

        References:
            Temporal execute_activity:
            https://python.temporal.io/temporalio.workflow.html#execute_activity
        """
        try:
            from temporalio import workflow

            in_workflow = workflow.in_workflow()
        except ImportError:
            in_workflow = False

        if not in_workflow:
            logger.debug(
                "TemporalLLM: forwarding acomplete directly (outside workflow), model=%r",
                self.model_name,
            )
            if stream:
                return await self.wrapped.acomplete(
                    messages,
                    llm_config,
                    tools,
                    output_schema,
                    stream=True,
                )
            return await self.wrapped.acomplete(
                messages,
                llm_config,
                tools,
                output_schema,
                stream=False,
            )

        if stream:
            raise NotImplementedError(
                "TemporalLLM does not support stream=True inside a Temporal workflow. "
                "Temporal activity results are non-streaming. Use TemporalStreamingLLM "
                "for Workflow Stream-backed streaming, or set stream=False."
            )

        logger.info(
            "TemporalLLM: routing acomplete through Temporal activity, model=%r",
            self.model_name,
        )
        return await self._execute_as_activity(messages, llm_config, tools, output_schema)

    async def _execute_as_activity(
        self,
        messages: str | list[Any],
        llm_config: LLMConfig | None,
        tools: list[Tool] | None,
        output_schema: AgentOutputSchemaBase | None,
    ) -> LLMResponse:
        """Serialize inputs and dispatch to the Temporal activity.

        Builds a :class:`~troopai.adk.workflows.temporal.activity.ModelActivityInput`,
        calls :func:`~temporalio.workflow.execute_activity`, and deserializes
        the returned dict back to an :class:`~troopai.adk.types.responses.llm_response.LLMResponse`.

        Args:
            messages: Conversation input.
            llm_config: Optional LLM configuration.
            tools: Optional tool list.
            output_schema: Optional structured output schema.

        Returns:
            The :class:`~troopai.adk.types.responses.llm_response.LLMResponse`
            produced by the worker-side LLM invocation.

        References:
            Temporal RetryPolicy:
            https://python.temporal.io/temporalio.common.html#RetryPolicy
        """
        from temporalio import workflow
        from temporalio.common import RetryPolicy

        from troopai.adk.workflows.temporal.activity import ModelActivityInput, invoke_model_activity
        from troopai.adk.workflows.temporal.serialization import (
            config_to_json_dict,
            output_schema_to_json_dict,
            tool_to_json_dict,
        )

        messages_json = json.dumps(messages)

        # Serialize only the LLM-facing tool *definitions* (name/description/
        # parameter schema). The executables stay in the workflow, where each
        # call is journaled as its own activity. A raw asdict() would instead
        # try (and fail) to JSON-encode the on_invoke callable.
        tools_list: list[dict[str, Any]] = []
        if tools is not None:
            for t in tools:
                definition = tool_to_json_dict(t)
                if definition is None:
                    logger.warning(
                        "durable LLM call: a tool of type %s cannot be forwarded across "
                        "the activity boundary; it is invisible to the model this turn",
                        type(t).__name__,
                    )
                else:
                    tools_list.append(definition)
        tools_json = json.dumps(tools_list)

        config_json = "{}"
        if llm_config is not None:
            # config_to_json_dict coerces the non-JSON-safe fields (retry_policy's
            # frozenset, an httpx.Timeout) that a raw json.dumps would reject —
            # which previously crashed the durable run at its first LLM turn.
            config_json = json.dumps(config_to_json_dict(llm_config))

        output_schema_json = ""
        if output_schema is not None and not output_schema.is_plain_text():
            output_schema_json = json.dumps(output_schema_to_json_dict(output_schema))

        activity_input = ModelActivityInput(
            model_name=self.model_name,
            messages_json=messages_json,
            tools_json=tools_json,
            config_json=config_json,
            output_schema_json=output_schema_json,
        )

        cfg = self.activity_config
        retry_policy = RetryPolicy(
            maximum_attempts=cfg.maximum_attempts,
            initial_interval=timedelta(seconds=cfg.initial_interval),
            backoff_coefficient=cfg.backoff_coefficient,
            non_retryable_error_types=list(cfg.non_retryable_error_types),
        )

        result_dict: dict[str, Any] = await workflow.execute_activity(
            invoke_model_activity,
            activity_input,
            start_to_close_timeout=timedelta(seconds=cfg.start_to_close_timeout),
            heartbeat_timeout=timedelta(seconds=cfg.heartbeat_timeout),
            retry_policy=retry_policy,
        )

        return _dict_to_llm_response(result_dict)

    # ------------------------------------------------------------------
    # Install helper
    # ------------------------------------------------------------------

    @classmethod
    def install(
        cls,
        agent: Any,
        *,
        activity_config: ModelActivityConfig | None = None,
        model_name: str = "",
    ) -> None:
        """Recursively wrap every agent LLM in the handoff graph with ``TemporalLLM``.

        Walks ``agent.llm`` and all handoff-target agents' LLMs, replacing each
        :class:`~troopai.adk.llms.llm.LLM` instance with a ``TemporalLLM`` wrapper.
        Agents that already carry a ``TemporalLLM`` are skipped to prevent
        double-wrapping.  Circular handoff references are handled via a visited
        set of agent ``id()`` values.

        Args:
            agent: The root :class:`~troopai.adk.agents.agent.Agent` to install on.
            activity_config: Timeout and retry policy.  Defaults to
                :class:`~troopai.adk.workflows.engine.ModelActivityConfig` defaults.
            model_name: Registry key for worker-side lookup.  Defaults to
                ``str(wrapped)`` for each individual LLM.
        """
        config = activity_config if activity_config is not None else ModelActivityConfig()
        _install_recursive(agent, config=config, model_name=model_name, visited=set())


# ------------------------------------------------------------------
# Module-level helpers (private to this file)
# ------------------------------------------------------------------


def _derive_registry_key(wrapped: LLM) -> str:
    """Return the worker-registry key for *wrapped*.

    The :class:`~troopai.adk.llms.llm.LLM` ABC does not declare a ``model``
    attribute, but every concrete provider (``LiteLLM``, ``AnthropicModel``,
    ``OpenAIResponsesModel``, ``GeminiModel``, …) exposes the model identifier
    there.  It is the stable key that matches ``register_model()`` on the
    worker.  Probing for it (rather than ``str(wrapped)``) avoids embedding a
    per-process object address like ``<LiteLLM object at 0x…>`` — a
    non-deterministic key that never matches the worker registry and breaks
    replay.  When no ``model`` is exposed, ``str(wrapped)`` is the documented
    fallback and the caller should pass ``model_name`` explicitly.
    """
    candidate = getattr(wrapped, "model", None)
    if isinstance(candidate, str) and len(candidate) > 0:
        return candidate
    return str(wrapped)


def _install_recursive(
    agent: Any,
    *,
    config: ModelActivityConfig,
    model_name: str,
    visited: set[int],
) -> None:
    """Recursively install ``TemporalLLM`` on *agent* and all reachable handoff targets."""
    agent_id = id(agent)
    if agent_id in visited:
        return
    visited.add(agent_id)

    # Replace the agent's LLM if it is a real LLM instance and not already wrapped.
    llm = getattr(agent, "llm", None)
    if isinstance(llm, LLM) and not isinstance(llm, TemporalLLM):
        # Pass model_name through as-is (possibly empty) so __post_init__ owns
        # derivation: it prefers the wrapped LLM's stable `model` id over the
        # per-process object address that str(llm) would yield.
        wrapped = TemporalLLM(
            wrapped=llm,
            activity_config=config,
            model_name=model_name,
        )
        agent.llm = wrapped
        logger.info(
            "TemporalLLM installed on agent %r (model_name=%r)",
            getattr(agent, "name", repr(agent)),
            wrapped.model_name,
        )

    # Walk handoff targets.
    handoffs = getattr(agent, "handoffs", None)
    if handoffs is None:
        return
    if isinstance(handoffs, list):
        for entry in handoffs:
            if hasattr(entry, "llm"):
                _install_recursive(entry, config=config, model_name=model_name, visited=visited)
            else:
                target = getattr(entry, "target", None)
                if target is not None:
                    _install_recursive(target, config=config, model_name=model_name, visited=visited)
                else:
                    logger.warning(
                        "TemporalLLM.install: unrecognized handoff entry type %r — skipped",
                        type(entry).__name__,
                    )
    else:
        logger.warning(
            "TemporalLLM.install: agent %r has non-list handoffs of type %r — "
            "target agents will NOT have TemporalLLM installed. "
            "Register their LLMs manually.",
            getattr(agent, "name", repr(agent)),
            type(handoffs).__name__,
        )


def _dict_to_llm_response(data: dict[str, Any]) -> LLMResponse:
    """Reconstruct an :class:`~troopai.adk.types.responses.llm_response.LLMResponse` from a dict.

    The dict is produced by ``dataclasses.asdict(response)`` inside
    :func:`~troopai.adk.workflows.temporal.activity.invoke_model_activity`.
    Each ``response`` part is re-instantiated from its ``type`` discriminator.

    Args:
        data: Dict produced by ``dataclasses.asdict(LLMResponse(...))``.

    Returns:
        A fully reconstructed :class:`~troopai.adk.types.responses.llm_response.LLMResponse`.
    """
    from troopai.adk.types.responses.llm_response import (
        LLMResponse,
        LLMResponseAnnotation,
        LLMResponseFunctionToolCall,
        LLMResponsePart,
        LLMResponseProviderItem,
        LLMResponseReasoning,
        LLMResponseRefusal,
        LLMResponseText,
    )

    parts: list[LLMResponsePart] = []
    for part in data.get("response", []):
        part_type = part.get("type")
        if part_type == "text":
            part_dict = dict(part)
            # asdict flattened each LLMResponseAnnotation to a plain dict; rebuild
            # it so a later to_param() (which calls asdict on each annotation)
            # does not receive a dict and raise.
            raw_annotations = part_dict.get("annotations")
            if raw_annotations is not None:
                part_dict["annotations"] = [
                    LLMResponseAnnotation(**a) if isinstance(a, dict) else a for a in raw_annotations
                ]
            parts.append(LLMResponseText(**part_dict))
        elif part_type == "thinking":
            parts.append(LLMResponseReasoning(**dict(part)))
        elif part_type == "function_call":
            parts.append(LLMResponseFunctionToolCall(**dict(part)))
        elif part_type == "refusal":
            parts.append(LLMResponseRefusal(**dict(part)))
        elif part_type == "provider_item":
            parts.append(LLMResponseProviderItem(**dict(part)))
        else:
            raise ValueError(
                f"Unknown part type {part_type!r} in durable LLM response — "
                "update _dict_to_llm_response to handle this type. "
                "Raising here causes the Temporal activity to fail and retry rather than "
                "committing a truncated response to durable history that replay would reuse forever."
            )

    return LLMResponse(
        response_id=data.get("response_id", ""),
        model=data.get("model", ""),
        response=parts,
        usage=_dict_to_usage(data.get("usage")),
        finish_reason=data.get("finish_reason"),
        timestamp=None,
    )


def _dict_to_usage(usage_data: dict[str, Any] | None) -> LLMUsage | None:
    """Reconstruct :class:`LLMUsage` from its ``asdict`` form.

    ``LLMUsage`` is a plain dataclass whose ``BeforeValidator`` normalizers fire
    only under pydantic validation, so a bare ``LLMUsage(**usage_data)`` would
    leave ``input_tokens_details`` / ``output_tokens_details`` / ``usage`` as raw
    dicts and break the runner's usage accumulation (``__add__`` reads
    ``.cached_tokens`` etc.). Validate through a ``TypeAdapter`` so every nested
    token-detail rebuilds as its proper type. Without this, durable runs never
    accumulate usage — ``UsageLimitExceeded`` never fires and tenant dollar
    budgets are never charged.
    """
    if usage_data is None:
        return None
    from pydantic import TypeAdapter

    from troopai.adk.types.tokens.llm_usage import LLMUsage

    return TypeAdapter(LLMUsage).validate_python(usage_data)
