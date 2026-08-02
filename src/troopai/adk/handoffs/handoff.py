"""Per-agent configuration for LLM-orchestrated handoffs.

When placed in an Agent's ``handoffs`` list, a ``Handoff`` appears as a
``transfer_to_<name>`` function tool that the LLM can call to initiate
a handoff to the wrapped agent.

Example:
    agent = Agent(
        name="triage",
        handoffs=[
            Handoff(target=refunds_agent, description="Handle refund requests"),
            billing_agent,   # bare Agent is auto-wrapped
        ],
    )

Cost-optimization example:
    # Limit how much history transfers to the target agent.
    Handoff(
        target=refunds_agent,
        config=HandoffConfig(budget=5_000),  # Max 5,000 tokens of history
    )

Typed handoff input example:
    class EscalationInput(BaseModel):
        reason: str
        priority: int

    Handoff(
        target=escalation_agent,
        input_type=EscalationInput,
        description="Escalate with reason and priority.",
    )
"""

from __future__ import annotations

import copy
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, NoReturn

from pydantic import TypeAdapter, ValidationError

from troopai.adk.exceptions import HandoffRejection
from troopai.adk.handoffs.handoff_config import HandoffConfig, apply_callback_error_policy
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.handoffs.handoff_target import (
    HandoffEnabledCallback,
    HandoffInputFilter,
    OnHandoffCallback,
    TAgent,
    THandoffInput,
    invoke_on_handoff,
)
from troopai.adk.run.context import RunContext, TContext
from troopai.adk.schemas.utils import SchemaEnforcement, enforce_schema
from troopai.adk.types.items import RunItem
from troopai.adk.utils import to_snake_case

logger = logging.getLogger(__name__)


HANDOFF_TOOL_PREFIX: str = "transfer_to_"
"""Prefix for auto-generated handoff tool names (e.g. ``transfer_to_refunds``)."""

if TYPE_CHECKING:
    from troopai.adk.agents import Agent
    from troopai.adk.tools.function_tool import FunctionTool


@dataclass(frozen=True)
class Handoff(Generic[TAgent, TContext, THandoffInput]):
    """Per-agent configuration for LLM-orchestrated handoffs.

    When placed in an Agent's handoffs list, appears as a
    ``transfer_to_<name>`` function tool that the LLM can call.

    Attributes:
        target: The agent to hand off to.
        name: Custom tool name. Default: ``transfer_to_{agent_name_snake_case}``.
        description: Tool description for the LLM. Default: auto-generated.
        on_handoff: Optional callback invoked when this handoff occurs.
        input_type: Pydantic model for typed handoff input. When set,
            auto-generates JSON schema and validates tool call args.
        schema: JSON Schema for the handoff tool-call arguments. Auto-generated
            from input_type if provided, or set explicitly.
        schema_enforcement: How strictly the schema should be enforced.
        input_filter: Optional function to transform handoff data.
        enabled: Whether this target is active (bool or callable).
        config: Handoff configuration (strategy, window, budget).
        metadata: Arbitrary string-valued labels for tracing and telemetry.
    """

    target: Agent[TContext]
    """The agent to hand off to."""

    name: str | None = None
    """Custom tool name. Default: ``transfer_to_{agent_name_snake_case}``."""

    description: str | None = None
    """Tool description for the LLM. Default: auto-generated."""

    on_handoff: OnHandoffCallback | None = None
    """Optional callback invoked when this handoff occurs.

    Accepts either:
    - ``(ctx, input)`` — receives the validated typed input
    - ``(ctx)`` — parameterless, only receives the run context

    The signature is detected automatically."""

    input_type: type[THandoffInput] | None = None
    """Pydantic model for typed handoff input. When set:
    - Auto-generates JSON schema for the tool definition.
    - Validates LLM tool call args in invoke().
    - Stores validated object in HandoffInputData.intent."""

    schema: dict[str, Any] | None = field(default=None, repr=False)
    """JSON Schema for the handoff tool-call arguments. Auto-generated
    from input_type if provided, or set explicitly for custom schemas.
    When None and input_type is also None, defaults to the canonical
    empty object schema ``{"type": "object", "properties": {}}``."""

    schema_enforcement: SchemaEnforcement = SchemaEnforcement.STRICT
    """How strictly the schema should be enforced by the LLM provider."""

    input_filter: HandoffInputFilter | None = None
    """Optional function to transform handoff data before passing to target agent."""

    enabled: HandoffEnabledCallback = True
    """Whether this target is active (bool or callable)."""

    config: HandoffConfig = HandoffConfig()
    """Handoff configuration (strategy, window, budget)."""

    metadata: Mapping[str, str] = field(default_factory=dict)
    """Arbitrary string-valued labels for tracing / telemetry.

    Symmetric with :attr:`FunctionTool.metadata`. Useful for tagging
    every transfer to a particular agent with team / owner / SLA
    metadata without participating in execution semantics. NOT
    shown to the LLM."""

    @property
    def agent_name(self) -> str:
        """Denormalized view of the target agent's name.

        Lets telemetry / handoff filters discriminate on the
        destination without dereferencing :attr:`target` (the
        ``Agent`` import pulls a heavier graph than callers usually
        want at log-emit time).
        """
        return self.target.name

    def __post_init__(self) -> None:
        """Derive schema from input_type if not explicitly provided."""
        if self.input_type is not None and self.schema is None:
            raw = TypeAdapter(self.input_type).json_schema()
            processed = enforce_schema(
                copy.deepcopy(raw),
                self.schema_enforcement,
            )
            object.__setattr__(self, "schema", processed)
        elif self.input_type is None and self.schema is None:
            # Parameterless transfer tool — canonical empty object schema.
            # Must include "type": "object" for Anthropic compatibility.
            object.__setattr__(
                self,
                "schema",
                {
                    "type": "object",
                    "properties": {},
                },
            )

    def get_name(self) -> str:
        """Generate the function tool name for this handoff.

        Default: ``transfer_to_{agent_name_snake_case}``. A target name
        written entirely in a non-Latin script sanitises to the empty
        string, so every such target would collapse to a bare
        ``transfer_to_`` and collide with the others. When the snake-case
        form is empty, a stable 12-hex-char digest of the raw name is
        used instead, keeping distinct targets distinct.
        """
        if self.name is not None:
            return self.name
        snake = to_snake_case(self.target.name)
        if len(snake) == 0:
            digest = hashlib.sha1(self.target.name.encode(), usedforsecurity=False).hexdigest()[:12]
            return f"{HANDOFF_TOOL_PREFIX}{digest}"
        return f"{HANDOFF_TOOL_PREFIX}{snake}"

    def get_description(self) -> str:
        """Get the tool description for the LLM.

        Fallback chain: Handoff.description → target.description →
        auto-generated from target name.
        """
        if self.description is not None:
            return self.description
        if self.target.description is not None:
            return self.target.description
        return f"Handoff to the {self.target.name} agent to handle the request."

    def to_tool(self) -> FunctionTool:
        """Convert to a ``FunctionTool`` for the LLM tool list.

        The schema and schema_enforcement flow through existing plumbing:
        ``build_handoff_tools()`` → ``_convert_tools()`` → ``parameters`` + ``strict``.
        No ``on_invoke`` is set — handoff tools are handled specially by the Runner.
        """
        from troopai.adk.tools.function_tool import FunctionTool

        return FunctionTool(
            name=self.get_name(),
            description=self.get_description(),
            schema=self.schema or {"type": "object", "properties": {}},
            schema_enforcement=self.schema_enforcement,
        )

    async def invoke(
        self,
        tool_args: str,
        context: tuple[RunItem, ...],
        output: tuple[RunItem, ...],
        run_context: RunContext[TContext],
    ) -> tuple[Agent[TContext], HandoffInputData]:
        """Execute this handoff target.

        1. Validates and parses ``tool_args`` if ``input_type`` is set
        2. Builds ``HandoffInputData`` from parsed/raw args + context + output
        3. Applies ``input_filter`` if configured
        4. Calls ``on_handoff`` callback if provided

        Note: Lifecycle hooks are NOT called here — they remain in the
        runner for proper orchestration and observability.

        Args:
            tool_args: Raw JSON arguments string from the tool call.
            context: Messages before the current agent's turn.
            output: Messages generated during the current agent's turn.
            run_context: The run context.

        Returns:
            Tuple of (target agent, filtered HandoffInputData).
        """
        # Parse and validate typed input if input_type is set.
        # ValidationError always surfaces to the LLM (it made the bad
        # tool call) — independent of `config.on_error` which governs
        # filter / callback errors in user code.
        intent: THandoffInput | str
        if self.input_type is not None:
            adapter: TypeAdapter[THandoffInput] = TypeAdapter(self.input_type)
            try:
                intent = adapter.validate_json(tool_args)
            except ValidationError as exc:
                raise HandoffRejection(
                    self.get_name(),
                    f"Invalid handoff arguments for '{self.get_name()}': {exc}",
                    cause=exc,
                ) from exc
            logger.debug(
                "Handoff '%s' validated typed input: %s",
                self.get_name(),
                type(intent).__name__,
            )
        else:
            intent = tool_args

        handoff_data = HandoffInputData(
            intent=intent,
            context=context,
            output=output,
        )

        if self.input_filter is not None:
            # User-supplied callback — any Exception is in scope. Routed
            # to the configured policy. BaseException (CancelledError,
            # KeyboardInterrupt, SystemExit) deliberately not caught.
            try:
                handoff_data = self.input_filter(handoff_data)
            except Exception as exc:
                self._handle_callback_error(exc, "input_filter")

        if self.on_handoff is not None:
            try:
                await invoke_on_handoff(
                    self.on_handoff,
                    run_context,
                    intent,
                    handoff_data=handoff_data,
                )
            except Exception as exc:
                self._handle_callback_error(exc, "on_handoff")

        return self.target, handoff_data

    def _handle_callback_error(self, exc: Exception, callback_kind: Literal["input_filter", "on_handoff"]) -> NoReturn:
        """Apply ``config.on_error`` to an exception from a user callback.

        Delegates to the shared :func:`apply_callback_error_policy` so this
        LLM-orchestrated path and the code-orchestrated ``HandoffTarget.invoke``
        path reject identically. Always raises.

        Args:
            exc: The original exception.
            callback_kind: ``"input_filter"`` or ``"on_handoff"``.

        Raises:
            HandoffRejection: when ``config.on_error == "reject_with_message"``.
            Exception: the original exception when ``config.on_error == "halt"``.
        """
        apply_callback_error_policy(
            name=self.get_name(),
            config=self.config,
            exc=exc,
            callback_kind=callback_kind,
        )


def handoff(
    target: Agent[TContext],
    *,
    name: str | None = None,
    description: str | None = None,
    on_handoff: OnHandoffCallback | None = None,
    input_type: type[THandoffInput] | None = None,
    input_filter: HandoffInputFilter | None = None,
    enabled: HandoffEnabledCallback = True,
    config: HandoffConfig = HandoffConfig(),
) -> Handoff[Any, TContext, THandoffInput]:
    """Build a Handoff for LLM-orchestrated agent routing.

    Convenience factory that mirrors OpenAI's ``handoff()`` pattern.
    Equivalent to constructing ``Handoff(...)`` directly — the
    ``__post_init__`` handles schema derivation from ``input_type``.

    Args:
        target: The agent to hand off to.
        name: Custom tool name.
        description: Tool description for the LLM.
        on_handoff: Callback invoked when this handoff occurs.
        input_type: Pydantic model for typed handoff input.
        input_filter: Function to transform handoff data.
        enabled: Whether this target is active.
        config: Handoff configuration.

    Returns:
        Configured Handoff instance.
    """
    return Handoff(
        target=target,
        name=name,
        description=description,
        on_handoff=on_handoff,
        input_type=input_type,
        input_filter=input_filter,
        enabled=enabled,
        config=config,
    )
