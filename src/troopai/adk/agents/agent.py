from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

from troopai.adk.agents.agent_guardrails import AgentGuardrails
from troopai.adk.agents.middleware import Middleware
from troopai.adk.exceptions import ToolDependencyError
from troopai.adk.handoffs import Handoff, HandoffRoute
from troopai.adk.llms import LLM, LLMConfig
from troopai.adk.prompts import DynamicSystemPrompt, SystemPrompt
from troopai.adk.schemas import AgentOutputSchemaBase
from troopai.adk.skills.activation import SkillActivation
from troopai.adk.tools import FunctionTool, FunctionToolset, Tool, Toolset
from troopai.adk.types.tools import ToolUseBehavior
from troopai.adk.utils import MaybeAwaitable, to_snake_case

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from troopai.adk.hooks import AgentHooks, RunHooks
    from troopai.adk.llms.llm_usage import LLMUsageLimits
    from troopai.adk.run import RunConfig
    from troopai.adk.run.stream import StreamEvent
    from troopai.adk.schemas import SchemaEnforcement
    from troopai.adk.skills.skill import Skill
    from troopai.adk.tools.function_tool import ToolErrorFunction
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.tools.tool_guardrails import ToolGuardrails
    from troopai.adk.types.agents.agent_as_tool_types import (
        AgentToolInputBuilder,
        AgentToolOutputExtractor,
    )
    from troopai.adk.verbose.config import VerboseConfig

TContext = TypeVar("TContext", default=Any)


@dataclasses.dataclass
class BaseAgent(Generic[TContext]):
    """Base class for an autonomous agent.

    Provides the minimal shared state that every agent needs: a
    unique name and a list of tools (which may include
    ``MCPToolset`` instances for Model Context Protocol integrations).

    Attributes:
        name: Unique identifier for this agent.
        description: Short description of what this agent does; flows to
            ``as_tool()``, ``subagents`` auto-conversion, and handoff
            target descriptions. ``None`` auto-generates a generic
            description from the agent's name.
        tools: All tools and toolsets the agent can use (function tools,
            built-in tools, ``MCPToolset`` for MCP servers, etc.).
    """

    name: str
    """Unique identifier for this agent."""

    description: str | None = None
    """Short description of what this agent does.

    Used as the default tool description when this agent is:
    - Listed in another agent's ``subagents`` (auto-converted to tool)
    - Wrapped via ``as_tool()`` without explicit ``tool_description``
    - Used as a handoff target without explicit ``Handoff.description``

    When ``None``, a generic description is auto-generated from the
    agent's name (e.g., ``"Delegate a task to the Researcher agent."``).
    """

    tools: list[Tool | Toolset] = dataclasses.field(default_factory=list)
    """All tools and toolsets the agent can use.

    Accepts any ``Tool`` variant or any ``Toolset`` instance:

    - ``FunctionTool`` — custom Python functions
    - ``ShellTool`` / ``ApplyPatchTool`` — local-execution tools with
      a user-supplied executor/editor; wrapped as ``FunctionTool`` at
      runtime
    - ``ExecutableBuiltinTool`` subclasses (``MemoryTool``,
      ``JITContextAwareTool``) — framework-run tools with an
      ``on_invoke`` callback
    - ``Toolset`` instances (``FunctionToolset``, ``PrefixedToolset``,
      ``FilteredToolset``, ``RenamedToolset``, ``CombinedToolset``,
      ``WrapperToolset``) — live tool collections that materialise
      into ``FunctionTool`` instances each turn via
      ``get_tools(ctx)``. Useful for organising MCP-heavy or
      multi-source tool collections without name collisions.

    Provider-native capabilities (web search, file search, computer
    use, code interpreter, image generation, remote MCP) are NOT
    wrapped as framework tool classes. Pass the raw provider JSON
    through ``LLMConfig.extra_body`` / ``extra_args`` instead.

    Example::

        from troopai.adk.tools import function_tool

        @function_tool(name="lookup", description="Look up a record")
        def lookup(record_id: str) -> str:
            return f"Record {record_id}"

        agent = Agent(
            name="Support Agent",
            system_prompt="Help customers.",
            tools=[lookup],
        )

    Toolset example::

        from troopai.adk.tools import FunctionToolset

        weather = FunctionToolset(tools=[get_temp, get_conditions]).prefixed("weather")
        db = FunctionToolset(tools=[query, insert]).prefixed("db")

        agent = Agent(
            name="Operations",
            system_prompt="Help with operations.",
            tools=[weather, db],
        )
        # The LLM sees: weather_get_temp, weather_get_conditions,
        # db_query, db_insert
    """


@dataclasses.dataclass
class Agent(BaseAgent, Generic[TContext]):
    """An autonomous agent that can execute tasks using tools and LLM capabilities.

    Agents are the core building blocks of the TroopAI framework. They combine:
    - Instructions that define the agent's behavior and capabilities
    - Tools that the agent can use to perform actions
    - Guardrails that validate inputs and outputs for safety

    Attributes:
        name: Unique identifier for this agent.
        description: Short description of what this agent does; flows to
            ``as_tool()``, ``subagents`` auto-conversion, and handoff
            descriptions. ``None`` auto-generates one from the name.
        tools: All tools and toolsets the agent can use (function tools,
            built-in tools, ``MCPToolset`` for MCP servers, ``Toolset``
            instances, etc.).
        system_prompt: Instructions as ``str``, ``SystemPrompt``, or a
            ``DynamicSystemPrompt`` callable resolved at runtime.
        handoffs: Delegation targets — a ``HandoffRoute`` (code-orchestrated)
            or a list of agents / ``Handoff`` objects (LLM-orchestrated).
        guardrails: ``AgentGuardrails`` config holding ``input`` and
            ``output`` phase-typed lists.
        llm: Model override — a model-name ``str`` or an ``LLM`` instance;
            ``None`` uses the default model from ``RunConfig``.
        llm_config: Model parameters (temperature, top_p, max_output_tokens,
            etc.); ``None`` uses provider defaults.
        middleware: ``Middleware`` config holding per-layer plumbing lists
            (``tools``, ``agents``, ``llms``).
        tool_use_behavior: What happens after tool execution
            (``"run_llm_again"``, ``"stop_on_first_tool"``, ``StopAtTools``,
            or a custom function).
        output_schema: Structured-output schema — a type or an
            ``AgentOutputSchema`` instance; ``None`` for free-form text.
        skills: Skills that augment the agent with extra instructions and
            tools when activated.
        skill_activation: When skill instructions enter the system prompt —
            ``EAGER`` (at run start) or ``LAZY`` (on first skill-tool call).
        hooks: Optional per-agent lifecycle hooks, scoped to this agent
            instance alongside run-level ``RunHooks``.
        verbose: Optional per-agent verbose-output override; ``None`` uses
            the run-level ``RunConfig.verbose``.

    Example::

        from troopai.adk.agents import Agent, AgentGuardrails

        agent = Agent(
            name="Customer Support",
            system_prompt="Help customers with their questions.",
            tools=[search_tool, database_tool],
            guardrails=AgentGuardrails(
                input=[pii_guardrail],
                output=[content_guardrail],
            ),
        )
    """

    system_prompt: str | SystemPrompt | DynamicSystemPrompt | None = None
    """Instructions that define the agent's behavior and capabilities.

    Accepts:

    - ``str``: Plain text system prompt.
    - ``SystemPrompt``: Structured prompt with role, context, guidelines.
    - ``DynamicSystemPrompt``: Callable that receives ``RunContext`` and
      ``Agent``, returning a ``str`` or ``SystemPrompt`` at runtime.
    """

    handoffs: HandoffRoute[Any, TContext] | list[Agent[Any] | Handoff[Any, TContext]] | None = None
    """Defines how this agent can delegate tasks to other agents.

    Supports two strategies:

    **Code-orchestrated** (HandoffRoute): Deterministic Intent-based routing.
    The LLM outputs a structured Intent, and code routes via isinstance().
    Zero routing tokens::

        handoffs=HandoffRoute("triage")
            .when(RefundIntent).to(refunds_agent, on_handoff=log_refund)
            .when(BillingIntent, CancelSubscription).to(billing_agent)
            .otherwise(general_agent)

    **LLM-orchestrated** (list): Agents exposed as transfer_to_<name> tools.
    The LLM autonomously decides when to hand off::

        handoffs=[refunds_agent, billing_agent]
        # or with config:
        handoffs=[
            Handoff(target=refunds_agent, description="Handle refunds"),
            billing_agent,
        ]
    """

    guardrails: AgentGuardrails = dataclasses.field(default_factory=AgentGuardrails)
    """Per-phase agent-level guardrails registered on this agent.

    A single :class:`~troopai.adk.agents.agent_guardrails.AgentGuardrails`
    config object holds two phase-typed lists:

    - ``guardrails.input``: validates input before (or in parallel with)
      agent execution. If any tripwire fires, an
      ``AgentInputGuardrailTripwireTriggered`` exception is raised.
      Input guardrails default to ``run_in_parallel=True`` (better
      latency); set ``run_in_parallel=False`` to block before the agent
      starts (saves tokens when the tripwire fires).
    - ``guardrails.output``: validates output after the agent completes.
      If any tripwire fires, an ``AgentOutputGuardrailTripwireTriggered``
      exception is raised — unless the guardrail sets ``remediation``,
      in which case the runner re-prompts the agent with that feedback.

    Example::

        from troopai.adk.agents import Agent, AgentGuardrails

        agent = Agent(
            name="Customer Support",
            system_prompt="...",
            guardrails=AgentGuardrails(
                input=[pii_guardrail],
                output=[content_guardrail],
            ),
        )
    """

    llm: str | LLM | None = None
    """Optional LLM override for this agent.

    - ``str``: Model name forwarded to the active LLM backend (e.g. ``"gpt-4o"``).
    - ``LLM`` instance: Custom LLM implementation.
    - ``None``: Uses the default model from ``RunConfig``.
    """

    llm_config: LLMConfig | None = None
    """Optional LLM configuration for this agent.

    Controls model parameters like temperature, top_p, max_output_tokens.
    If not set, uses the active LLM backend's defaults.

    Example::

        agent = Agent(
            name="Creative Writer",
            system_prompt="Write creative stories.",
            llm_config=LLMConfig(temperature=0.9, max_output_tokens=2000),
        )
    """

    middleware: Middleware = dataclasses.field(default_factory=Middleware)
    """Per-layer middleware lists registered on this agent.

    A single :class:`~troopai.adk.agents.middleware.Middleware` config
    object holds typed lists for each execution layer the framework
    supports. Three slots are wired:

    - ``tools`` — wraps each ``tool.on_invoke`` call with
      ``(ctx, tool, args, next) -> result``. See
      :class:`~troopai.adk.tools.tool_middleware.ToolMiddleware`.
    - ``agents`` — wraps each per-agent block in the run loop with
      ``(agent, messages, ctx, next) -> AgentBlockOutcome``. The
      chain re-fires on every handoff / swarm transition. See
      :class:`~troopai.adk.run.agent_middleware.AgentMiddleware`.
    - ``llms`` — wraps each ``LLM.acomplete()`` call the runner
      issues with ``(agent, messages, llm_config, ctx, next)
      -> LLMResponse``. See
      :class:`~troopai.adk.llms.llm_middleware.LLMMiddleware`.

    All three lists are composed outer-to-inner: the first entry
    runs first (outermost), the last entry runs last (innermost,
    just before the terminal).

    Use middleware for cross-cutting plumbing that should apply to
    every call at the matching layer during normal (non-HITL)
    execution: distributed-tracing spans, latency metrics, structured
    audit logs, retry-with-backoff policies, agent-global arg
    injection (``request_id`` from ``RunContext``), caching. Notes:

    - Tool calls resumed after a Human-in-the-Loop approval currently
      bypass the ``tools`` chain — input/output guardrails still apply.
    - Under ``Runner.arun(stream=True)``, the ``agents`` chain still
      applies; the ``llms`` chain does not — streaming LLM calls route
      through the separate ``stream_llms`` slot. The runner emits a
      warning per streaming call when ``llms`` is non-empty.

    Plumbing-only contract — distinct from guardrails:

    - **Guardrails** are typed verdicts (``allow`` / ``reject_content``
      / ``raise_exception``). They decide *whether* a call should
      proceed and *what to do about it*. Anything safety- or policy-
      flavoured (PII, jailbreak, content filtering, schema validation,
      rate limiting, approval gates) belongs here.
    - **Middleware** is plumbing that wraps the call. It can observe,
      transform args/messages, transform results, retry, or short-
      circuit, but MUST NOT encode policy. Each Protocol's docstring
      carries its own forbidden-vs-allowed table.

    Composition with toolset-scoped middleware on ``WrapperToolset``:
    the agent-global ``tools`` chain wraps the toolset-scoped chain,
    which wraps the tool's own ``on_invoke``.

    See ``docs/tools/middleware.md``, ``docs/agents/agent_middleware.md``,
    and ``docs/llms/llm_middleware.md`` for usage. See
    ``examples/tools/middleware/``, ``examples/agents/agent_middleware/``,
    and ``examples/llms/llm_middleware/`` for examples.

    Example::

        from troopai.adk.agents import Middleware
        from troopai.adk.tools import ToolLoggingMiddleware, ToolMetricsMiddleware
        from troopai.adk.run.agent_middleware import AgentLoggingMiddleware
        from troopai.adk.llms.llm_middleware import LLMLoggingMiddleware

        agent = Agent(
            name="Service",
            system_prompt="...",
            tools=[search, summarise],
            middleware=Middleware(
                tools=[ToolLoggingMiddleware(), ToolMetricsMiddleware(recorder=my_metrics)],
                agents=[AgentLoggingMiddleware()],
                llms=[LLMLoggingMiddleware()],
            ),
        )
    """

    tool_use_behavior: ToolUseBehavior = "run_llm_again"
    """Controls what happens after tool execution:
    - "run_llm_again" (default): Results go back to LLM.
    - "stop_on_first_tool": First tool's output = final result.
    - StopAtTools(stop_at_tool_names=[...]): Stop on specific tools.
    - Custom function: (RunContext, tool_results) → ToolsToFinalOutputResult.

    Only applies to FunctionTools, not handoff tools.
    Combine with LLMConfig(tool_choice="required") to force+stop pattern."""

    output_schema: Any | AgentOutputSchemaBase | None = None
    """Optional output schema for structured output.

    Accepts any Python type or an ``AgentOutputSchema`` instance.  When
    set, the LLM is instructed to return JSON matching this schema, and
    the response is validated automatically.

    Supported types:

    - ``BaseModel`` subclass: Direct JSON schema generation.
    - ``TypedDict``, ``@dataclass``, ``dict``: Direct schema.
    - Plain types (``int``, ``list[str]``, etc.): Auto-wrapped.
    - ``AgentOutputSchema(MyType, strict=False)``: Advanced config.

    Example::

        from pydantic import BaseModel

        class SupportResponse(BaseModel):
            category: str
            priority: int
            response: str

        # Simple: pass the type directly (auto-wrapped in AgentOutputSchema)
        agent = Agent(
            name="Support",
            system_prompt="...",
            output_schema=SupportResponse,
        )

        # Advanced: pass AgentOutputSchema with custom enforcement
        from troopai.adk.schemas import AgentOutputSchema, SchemaEnforcement

        agent = Agent(
            name="Support",
            system_prompt="...",
            output_schema=AgentOutputSchema(
                SupportResponse, schema_enforcement=SchemaEnforcement.NORMALIZED,
            ),
        )
    """

    skills: list[Skill] = dataclasses.field(default_factory=list)
    """Skills that augment this agent's capabilities.

    Each skill contributes instructions (injected into the system
    prompt) and tools (merged into the tool list) when activated.

    Skills are activated based on ``skill_activation``:

    - **EAGER**: All skill instructions and tools available
      immediately at run start.
    - **LAZY** (default): Instructions loaded on first tool call from
      the skill.  Tool descriptions are always present so the LLM can
      discover them.
    """

    skill_activation: SkillActivation = SkillActivation.LAZY
    """Controls when skill instructions enter the system prompt.

    - **EAGER**: Instructions injected at run start (every turn pays
      the full skill instruction-text cost regardless of whether the
      skill is used).
    - **LAZY** (default): Instructions injected when a skill's tool
      is first called. Zero-overhead discovery since tool descriptions
      are already in the tool list. ``LAZY`` is the default so the
      developer never pays per-turn skill-text cost they did not
      explicitly choose.
    """

    hooks: AgentHooks[Any] | None = None
    """Optional per-agent lifecycle hooks.

    Fires alongside run-level ``RunHooks`` at matching lifecycle
    points, but scoped to this specific agent instance. Useful in
    multi-agent swarms where each agent wants its own observability,
    metrics, or side effects without cluttering the run-level hooks
    with per-agent conditionals.

    Subset of lifecycle events exposed:

    - ``on_start`` / ``on_end`` — agent turn boundaries
    - ``on_handoff`` — fires on the incoming agent when handed off TO
    - ``on_llm_start`` / ``on_llm_end`` — each LLM call
    - ``on_tool_start`` / ``on_tool_end`` — each tool execution

    Guardrail, skill-activation, and session hooks stay run-level
    only — they are cross-cutting concerns, not per-agent behavior.
    See ``docs/hooks/agent_hooks.md`` for usage.
    """

    verbose: VerboseConfig | None = None
    """Optional per-agent verbose-output override.

    When set, events attributed to this agent (LLM calls, tool
    calls, start/end, handoff source) are rendered with this
    config's styles instead of the run-level ``RunConfig.verbose``.

    A typical usage: silence one agent in a handoff chain while
    keeping others loud::

        Agent(name="Summariser", verbose=VerboseConfig(enabled=False))

    Or recolour a specific agent so it stands out in multi-agent
    output::

        Agent(name="Router", verbose=VerboseConfig())  # default styles
        Agent(
            name="Specialist",
            verbose=VerboseConfig(
                styles={**_default_styles(), EVENT_TOOL_START: EventStyle(color="bright_magenta")},
            ),
        )

    When ``None``, the run-level config (if any) applies.
    """

    def __post_init__(self) -> None:
        """Validate agent configuration after initialization."""
        if len(self.name) == 0:
            raise ValueError("Agent name cannot be empty")
        if (
            self.system_prompt is None
            or (isinstance(self.system_prompt, str) and len(self.system_prompt) == 0)
            or (isinstance(self.system_prompt, SystemPrompt) and len(self.system_prompt.role) == 0)
        ):
            raise ValueError("Agent instructions cannot be empty")

        # Validate tool name uniqueness across agent tools and skill tools.
        # Toolset entries are skipped here — their names are only known
        # after ``get_tools(ctx)`` materialises them per turn, so name-
        # uniqueness is checked by ``build_tools()`` against the
        # post-materialisation dict.
        if len(self.skills) > 0:
            tool_names: set[str] = set()
            for tool in self.tools:
                if isinstance(tool, Toolset):
                    continue
                tool_name = getattr(tool, "name", None)
                if tool_name is not None:
                    tool_names.add(tool_name)
            for skill in self.skills:
                for tool in skill.tools:
                    tool_name = getattr(tool, "name", None)
                    if tool_name is not None and tool_name in tool_names:
                        raise ValueError(
                            f"Duplicate tool name '{tool_name}' — found in "
                            f"both agent tools and skill '{skill.name}'. "
                            f"Tool names must be unique across all skills "
                            f"and agent tools."
                        )
                    if tool_name is not None:
                        tool_names.add(tool_name)

        # Validate per-tool runtime dependencies (env vars, packages).
        # Failing fast at construction surfaces misconfiguration before
        # any LLM token is spent. Walks every FunctionTool in self.tools
        # AND in every attached skill, aggregating all unsatisfied
        # requirements into a single ToolDependencyError. ``setdefault``
        # is defensive against same-named tools across skills — name
        # uniqueness is checked above, but the merge keeps the validator
        # robust if that check ever moves.
        #
        # For Toolset entries the framework can introspect statically
        # (``FunctionToolset`` exposes its ``.tools`` field), validate
        # those tools too. Live wrappers (filtered/prefixed/etc.) carry
        # only a ``wrapped`` reference; their materialised tool list is
        # unknown until runtime, so dependency validation defers to the
        # individual tools the user supplied to the leaf
        # ``FunctionToolset``.
        unsatisfied: dict[str, list[str]] = {}
        for tool in self.tools:
            if isinstance(tool, FunctionTool):
                missing = tool.validate_dependencies()
                if len(missing) > 0:
                    unsatisfied.setdefault(tool.name, []).extend(missing)
            elif isinstance(tool, FunctionToolset):
                for inner_tool in tool.tools:
                    inner_missing = inner_tool.validate_dependencies()
                    if len(inner_missing) > 0:
                        unsatisfied.setdefault(inner_tool.name, []).extend(inner_missing)
        for skill in self.skills:
            for tool in skill.tools:
                if isinstance(tool, FunctionTool):
                    missing = tool.validate_dependencies()
                    if len(missing) > 0:
                        unsatisfied.setdefault(tool.name, []).extend(missing)
        if len(unsatisfied) > 0:
            raise ToolDependencyError(self.name, unsatisfied)

    def get_delegate_tools(self) -> list[FunctionTool]:
        """Return all tools that wrap delegate agents.

        Filters ``self.tools`` for ``FunctionTool`` instances whose
        ``get_delegate_agent()`` returns an Agent.  Useful for
        introspecting which tools represent sub-agent delegations
        versus regular function tools.

        Returns:
            A list of ``FunctionTool`` instances whose delegate agent is
            set (i.e., created via ``as_tool()``).
        """
        return [t for t in self.tools if isinstance(t, FunctionTool) and t.get_delegate_agent() is not None]

    def get_agent_graph(self) -> dict[str, Any]:
        """Build a recursive graph of agent relationships.

        Walks delegate tools (via ``as_tool()``) and handoff targets
        to build a traversable topology dict.  Each node contains the
        agent name, its delegate sub-agents, and its handoff targets.

        Returns:
            A dict with keys ``name``, ``delegates``, and ``handoffs``.

        Example::

            graph = supervisor.get_agent_graph()
            # {
            #     "name": "Supervisor",
            #     "delegates": [
            #         {"name": "Researcher", "delegates": [], "handoffs": []},
            #         {"name": "Writer", "delegates": [], "handoffs": []},
            #     ],
            #     "handoffs": ["Refunds Agent"],
            # }
        """
        return _build_agent_graph(self, set())

    def as_tool(
        self,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
        input_schema: type[BaseModel] | None = None,
        extractor: AgentToolOutputExtractor | None = None,
        input_builder: AgentToolInputBuilder | None = None,
        on_stream: Callable[[StreamEvent], MaybeAwaitable[None]] | None = None,
        max_turns: int = 10,
        timeout: float | None = None,
        budget: LLMUsageLimits | None = None,
        max_result_tokens: int | None = None,
        run_config: RunConfig | None = None,
        hooks: RunHooks[Any] | None = None,
        error_function: ToolErrorFunction | None = None,
        guardrails: ToolGuardrails | None = None,
        enabled: bool | Callable | MaybeAwaitable[bool] = True,
        requires_approval: bool | Callable = False,
        enforcement: SchemaEnforcement | None = None,
    ) -> FunctionTool:
        """Wrap this agent as a FunctionTool for LLM-orchestrated delegation.

        Returns a standard ``FunctionTool`` that, when called by the parent
        LLM, runs this agent as a sub-agent and returns the result as tool
        output.  Control stays with the parent agent (unlike handoffs where
        control transfers permanently).

        The sub-agent starts fresh with only the tool input (no shared
        conversation history).

        Args:
            tool_name: Name the parent LLM sees.  Defaults to
                ``snake_case(self.name)``.
            tool_description: Description the parent LLM sees.  Defaults to
                ``"Delegate a task to the {name} agent."``.
            input_schema: Pydantic model for the tool's input.  Defaults to
                ``AgentToolInput`` (single ``input`` string field).
            extractor: Callback to customise what the parent LLM sees
                from the sub-agent's ``RunResult``.  Receives the full
                ``RunResult`` and returns a string.  If ``None``,
                ``str(result.final_output)`` is used.
            input_builder: Callback to transform the parsed input model into
                the value passed to ``Runner.arun(agent, user_prompt=...)``.
                Receives the validated Pydantic instance and the
                ``ToolContext``.  If ``None``, the default schema's
                ``.input`` field is used, or ``json.dumps(parsed.model_dump())``
                for custom schemas.
            on_stream: Optional callback to receive streaming events from the
                nested agent run. When provided, the sub-agent runs in streaming
                mode via ``Runner.arun(stream=True)``. The callback receives each
                ``StreamEvent`` (raw tokens, tool calls, handoffs). Supports
                both sync and async callables.
            max_turns: Maximum turns for the sub-agent's execution loop.
            timeout: Maximum seconds for the sub-agent run.  Uses
                ``asyncio.wait_for()`` — on timeout, returns an error
                string to the parent LLM (does not raise).
            budget: Token/request limits for this sub-agent's execution.
                Merged into the sub-agent's ``RunConfig.usage_limits``.
                The sub-agent raises ``UsageLimitExceeded`` if exceeded,
                which is caught and returned as an error string.
            max_result_tokens: Maximum token count for the sub-agent's
                result string.  The Runner truncates any result exceeding
                this limit before inserting into the parent's message
                history.  Passed through to the underlying ``FunctionTool``.
            run_config: Optional ``RunConfig`` for the sub-agent run.  If
                ``None``, inherits the parent run's config (threaded to the
                tool context and read via ``ToolContext.get_run_config()``).
            hooks: Optional ``RunHooks`` for the sub-agent run.
            error_function: Custom error handler.  Receives
                ``(RunContext, Exception)`` and returns an error string.
                If ``None``, a descriptive default message is returned.
            guardrails: ``ToolGuardrails`` config holding ``input``
                and ``output`` phase-typed lists (default ``None``).
            enabled: Whether the tool is enabled (static bool or callable).
            requires_approval: Whether the tool needs human approval.
            enforcement: Schema enforcement strategy.  Defaults to
                ``SchemaEnforcement.NORMALIZED``.

        Returns:
            A ``FunctionTool`` that can be added to another agent's tool list.

        Example:
            .. code-block:: python

                researcher = Agent(name="Researcher", system_prompt="...")
                writer = Agent(name="Writer", system_prompt="...")

                supervisor = Agent(
                    name="Supervisor",
                    system_prompt="Coordinate research and writing.",
                    tools=[
                        researcher.as_tool(),
                        writer.as_tool(),
                    ],
                )


                # With streaming (observe sub-agent tokens in real-time):
                async def on_sub_agent_event(event):
                    if event.type == "raw_response_event":
                        logger.info(event.data)


                supervisor = Agent(
                    name="Supervisor",
                    system_prompt="Coordinate research and writing.",
                    tools=[
                        researcher.as_tool(on_stream=on_sub_agent_event),
                    ],
                )
        """
        from troopai.adk.exceptions import UserError
        from troopai.adk.schemas import SchemaEnforcement
        from troopai.adk.types.agents.agent_as_tool_types import AgentToolInput

        # --- Resolve defaults ---
        resolved_name = tool_name or to_snake_case(self.name)
        if len(resolved_name) == 0:
            raise UserError(
                f"Agent name {self.name!r} produces an empty tool name after "
                "conversion to snake_case. Pass an explicit tool_name= to as_tool()."
            )
        resolved_description = tool_description or self.description or f"Delegate a task to the {self.name} agent."
        resolved_schema = input_schema or AgentToolInput
        resolved_enforcement = enforcement or SchemaEnforcement.NORMALIZED

        # Capture references for the closure
        agent = self
        _extractor = extractor
        _input_builder = input_builder
        _on_stream = on_stream
        _max_turns = max_turns
        _timeout = timeout
        _budget = budget
        _run_config = run_config
        _hooks = hooks
        _error_function = error_function
        _is_default_schema = input_schema is None

        async def _on_invoke_tool(ctx: ToolContext, raw_input: str) -> str:
            """Inner tool invocation that runs the sub-agent."""
            try:
                # Stage 1: Parse and validate input
                json_data = json.loads(raw_input) if len(raw_input) > 0 else {}
                parsed = resolved_schema(**json_data)

                # Stage 2: Build agent input
                if _input_builder is not None:
                    agent_input = _input_builder(parsed, ctx)
                    if inspect.isawaitable(agent_input):
                        agent_input = await agent_input
                elif _is_default_schema and isinstance(parsed, AgentToolInput):
                    # Default schema — use the ``input`` string directly.
                    # ``isinstance`` narrows ``parsed`` to ``AgentToolInput``
                    # so pyright can see the attribute without a marker.
                    agent_input = parsed.input
                else:
                    # Custom schema without builder - serialise back to JSON
                    agent_input = json.dumps(parsed.model_dump())

                # Stage 3: Resolve RunConfig (explicit > inherited > None)
                resolved_run_config = _run_config
                inherited_run_config = ctx.get_run_config()
                if resolved_run_config is None and inherited_run_config is not None:
                    resolved_run_config = inherited_run_config

                # Stage 3b: Merge budget into RunConfig
                if _budget is not None:
                    resolved_run_config = _merge_budget(
                        resolved_run_config,
                        _budget,
                    )

                # Stage 4: Run the sub-agent
                from troopai.adk.run import Runner

                async def _run_sub_agent() -> Any:
                    if _on_stream is not None:
                        # Streaming mode: iterate events, forward to callback
                        sr = await Runner.arun(
                            agent,
                            user_prompt=agent_input,
                            stream=True,
                            context=ctx.context,
                            hooks=_hooks,
                            max_turns=_max_turns,
                            run_config=resolved_run_config,
                        )

                        async for event in sr.stream_events():
                            callback_result = _on_stream(event)
                            if inspect.isawaitable(callback_result):
                                await callback_result
                        return sr
                    else:
                        # Standard mode
                        return await Runner.arun(
                            agent,
                            user_prompt=agent_input,
                            context=ctx.context,
                            hooks=_hooks,
                            max_turns=_max_turns,
                            run_config=resolved_run_config,
                        )

                # Apply timeout governance
                if _timeout is not None:
                    logger.debug(
                        "Sub-agent '%s' running with %.1fs timeout",
                        agent.name,
                        _timeout,
                    )
                    try:
                        result = await asyncio.wait_for(
                            _run_sub_agent(),
                            timeout=_timeout,
                        )
                    except TimeoutError:
                        logger.warning(
                            "Sub-agent '%s' timed out after %.1fs",
                            agent.name,
                            _timeout,
                        )
                        return (
                            f"Sub-agent '{agent.name}' timed out after "
                            f"{_timeout}s. Consider increasing the timeout "
                            f"or simplifying the task."
                        )
                else:
                    result = await _run_sub_agent()

                # Stage 5: Propagate HITL through tool boundary
                if result.requires_action:
                    from troopai.adk.exceptions import AgentToolDeferral

                    if result.deferred_requests is None or result.state is None:
                        raise ValueError(
                            f"Sub-agent '{agent.name}' reported requires_action but "
                            "deferred_requests/state are unset; cannot propagate the deferral."
                        )
                    raise AgentToolDeferral(
                        agent_name=agent.name,
                        deferred_requests=result.deferred_requests,
                        state=result.state,
                    )

                # Stage 6: Extract output
                if _extractor is not None:
                    # Annotated ``object``: the extractor is user-supplied and typed
                    # ``-> str``, but Python does not enforce annotations, so a
                    # misbehaving extractor can return a non-str at runtime. Trusting
                    # the static type would make the isinstance guard below unreachable;
                    # ``object`` keeps it a real trust-boundary validation.
                    output: object = _extractor(result)
                    if inspect.isawaitable(output):
                        output = await output
                    if not isinstance(output, str):
                        from troopai.adk.exceptions import UserError

                        raise UserError(f"as_tool extractor must return str, got {type(output).__name__}")
                    return output

                return str(result.final_output)

            except Exception as e:
                # Let HITL deferrals and programmer errors propagate —
                # AgentToolDeferral is caught by _execute_tool_calls;
                # UserError signals a misconfiguration that must surface.
                from troopai.adk.exceptions import AgentToolDeferral, UserError

                if isinstance(e, (AgentToolDeferral, UserError)):
                    raise

                if _error_function is not None:
                    from troopai.adk.run.context import RunContext

                    err_ctx = RunContext(context=ctx.context)
                    err_result = _error_function(err_ctx, e)
                    if inspect.isawaitable(err_result):
                        return await err_result
                    return err_result

                return f"Error running {agent.name} agent: {type(e).__name__}: {e}"

        tool = FunctionTool(
            name=resolved_name,
            description=resolved_description,
            schema=resolved_schema,
            schema_enforcement=resolved_enforcement,
            guardrails=guardrails,
            enabled=enabled,
            requires_approval=requires_approval,
            max_result_tokens=max_result_tokens,
            on_invoke=_on_invoke_tool,
        )
        # Set internal delegate reference (not a constructor field)
        object.__setattr__(tool, "_agent", agent)
        return tool


def _build_agent_graph(
    agent: Agent[Any],
    visited: set[str],
) -> dict[str, Any]:
    """Recursively build an agent relationship graph.

    Handles cycles by tracking visited agent names.  Delegate agents
    are discovered via ``get_delegate_tools()`` (tools created with
    ``as_tool()``).  Handoff targets come from ``agent.handoffs``.

    Args:
        agent: The agent to build the graph node for.
        visited: Set of agent names already visited in the current
            traversal path; used to detect and break cycles.

    Returns:
        A dict with keys ``name`` (str), ``delegates`` (list of
        recursively-built sub-graphs), and ``handoffs`` (list of
        handoff-target names).
    """
    if agent.name in visited:
        return {"name": agent.name, "delegates": [], "handoffs": ["(cycle)"]}

    visited = visited | {agent.name}

    # Discover delegate sub-agents from tools
    delegates: list[dict[str, Any]] = []
    for tool in agent.get_delegate_tools():
        delegate_agent = tool.get_delegate_agent()
        if delegate_agent is not None:
            delegates.append(_build_agent_graph(delegate_agent, visited))

    # Discover handoff targets
    handoff_names: list[str] = []
    if isinstance(agent.handoffs, list):
        for h in agent.handoffs:
            if isinstance(h, Handoff):
                handoff_names.append(h.target.name)
            elif isinstance(h, Agent):
                handoff_names.append(h.name)
    elif isinstance(agent.handoffs, HandoffRoute):
        # Code-orchestrated routing — targets live inside the route's rules
        # and ``otherwise`` fallback, surfaced via ``all_targets()``.
        for handoff_target in agent.handoffs.all_targets():
            handoff_names.append(handoff_target.target.name)

    return {
        "name": agent.name,
        "delegates": delegates,
        "handoffs": handoff_names,
    }


def _merge_budget(
    config: RunConfig | None,
    budget: LLMUsageLimits,
) -> RunConfig:
    """Create or update a RunConfig with the given usage limits.

    If *config* is ``None``, returns a new ``RunConfig`` with only
    ``usage_limits`` set.  If *config* already has ``usage_limits``,
    the budget parameter takes precedence (explicit sub-agent governance
    overrides inherited config).

    Args:
        config: Existing ``RunConfig`` to update, or ``None`` to create
            a fresh one with only usage limits applied.
        budget: Usage limits to set on the returned config.

    Returns:
        A ``RunConfig`` with ``usage_limits`` set to *budget*.
    """
    from troopai.adk.run import RunConfig as _RunConfig

    if config is None:
        return _RunConfig(usage_limits=budget)
    # Frozen-safe: RunConfig is a regular dataclass, use dataclasses.replace
    return dataclasses.replace(config, usage_limits=budget)
