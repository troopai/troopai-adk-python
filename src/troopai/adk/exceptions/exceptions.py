from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    from troopai.adk.agents.agent_guardrails import AgentInputGuardrailResult, AgentOutputGuardrailResult
    from troopai.adk.run.state import RunState
    from troopai.adk.status.types import AgentQuota
    from troopai.adk.tools.deferred_tool import DeferredToolRequests


class TroopAIError(Exception):
    """Base exception for all TroopAI ADK errors."""

    message: str
    """A human-readable message describing the error."""

    def __init__(self, message: str | None = None):
        """Initialize the error with an optional human-readable message.

        Args:
            message: Human-readable description of the error. Defaults to a
                generic ADK error message when not provided.
        """
        self.message = message or "An error occurred in the TroopAI ADK."
        super().__init__(self.message)

    @override
    def __str__(self) -> str:
        return self.message


class UserError(TroopAIError):
    """
    Exception raised when user-provided configuration or input is invalid.

    This indicates a problem with how the user has configured or used the ADK,
    not an internal ADK error.
    """

    pass


class ConfigError(UserError):
    """Base exception for declarative agent-configuration loading failures."""

    pass


class ConfigParseError(ConfigError):
    """Raised when a config document is malformed or fails schema validation.

    Covers a non-UTF-8 file, invalid JSON text, a non-mapping root, a field
    that violates the config schema (missing required key, wrong type, unknown
    key under strict validation), and a structurally invalid declared graph.
    """

    pass


class ConfigResolutionError(ConfigError):
    """Raised when a string reference in a config cannot be resolved.

    A config file names Python symbols (tool functions, output-schema
    classes, edge-condition predicates) by dotted path, and agents/nodes by
    local name. This is raised when such a name does not resolve to an
    importable object of the expected kind, or names an agent/node absent from
    the topology.
    """

    pass


class MemoryExtractionError(TroopAIError):
    """Raised when the memory extractor cannot parse or validate LLM output.

    Callers can catch this to distinguish a genuine empty extraction result
    from a failure to parse the LLM response (malformed JSON, non-array root,
    etc.).
    """

    pass


class GuardrailTripwireTriggered(TroopAIError):
    """Base exception for guardrail tripwire violations."""

    pass


class AgentInputGuardrailTripwireTriggered(GuardrailTripwireTriggered):
    """
    Exception raised when an input guardrail's tripwire is triggered.

    This indicates that the input to an agent was rejected by a guardrail,
    such as PII detection, jailbreak detection, or off-topic detection.

    Attributes:
        guardrail_result: The result from the guardrail that triggered.
        all_results: All guardrail results collected before the tripwire fired.
    """

    def __init__(
        self,
        guardrail_result: AgentInputGuardrailResult,
        message: str | None = None,
        all_results: list[AgentInputGuardrailResult] | None = None,
    ):
        """Initialize with the triggering guardrail result.

        Args:
            guardrail_result: The result from the guardrail that triggered the
                tripwire.
            message: Optional override for the default tripwire message.
            all_results: All guardrail results collected before the tripwire
                fired, for inspection by the handler.
        """
        self.guardrail_result = guardrail_result
        self.all_results = all_results
        guardrail_name = guardrail_result.guardrail.get_name()
        default_message = f"Input guardrail '{guardrail_name}' tripwire triggered"
        super().__init__(message or default_message)


class AgentOutputGuardrailTripwireTriggered(GuardrailTripwireTriggered):
    """
    Exception raised when an output guardrail's tripwire is triggered.

    This indicates that the output from an agent was rejected by a guardrail,
    such as PII redaction, hallucination detection, or compliance validation.

    Attributes:
        guardrail_result: The result from the guardrail that triggered.
        all_results: All guardrail results collected before the tripwire fired.
    """

    def __init__(
        self,
        guardrail_result: AgentOutputGuardrailResult,
        message: str | None = None,
        all_results: list[AgentOutputGuardrailResult] | None = None,
    ):
        """Initialize with the triggering guardrail result.

        Args:
            guardrail_result: The result from the guardrail that triggered the
                tripwire.
            message: Optional override for the default tripwire message.
            all_results: All guardrail results collected before the tripwire
                fired, for inspection by the handler.
        """
        self.guardrail_result = guardrail_result
        self.all_results = all_results
        guardrail_name = guardrail_result.guardrail.get_name()
        default_message = f"Output guardrail '{guardrail_name}' tripwire triggered"
        super().__init__(message or default_message)


class ToolGuardrailTripwireTriggered(TroopAIError):
    """
    Exception raised when a tool guardrail's raise_exception behavior is triggered.

    This indicates that a tool input or output guardrail rejected the operation
    and requested execution to halt.

    Attributes:
        guardrail_name: Name of the guardrail that triggered.
        output_info: Optional backend-specific structured details about the
            rejection, or ``None`` if not provided.
    """

    def __init__(self, guardrail_name: str, output_info: Any = None, message: str | None = None):
        """Initialize with the name of the triggering guardrail.

        Args:
            guardrail_name: Name of the tool guardrail that triggered the
                tripwire.
            output_info: Optional backend-specific structured details about the
                rejection.
            message: Optional override for the default tripwire message.
        """
        self.guardrail_name = guardrail_name
        self.output_info = output_info
        default_message = f"Tool guardrail '{guardrail_name}' tripwire triggered"
        super().__init__(message or default_message)


class MaxTurnsExceeded(TroopAIError):
    """
    Exception raised when the agent loop exceeds the maximum number of turns.

    This indicates that the agent did not produce a final output within the
    configured max_turns limit. This can happen when:
    - The agent is stuck in a loop
    - The task requires more iterations than allowed
    - There's an infinite tool call loop
    """

    pass


class UsageLimitExceeded(TroopAIError):
    """
    Exception raised when the token usage exceeds the configured limit.

    This indicates that the total number of tokens used in requests and responses
    has exceeded the allowed usage limit for the agent or API key.
    """

    pass


class ToolTimeoutError(TroopAIError):
    """Raised when tool execution exceeds its timeout.

    Attributes:
        tool_name: Name of the tool that timed out.
        timeout: The timeout duration in seconds.
    """

    def __init__(self, tool_name: str, timeout: float, message: str | None = None):
        """Initialize with the name and timeout of the timed-out tool.

        Args:
            tool_name: Name of the tool that exceeded its timeout.
            timeout: The timeout duration in seconds.
            message: Optional override for the default timeout message.
        """
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(message or f"Tool '{tool_name}' timed out after {timeout}s")


class GraphNodeTimeoutError(TroopAIError):
    """Raised when a graph node exceeds its effective per-attempt timeout.

    Attributes:
        node_id: Id of the node that timed out.
        timeout: The per-attempt timeout in seconds.
        attempts: Number of attempts made before giving up.
    """

    def __init__(
        self,
        node_id: str,
        timeout: float,
        attempts: int,
        message: str | None = None,
    ):
        """Initialize with the node id, timeout, and attempt count.

        Args:
            node_id: Id of the graph node that exceeded its timeout.
            timeout: The per-attempt timeout in seconds.
            attempts: Number of attempts made before giving up.
            message: Optional override for the default timeout message.
        """
        self.node_id = node_id
        self.timeout = timeout
        self.attempts = attempts
        super().__init__(message or f"Graph node {node_id!r} timed out after {timeout}s (attempts={attempts})")


class NodeRetriesExhaustedError(TroopAIError):
    """Raised when a graph node fails after exhausting its retry budget.

    Attributes:
        node_id: Id of the node that failed.
        attempts: Number of attempts made (== ``policy.max_attempts``).
        last_error: The exception raised by the final attempt. The
            graph loop raises this error chained from it
            (``raise NodeRetriesExhaustedError(...) from last_error``),
            so ``__cause__`` is the original exception when raised by
            the framework.
    """

    def __init__(
        self,
        node_id: str,
        attempts: int,
        last_error: Exception,
        message: str | None = None,
    ):
        """Initialize with the node id, attempt count, and final error.

        Args:
            node_id: Id of the graph node that exhausted its retries.
            attempts: Total number of attempts made.
            last_error: The exception raised by the final attempt; also
                set as ``__cause__`` by the framework via
                ``raise ... from last_error``.
            message: Optional override for the default exhausted-retries message.
        """
        self.node_id = node_id
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            message
            or f"Graph node {node_id!r} failed after {attempts} attempt(s): {type(last_error).__name__}: {last_error}"
        )


class ToolRetry(TroopAIError):
    """Signal from a tool ``on_invoke`` requesting the LLM to retry with a hint.

    When a tool ``on_invoke`` raises this exception, the executor returns
    the hint message as the tool result instead of an error.  The LLM
    sees the hint and can adjust its next tool call accordingly.

    Unlike a plain error string, ``ToolRetry`` is semantically distinct:
    it does NOT count toward the retry budget (``max_retries``), and
    the hint message is designed to guide the LLM toward a correct call.

    Example::

        @function_tool(name="query_db", description="Query the database")
        def query_db(sql: str) -> str:
            if "DROP" in sql.upper():
                raise ToolRetry("Cannot run DROP statements. Use SELECT only.")
            return execute_sql(sql)

    Attributes:
        hint: The guidance message the LLM will see as the tool result.
    """

    def __init__(self, hint: str):
        """Initialize with the guidance hint for the LLM.

        Args:
            hint: The guidance message the LLM will see as the tool result,
                directing it toward a corrected call.
        """
        self.hint = hint
        super().__init__(hint)


class AgentToolDeferral(TroopAIError):
    """Signal that a sub-agent (running via as_tool()) requires human approval.

    This exception propagates HITL deferrals through as_tool() boundaries.
    When a sub-agent encounters a tool requiring approval, it raises this
    exception instead of returning an error string, allowing the parent's
    _execute_tool_calls to capture the deferral and surface it in the
    parent's RunResult.

    This is an internal signaling mechanism — it should never escape to
    user code. The Runner catches it in _execute_tool_calls and converts
    it to a DeferredToolCall with nested state in metadata.

    Attributes:
        agent_name: Name of the sub-agent that deferred.
        deferred_requests: The sub-agent's deferred tool requests.
        state: The sub-agent's RunState for resumption.
    """

    def __init__(
        self,
        agent_name: str,
        deferred_requests: DeferredToolRequests,
        state: RunState,
    ):
        """Initialize with the deferring sub-agent's identity and state.

        Args:
            agent_name: Name of the sub-agent that requires human approval.
            deferred_requests: The sub-agent's deferred tool requests, carrying
                the pending approvals.
            state: The sub-agent's ``RunState`` at the point of deferral,
                used to resume execution after approval.
        """
        self.agent_name = agent_name
        self.deferred_requests = deferred_requests
        self.state = state
        super().__init__(
            f"Sub-agent '{agent_name}' requires human approval for {len(deferred_requests.approvals)} tool(s)"
        )


class HandoffRejection(TroopAIError):
    """Signal that a handoff cannot proceed, surfacing a message to the LLM.

    Raised by ``Handoff.invoke`` when:

    - Pydantic validation of ``input_type`` rejects the LLM's tool-call
      arguments — always (the LLM should retry with corrected args).
    - The ``input_filter`` or ``on_handoff`` callback raises and the
      handoff's ``HandoffConfig.on_error`` is ``"reject_with_message"``.

    The runner catches this exception in the LLM-orch dispatch path,
    emits a tool-result item carrying :attr:`tool_message`, and lets
    the loop continue so the LLM can react.

    With ``on_error="halt"`` (the default), filter / callback errors
    are NOT wrapped in this exception — they propagate as the original
    exception type and halt the run.

    Attributes:
        handoff_name: The handoff name (typically ``transfer_to_<agent>``).
        tool_message: The message the LLM will see as the tool result.
        cause: The original exception that triggered the rejection
            (also set as ``__cause__`` via ``raise ... from``).
    """

    def __init__(
        self,
        handoff_name: str,
        tool_message: str,
        *,
        cause: Exception,
    ) -> None:
        """Initialize with the handoff name, LLM-visible message, and cause.

        Args:
            handoff_name: The handoff name (typically ``transfer_to_<agent>``).
            tool_message: The message the LLM will see as the tool-result item.
            cause: The original exception that triggered the rejection; also
                set as ``__cause__`` by the framework via ``raise ... from``.
        """
        self.handoff_name = handoff_name
        self.tool_message = tool_message
        self.cause = cause
        super().__init__(f"Handoff '{handoff_name}' rejected: {tool_message}")


class HandoffDefinitionError(UserError):
    """Raised when a Handoff's configuration is rejected at evaluation time.

    Surfaces conditions the framework cannot reasonably guess around:

    - The ``enabled`` callable's signature cannot be introspected
      (e.g. a C-implemented builtin).
    - The ``enabled`` callable accepts no positional argument the
      dispatcher can fill (e.g. a required keyword-only param).
    - The ``enabled`` callable returns an async generator, or any
      value that is not a bool after await.
    - The caller invoked ``build_handoff_tools`` / ``find_handoff_target``
      / ``HandoffRoute.resolve`` without a ``RunContext`` while the
      handoff's ``enabled`` is callable.

    Attributes:
        handoff_name: The handoff name (typically ``transfer_to_<agent>``).
    """

    def __init__(self, handoff_name: str, message: str | None = None) -> None:
        """Initialize with the misconfigured handoff's name.

        Args:
            handoff_name: The handoff name (typically ``transfer_to_<agent>``).
            message: Optional override for the default misconfiguration message.
        """
        self.handoff_name = handoff_name
        super().__init__(message or f"Handoff '{handoff_name}' is misconfigured.")


class ToolDependencyError(TroopAIError):
    """Raised when one or more tools declare runtime requirements that are
    not satisfied at agent construction time.

    A ``FunctionTool`` may declare ``requires_env=("API_KEY",)`` and/or
    ``requires_packages=("requests>=2.30",)``. ``Agent.__post_init__``
    walks every tool, calls ``validate_dependencies()``, and aggregates
    the missing entries into one error. Failing fast at construction
    (rather than at first invocation, deep inside a turn) means the
    misconfiguration is surfaced before any LLM token is spent.

    Attributes:
        agent_name: The agent that failed to satisfy its tool dependencies.
        missing: Mapping ``tool_name -> list[str]`` of unsatisfied
            requirement strings (e.g. ``"env:API_KEY"``,
            ``"package:requests>=2.30"``).
    """

    def __init__(
        self,
        agent_name: str,
        missing: dict[str, list[str]],
        message: str | None = None,
    ) -> None:
        """Initialize with the agent name and its mapping of missing dependencies.

        Args:
            agent_name: Name of the agent whose tools have unsatisfied
                dependencies.
            missing: Mapping of ``tool_name -> list[str]`` where each value is
                a list of unsatisfied requirement strings
                (e.g. ``"env:API_KEY"``, ``"package:requests>=2.30"``).
            message: Optional override for the auto-generated dependency report.
        """
        self.agent_name = agent_name
        self.missing = missing
        if message is None:
            entries = []
            for tool_name, items in missing.items():
                entries.append(f"  - {tool_name}: {', '.join(items)}")
            message = f"Agent '{agent_name}' has tools with unsatisfied dependencies:\n" + "\n".join(entries)
        super().__init__(message)


class DocumentLoadError(TroopAIError):
    """Raised when a document source cannot be read or parsed.

    Covers a missing file or unreachable URL, an unreadable / corrupt
    document, and an extraction failure inside a format loader (PDF, DOCX,
    website, …). The originating exception is chained via ``raise ... from``.

    Attributes:
        source: The path or URL that failed to load.
    """

    source: str
    """The path or URL that failed to load."""

    def __init__(self, source: str, message: str | None = None):
        """Initialize the error.

        Args:
            source: The path or URL that failed to load.
            message: Human-readable detail; a default is derived from
                ``source`` when omitted.
        """
        self.source = source
        super().__init__(message or f"Failed to load document source: {source}")


class UnsupportedDocumentSourceError(TroopAIError):
    """Raised when no loader can handle a given document source.

    A source whose extension / URL shape matches none of the registered
    loaders (e.g. a ``.bin`` file or an unrecognised scheme) raises this
    rather than guessing a loader.

    Attributes:
        source: The path or URL with no matching loader.
    """

    source: str
    """The path or URL with no matching loader."""

    def __init__(self, source: str, message: str | None = None):
        """Initialize the error.

        Args:
            source: The path or URL with no matching loader.
            message: Human-readable detail; a default is derived from
                ``source`` when omitted.
        """
        self.source = source
        super().__init__(message or f"No document loader handles source: {source}")


class ToolsetNameConflictError(TroopAIError):
    """Raised when materialising an agent's toolsets surfaces two tools
    with the same name.

    Toolsets resolve to ``dict[str, FunctionTool]`` per turn via
    ``Toolset.get_tools(ctx)``. ``build_tools()`` flattens every
    toolset entry into the per-turn tool list and aggregates conflicts
    so the developer sees every collision in one error rather than
    re-running the agent and discovering them one at a time.

    Attributes:
        agent_name: The agent whose toolsets produced the conflict.
        conflicts: Mapping ``tool_name -> list[str]`` where each value is
            a list of source descriptions identifying the toolsets (or
            standalone tool entries) that contributed the colliding name.
    """

    def __init__(
        self,
        agent_name: str,
        conflicts: dict[str, list[str]],
        message: str | None = None,
    ) -> None:
        """Initialize with the agent name and its toolset name conflicts.

        Args:
            agent_name: Name of the agent whose toolsets produced the conflict.
            conflicts: Mapping of ``tool_name -> list[str]`` where each value
                is a list of source descriptions identifying the toolsets or
                standalone tool entries that contributed the colliding name.
            message: Optional override for the auto-generated conflict report.
        """
        self.agent_name = agent_name
        self.conflicts = conflicts
        if message is None:
            entries = []
            for tool_name, sources in conflicts.items():
                entries.append(f"  - '{tool_name}' contributed by: {', '.join(sources)}")
            message = f"Agent '{agent_name}' has toolset name conflicts:\n" + "\n".join(entries)
        super().__init__(message)


class TracingDependencyError(TroopAIError):
    """Raised when the OpenTelemetry extra is required but not installed.

    The OTel bridge (:mod:`troopai.adk.tracing.otel`) performs a soft
    import of the ``opentelemetry`` packages at construction time. When
    they are missing, the bridge raises this error with the install
    command the user needs, rather than surfacing a low-level
    ``ImportError`` from deep inside the framework.

    Attributes:
        missing: The missing package name reported by the underlying
            ``ImportError`` (e.g. ``"opentelemetry"``).
    """

    def __init__(self, missing: str, message: str | None = None) -> None:
        """Initialize with the name of the missing package.

        Args:
            missing: The missing package name as reported by the underlying
                ``ImportError`` (e.g. ``"opentelemetry"``).
            message: Optional override for the default install-hint message.
        """
        self.missing = missing
        super().__init__(
            message
            or (
                f"OpenTelemetry tracing requires the '{missing}' package. "
                "Install the optional extra: pip install 'troopai-adk-python[otel]'"
            )
        )


class QuotaExceeded(TroopAIError):
    """Raised when cumulative usage exceeds a configured quota.

    Checked by :class:`~troopai.adk.status.hooks.StatusTrackingHooks`
    in ``on_agent_start`` before any LLM call is made.

    Attributes:
        agent_name: The agent that exceeded its quota.
        quota: The quota that was exceeded.
        current_value: The current accumulated value.
        limit: The limit that was breached.
        resource: The resource type (``"requests"``, ``"total_tokens"``,
            or ``"runs"``).
    """

    def __init__(
        self,
        agent_name: str,
        quota: AgentQuota,
        current_value: int,
        limit: int,
        resource: str,
    ):
        """Initialize with the agent and quota details that were exceeded.

        Args:
            agent_name: Name of the agent that exceeded its quota.
            quota: The ``AgentQuota`` configuration that was breached.
            current_value: The accumulated value at the time of the breach.
            limit: The configured cap that was exceeded.
            resource: The resource type that was measured (``"requests"``,
                ``"total_tokens"``, or ``"runs"``).
        """
        self.agent_name = agent_name
        self.quota = quota
        self.current_value = current_value
        self.limit = limit
        self.resource = resource
        super().__init__(
            f"Quota exceeded for agent '{agent_name}': "
            f"{resource} {current_value} >= {limit} "
            f"(window: {quota.window_seconds}s)"
        )


class TenantBudgetExceeded(TroopAIError):
    """Raised pre-call when a tenant's estimated spend would exceed its budget.

    Attributes:
        tenant_id: The tenant whose budget was exceeded.
        scope: ``"run"`` (per-run cap) or ``"period"`` (per-window cap).
        spend: Accumulated spend before this call (run total or window total).
        budget: The cap that would be breached.
        estimated_cost: The estimated USD of the blocked call.
    """

    def __init__(
        self,
        tenant_id: str,
        scope: Literal["run", "period"],
        spend: float,
        budget: float,
        estimated_cost: float,
    ) -> None:
        """Initialize with the tenant id and spend details that triggered the cap.

        Args:
            tenant_id: The tenant whose budget cap was about to be exceeded.
            scope: ``"run"`` for a per-run cap or ``"period"`` for a
                per-window cap.
            spend: Accumulated spend before this call (run total or window
                total, in USD).
            budget: The configured cap in USD that would be breached.
            estimated_cost: Estimated USD cost of the blocked call.
        """
        self.tenant_id = tenant_id
        self.scope = scope
        self.spend = spend
        self.budget = budget
        self.estimated_cost = estimated_cost
        super().__init__(
            f"Tenant budget exceeded for '{tenant_id}' ({scope}): "
            f"spend {spend:.6f} + est {estimated_cost:.6f} > budget {budget:.6f}"
        )


class ToolNotPermittedForTenant(TroopAIError):
    """Raised when a tenant calls a tool not on its allowlist.

    Hard-deny (the default): the tool provably never executes. Configure
    ``RunConfig.tenant_allowlist_soft_deny`` to return a message to the
    model instead of raising.

    Attributes:
        tenant_id: The tenant whose call was rejected.
        tool_name: The tool that was not permitted.
        agent_name: The agent that attempted the call.
    """

    def __init__(
        self,
        tenant_id: str,
        tool_name: str,
        agent_name: str,
        message: str | None = None,
    ) -> None:
        """Initialize with the tenant, tool, and agent involved in the denial.

        Args:
            tenant_id: The tenant whose allowlist rejected the call.
            tool_name: The tool that was not on the tenant's allowlist.
            agent_name: The agent that attempted to invoke the tool.
            message: Optional override for the default denial message.
        """
        self.tenant_id = tenant_id
        self.tool_name = tool_name
        self.agent_name = agent_name
        super().__init__(
            message or (f"Tool '{tool_name}' is not permitted for tenant '{tenant_id}' on agent '{agent_name}'")
        )


class NoRoutingCandidateError(TroopAIError):
    """Raised when every routing candidate failed (or the list was empty)."""


class ModelRefusalError(TroopAIError):
    """Raised when the model returns a content-policy refusal.

    Providers (e.g. OpenAI) sometimes refuse to generate a response and
    return a structured refusal part instead of text.  The run loop raises
    this exception so callers can distinguish a content-policy refusal from
    a clean empty output or an HITL interruption.

    Attributes:
        refusal: The refusal text from the provider.
    """

    refusal: str
    """The refusal text from the provider."""

    def __init__(self, refusal: str) -> None:
        self.refusal = refusal
        super().__init__(f"Model refused to respond: {refusal}")


# ----- Sandbox errors ---------------------------------------------------------
# The sandbox feature surfaces a substantial error hierarchy. The three top
# branches mirror the OpenAI Agents SDK split (configuration / runtime /
# artifact) so production deployments can write coarse-grained handlers
# against the branch base classes and let backends raise specific subclasses.


class SandboxError(TroopAIError):
    """Base exception for every sandbox-related error.

    Three concrete branches inherit from this: ``SandboxConfigurationError``
    (manifest / config / skill misconfiguration), ``SandboxRuntimeError``
    (per-command / per-session runtime failures), and ``SandboxArtifactError``
    (local-file / git-clone / mount / snapshot materialization failures).
    """

    pass


class SandboxConfigurationError(SandboxError):
    """Manifest / config / skill misconfiguration.

    Raised at session-create time when a manifest, capability config, or
    runtime option is structurally invalid.
    """

    pass


class SandboxSelectionError(SandboxConfigurationError):
    """Raised when no sandbox candidate satisfies a run's requirements.

    Carries a human-readable reason naming the unmet constraint or the
    empty-candidate condition.
    """

    def __init__(self, message: str | None = None) -> None:
        """Initialize with an optional message describing the unmet constraint.

        Args:
            message: Human-readable description of why no candidate was
                selected. Defaults to a generic no-candidate message.
        """
        super().__init__(message or "No sandbox candidate satisfied the run requirements.")


class InvalidManifestPathError(SandboxConfigurationError):
    """A manifest entry path is invalid (absolute, traversal, drive letter)."""

    pass


class InvalidCompressionSchemeError(SandboxConfigurationError):
    """A backend was asked to extract an archive with an unsupported scheme."""

    pass


class ApplyPatchError(SandboxConfigurationError):
    """Base for ``apply_patch`` failures (path resolution + diff parsing)."""

    pass


class SkillsConfigError(SandboxConfigurationError):
    """A ``SkillsCapability`` configuration is structurally invalid."""

    pass


class UnsupportedSandboxClientError(SandboxConfigurationError):
    """The configured backend does not support a requested feature.

    Raised when, for example, a developer asks ``LocalSubprocessSandboxClient``
    to enforce a ``NetworkPolicy(deny_default=True)`` — local subprocesses
    cannot enforce filesystem-level network isolation.
    """

    pass


class UnsupportedMountStrategyError(SandboxConfigurationError):
    """A mount's strategy is incompatible with the target backend.

    Raised at mount-translation time — e.g. a ``DockerVolumeMountStrategy``
    applied to a Kubernetes pod (K8s uses CSI drivers, not Docker volume
    drivers) or to the local-subprocess backend. Surfaced eagerly at
    session-create time rather than as a confusing missing-directory
    failure at first tool invocation.

    Attributes:
        mount_type: The Mount subclass name (e.g. ``"S3Mount"``).
        strategy_type: The strategy discriminator (e.g. ``"docker_volume"``).
        backend: The backend that cannot honor the strategy
            (e.g. ``"k8s"``, ``"local"``).
    """

    def __init__(self, *, mount_type: str, strategy_type: str, backend: str) -> None:
        """Initialize with the mount type, strategy, and incompatible backend.

        Args:
            mount_type: The Mount subclass name (e.g. ``"S3Mount"``).
            strategy_type: The strategy discriminator
                (e.g. ``"docker_volume"``).
            backend: The backend that cannot honor the strategy
                (e.g. ``"k8s"``, ``"local"``).
        """
        self.mount_type = mount_type
        self.strategy_type = strategy_type
        self.backend = backend
        super().__init__(
            f"{mount_type} uses mount strategy {strategy_type!r} which the "
            f"{backend!r} backend cannot attach; choose a strategy this "
            f"backend supports"
        )


class UnsupportedMountPatternError(SandboxConfigurationError):
    """A Mount subclass + in-container mount pattern have no translation.

    Raised when an ``InContainerMountStrategy`` pairs a mount with a
    pattern that cannot serve it — e.g. ``BoxMount`` with
    ``MountpointMountPattern`` (``mount-s3`` is S3-only). Returning a
    no-op spec would silently omit the mount; raising surfaces the
    misconfiguration at translation time.

    Attributes:
        mount_type: The Mount subclass name (e.g. ``"BoxMount"``).
        pattern_type: The pattern discriminator (e.g. ``"mountpoint"``).
    """

    def __init__(self, *, mount_type: str, pattern_type: str) -> None:
        """Initialize with the incompatible mount type and pattern.

        Args:
            mount_type: The Mount subclass name (e.g. ``"BoxMount"``).
            pattern_type: The pattern discriminator
                (e.g. ``"mountpoint"``).
        """
        self.mount_type = mount_type
        self.pattern_type = pattern_type
        super().__init__(
            f"{mount_type} has no in-container mount translation for pattern "
            f"{pattern_type!r}; this mount subclass and pattern are incompatible"
        )


class UnsupportedSnapshotFeatureError(SandboxConfigurationError):
    """A backend was asked to honor a snapshot feature it does not implement.

    Raised at session-create time when a configured snapshot feature
    (``snapshot_store`` persistence, or ``snapshot`` restore) targets
    a backend that has no implementation for it. Silently discarding
    the configured store / spec would lose the caller's
    data-durability intent with no signal; raising surfaces the
    misconfiguration so the caller can select a backend that
    implements it (or drop the configuration).

    Attributes:
        feature: The unsupported feature (``"snapshot_store"`` or
            ``"snapshot"``).
        backend_id: The backend that cannot honor it (e.g.
            ``"k8s_pod"``).
        supported_backends: Backends that DO implement the feature
            (empty when none do yet).
    """

    def __init__(
        self,
        feature: str,
        backend_id: str,
        *,
        supported_backends: tuple[str, ...] = (),
    ) -> None:
        """Initialize with the unsupported feature and the target backend.

        Args:
            feature: The unsupported snapshot feature
                (``"snapshot_store"`` or ``"snapshot"``).
            backend_id: The backend that cannot honor the feature
                (e.g. ``"k8s_pod"``).
            supported_backends: Backends that implement the feature;
                included in the error message when non-empty so the
                caller can pick a supported alternative.
        """
        self.feature = feature
        self.backend_id = backend_id
        self.supported_backends = supported_backends
        if len(supported_backends) > 0:
            supported = ", ".join(sorted(supported_backends))
            message = (
                f"Backend {backend_id!r} does not support the {feature!r} "
                f"snapshot feature. Supported backends: {supported}. "
                f"Select one of those or remove the {feature!r} configuration."
            )
        else:
            message = (
                f"Backend {backend_id!r} does not support the {feature!r} "
                f"snapshot feature (no backend implements it yet); remove "
                f"the {feature!r} configuration."
            )
        super().__init__(message)


class UnsupportedManifestEntryError(SandboxConfigurationError):
    """A manifest entry's concrete type has no materializer.

    Raised by the materialization dispatcher when an entry's type
    matches no known materializer arm. The entry registry and the
    dispatcher are a closed set, so a mismatch means a registered
    entry type was added without wiring its materializer. Raising
    (rather than silently skipping the entry) surfaces the gap loudly
    instead of materializing an incomplete workspace.

    Attributes:
        entry_type: The unhandled entry ``type`` discriminator.
        supported_types: Entry types the dispatcher does handle.
    """

    def __init__(self, entry_type: str, *, supported_types: tuple[str, ...] = ()) -> None:
        """Initialize with the unhandled entry type.

        Args:
            entry_type: The entry ``type`` discriminator that has no
                materializer.
            supported_types: Entry type discriminators the dispatcher
                does handle; included in the error message when
                non-empty.
        """
        self.entry_type = entry_type
        self.supported_types = supported_types
        if len(supported_types) > 0:
            supported = ", ".join(sorted(supported_types))
            message = f"Manifest entry type {entry_type!r} has no materializer. Supported types: {supported}."
        else:
            message = f"Manifest entry type {entry_type!r} has no materializer."
        super().__init__(message)


class SandboxRuntimeError(SandboxError):
    """Per-command / per-session runtime failure inside a sandbox."""

    pass


class SandboxStartFailed(SandboxRuntimeError):
    """A sandbox session failed to start.

    Attributes:
        backend_id: Backend that failed to start the session.
        reason: Short human-readable reason.
        details: Optional backend-specific structured details.
    """

    def __init__(
        self,
        backend_id: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ):
        """Initialize with the failing backend id and reason.

        Args:
            backend_id: The backend that failed to start the session.
            reason: Short human-readable description of why the start failed.
            details: Optional backend-specific structured details about the
                failure; stored as an empty dict when not provided.
        """
        self.backend_id = backend_id
        self.reason = reason
        self.details = details or {}
        super().__init__(f"Sandbox session failed to start ({backend_id}): {reason}")


class SandboxStopFailed(SandboxRuntimeError):
    """A sandbox session failed to stop cleanly.

    Surfaced when stop semantics (snapshot persist) fail. ``aclose`` still
    runs in the lifecycle ``finally`` block so resources are released.
    """

    pass


class ExecFailureError(SandboxRuntimeError):
    """Base for command-execution failures inside a sandbox."""

    pass


class ExecNonZeroError(ExecFailureError):
    """A command exited with a non-zero code AND was configured to raise.

    Most callers should NOT raise on non-zero exits — the ``ExecResult``
    surfaces the exit code and the tool decides. Raise this only when a
    backend internal command (snapshot tar, manifest materialization) fails.
    """

    pass


class ExecTimeoutError(ExecFailureError):
    """A command exceeded its per-command wall-clock timeout."""

    pass


class ExecTransportError(ExecFailureError):
    """The transport carrying a command's stdio failed (network / IPC)."""

    pass


class ExposedPortUnavailableError(SandboxRuntimeError):
    """A port the sandbox tried to expose is unavailable or unmapped."""

    pass


class WorkspaceIOError(SandboxRuntimeError):
    """Base for filesystem-level errors inside the sandbox workspace."""

    pass


class WorkspaceReadNotFoundError(WorkspaceIOError):
    """A read targeted a workspace path that does not exist."""

    pass


class WorkspaceArchiveReadError(WorkspaceIOError):
    """An archive (tar / zip) failed to read or extract."""

    pass


class WorkspaceArchiveWriteError(WorkspaceIOError):
    """An archive (tar / zip) failed to write."""

    pass


class WorkspaceWriteTypeError(WorkspaceIOError):
    """A workspace write received a stream whose ``read()`` returned an unsupported type.

    Sessions normalize write payloads through ``coerce_write_payload``;
    when a chunk is neither ``bytes`` nor ``bytearray`` the adapter
    raises this so backends never silently truncate or misencode.

    Attributes:
        path: The workspace path targeted by the write.
        actual_type: The type name returned by ``stream.read()``
            (e.g. ``"str"``).
    """

    def __init__(self, *, path: object, actual_type: str) -> None:
        """Initialize with the target path and the unexpected type name.

        Args:
            path: The workspace path targeted by the write.
            actual_type: The type name returned by ``stream.read()``
                (e.g. ``"str"``); must be ``bytes`` or ``bytearray``.
        """
        super().__init__(
            f"workspace write at {path!r}: stream.read() returned unsupported type "
            f"{actual_type!r}; expected bytes or bytearray"
        )
        self.path = path
        self.actual_type = actual_type


class PtySessionNotFoundError(SandboxRuntimeError):
    """A ``PtyHandle`` referenced a session that no longer exists."""

    pass


class SandboxConcurrencyError(SandboxRuntimeError):
    """A ``SandboxAgent`` was used concurrently from a second run.

    The per-agent ``SandboxConcurrencyGuard`` raises this when a
    second ``Runner.arun()`` enters before the first releases.
    """

    pass


class SandboxNetworkPolicyViolation(SandboxRuntimeError):
    """The sandbox attempted (or was asked to attempt) a network operation
    forbidden by its ``NetworkPolicy``."""

    pass


class SandboxCommandRejected(SandboxRuntimeError):
    """A command was rejected by a ``SandboxCommandGuardrail``.

    Attributes:
        command: The command that was rejected (truncated for tracing).
        reason: Short human-readable reason.
    """

    def __init__(self, command: str, reason: str):
        """Initialize with the rejected command and the guardrail's reason.

        Args:
            command: The command string that was rejected (may be truncated
                for tracing purposes).
            reason: Short human-readable description of why the command was
                rejected.
        """
        self.command = command
        self.reason = reason
        super().__init__(f"Sandbox command rejected: {reason}")


class SandboxResourceLimitExceeded(SandboxRuntimeError):
    """A ``SandboxResourceLimits`` cap was exceeded.

    Attributes:
        resource: Which limit was exceeded
            (``"cpu"``, ``"memory"``, ``"disk"``, ``"exec_timeout"``,
            ``"session_timeout"``, ``"max_processes"``, ``"max_egress_bytes"``).
        limit: Configured limit value.
        observed: Observed value at the time of the breach.
    """

    def __init__(self, resource: str, limit: int | float, observed: int | float):
        """Initialize with the breached resource, its limit, and the observed value.

        Args:
            resource: The resource type whose cap was exceeded
                (``"cpu"``, ``"memory"``, ``"disk"``, ``"exec_timeout"``,
                ``"session_timeout"``, ``"max_processes"``,
                ``"max_egress_bytes"``).
            limit: The configured cap value.
            observed: The observed value at the time of the breach.
        """
        self.resource = resource
        self.limit = limit
        self.observed = observed
        super().__init__(f"Sandbox resource limit exceeded: {resource}={observed} > {limit}")


class SandboxArtifactError(SandboxError):
    """Base for failures while materializing a manifest entry."""

    pass


class LocalArtifactError(SandboxArtifactError):
    """A ``LocalFile`` / ``LocalDir`` failed to copy from the host."""

    pass


class GitArtifactError(SandboxArtifactError):
    """A ``GitRepo`` clone failed."""

    pass


class MountArtifactError(SandboxArtifactError):
    """A ``Mount`` (S3/GCS/R2/Azure/Box/S3Files) failed to attach."""

    pass


class SnapshotError(SandboxArtifactError):
    """Base for snapshot persistence / restoration failures."""

    pass


class SnapshotPersistError(SnapshotError):
    """A snapshot could not be persisted to its store."""

    pass


class SnapshotRestoreError(SnapshotError):
    """A snapshot could not be restored from its store."""

    pass


class SnapshotNotRestorableError(SnapshotError):
    """A snapshot is structurally present but not restorable.

    The store can address it, but the backend cannot use it to seed a
    fresh session (incompatible manifest hash, corrupted payload, ...).
    """

    pass


class CheckpointConflictError(TroopAIError):
    """Raised when a checkpoint save loses a concurrent-write race.

    The persisted state for this thread_id was modified by another
    writer since this checkpointer last observed it. Reload the latest
    state and retry the operation if appropriate.

    Attributes:
        thread_id: The logical run identifier that experienced the conflict.
    """

    def __init__(self, thread_id: str) -> None:
        """Initialize with the thread id that experienced the write conflict.

        Args:
            thread_id: The logical run identifier whose persisted state was
                concurrently modified.
        """
        self.thread_id = thread_id
        super().__init__(f"Concurrent modification detected for thread_id={thread_id!r}; reload and retry.")


class SessionAppendConflictError(TroopAIError):
    """Raised when a strict-concurrency session handle detects a concurrent write.

    Another writer appended to this session after this handle last loaded or
    appended.  Reload the session via its manager and retry.

    Attributes:
        session_id: The session identifier that experienced the conflict.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize with the session id that experienced the conflict.

        Args:
            session_id: The session whose rows were concurrently modified.
        """
        self.session_id = session_id
        super().__init__(
            f"Concurrent write detected for session_id={session_id!r}; reload the session from its manager and retry."
        )
