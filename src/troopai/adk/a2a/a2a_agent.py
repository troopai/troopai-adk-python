"""``A2AAgent`` — a remote A2A endpoint as a ``BaseAgent`` peer.

``A2AAgent`` is a sibling of :class:`troopai.adk.agents.Agent` — both
extend :class:`troopai.adk.agents.BaseAgent`. The split signals to a
reader (and to the type system) that an A2A agent is a different kind
of orchestration entity than a local agent: it is **remote**,
**non-deterministic**, and **opaque** — no shared context window, no
shared guardrails, no shared tools. It speaks JSON-RPC over HTTP, has
its own internal LLM and reasoning, and answers via the A2A protocol.

``A2AAgent`` is **pure config**. It has no execution methods —
mirroring how local :class:`Agent`, :class:`Swarm`, and
:class:`Graph` are also pure configs. Execution against a remote A2A
peer flows through :class:`troopai.adk.a2a.a2a_runner.A2ARunner`, the
sibling of :class:`troopai.adk.run.runner.Runner`. ``A2ARunner``
accepts ONLY ``A2AAgent`` instances; for local primitives, use
``Runner``.

Two developer-facing surfaces on the agent itself:

* :meth:`A2AAgent.as_tool` — wrap the remote agent as a
  :class:`FunctionTool` so a parent local ``Agent``'s LLM can invoke
  it mid-turn as if it were any other tool. Mirrors the
  :meth:`Agent.as_tool` surface 1:1. The wrapping ``FunctionTool``
  flows through the standard tool execution path (middleware,
  tracing, hooks); the inner call dispatches via
  :meth:`A2ARunner.arun` and never enters :meth:`Runner.arun`.
* :meth:`A2AAgent.get_client` — public accessor for the lazily
  constructed :class:`A2AClient`. Used internally by
  :class:`A2ARunner`; exposed because some advanced flows (custom
  interceptor swaps, connection-state inspection) need the client
  directly.

``A2ARunner.arun`` does **not** enter :class:`Runner.arun`. The
Runner manages LLM calls, tool execution, context tracking,
handoffs, and guardrails for a local agent it controls turn-by-turn
— none of which applies to a remote endpoint. ``A2ARunner.arun``
makes a direct HTTP call via the underlying :class:`A2AClient`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, Required

from typing_extensions import TypedDict

try:
    from a2a.client import ClientCallInterceptor, ClientConfig
    from a2a.types import AgentCard
except ImportError as ie:
    if ie.name is not None and ie.name != "a2a" and not ie.name.startswith("a2a."):
        # A transitive dependency of an installed a2a-sdk failed — surface
        # the real error instead of mislabeling it "extra not installed".
        raise
    raise ImportError(
        "Please install the 'a2a' extra to use A2A protocol support. Run: pip install 'troopai-adk-python[a2a]'",
        # name="a2a" lets optional-extra guards (e.g. the graphs adapter's
        # A2A fallthrough) recognize the missing extra and degrade gracefully.
        name="a2a",
    ) from ie

from troopai.adk.a2a.a2a_client import (
    DEFAULT_MAX_STREAM_BYTES,
    DEFAULT_MAX_STREAM_CHUNKS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    A2AClient,
)
from troopai.adk.a2a.a2a_continuation_token import A2ATaskStateLiteral
from troopai.adk.agents import BaseAgent

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.tools.tool_guardrails import ToolGuardrails

logger = logging.getLogger(__name__)

# Sentinel placed in _client during close() to block concurrent get_client()
# calls from receiving a client that is actively being torn down.
_CLOSING: object = object()


# ----------------------------------------------------------------------
# Public framework-typed result + stream-event types
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class A2ARunResult:
    """Outcome of a non-streaming :meth:`A2ARunner.arun` call.

    Attributes:
        text: The remote agent's final output text.
        task_id: The remote task identifier (for correlation /
            tracing). Surfaced even on success so callers can re-use
            it as a ``reference_task_id`` in a follow-up turn.
        context_id: The conversation context identifier.
    """

    text: str
    """The remote agent's final output text."""

    task_id: str
    """Remote task identifier — useful for correlation."""

    context_id: str
    """Conversation context identifier."""


class A2AStreamEvent(TypedDict, total=False):
    """A single event yielded by :meth:`A2ARunner.arun` in streaming mode.

    Discriminator field ``type`` selects which other fields are populated:

    * ``"text_delta"`` — incremental text chunk in ``text_delta``.
    * ``"status"`` — non-terminal state transition reflected in
      ``state`` (one of the framework state literals).
    * ``"completed"`` — terminal success; ``result_text`` carries the
      full accumulated response.
    * ``"failed"`` — terminal failure; ``state`` and ``message`` describe
      the cause. The non-streaming :meth:`A2ARunner.arun` path raises
      :class:`A2ATaskError` directly for terminal failures.

    Attributes:
        type: Discriminator — one of ``"text_delta"``, ``"status"``,
            ``"completed"``, ``"failed"``. Required on every event
            (``Required[Literal[...]]``).
        text_delta: Incremental text chunk. Populated for
            ``type='text_delta'`` only.
        state: Task-state literal. Populated for ``type='status'``
            and ``type='failed'``.
        message: Failure detail from the remote agent. Populated for
            ``type='failed'``.
        result_text: Full accumulated response text. Populated for
            ``type='completed'``.
    """

    type: Required[Literal["text_delta", "status", "completed", "failed"]]
    """Discriminator."""

    text_delta: str
    """Incremental text chunk (populated for ``type='text_delta'``)."""

    state: A2ATaskStateLiteral
    """State literal (populated for ``type='status'`` and ``type='failed'``)."""

    message: str
    """Failure detail (populated for ``type='failed'``)."""

    result_text: str
    """Full accumulated text (populated for ``type='completed'``)."""


# ----------------------------------------------------------------------
# A2AAgent dataclass
# ----------------------------------------------------------------------


# Standard two-pass camel-to-snake. Splits "HTTPClient" → "HTTP_Client"
# (acronym followed by Pascal-case word), then "ResearchBot" → "Research_Bot"
# (lowercase-then-uppercase boundary).
_CAMEL_ACRONYM_BOUNDARY = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_LOWER_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_IDENT_RUN = re.compile(r"[^a-z0-9_]+")
_UNDERSCORE_RUN = re.compile(r"_+")


def _snake_case(name: str) -> str:
    """Convert ``Name`` / ``camelName`` / ``HTTPClient`` / ``Mixed Name`` to ``snake_case``.

    Two camel-case passes (acronym-aware) + a non-identifier collapse
    + a repeat-underscore collapse:

    * ``"Research Bot"`` → ``"research_bot"``
    * ``"HTTPClient"`` → ``"http_client"``
    * ``"my-special.agent"`` → ``"my_special_agent"``
    * ``"ParseURL2HTML"`` → ``"parse_url2_html"``

    Args:
        name: The source name in any mixed format.

    Returns:
        The ``snake_case`` version of ``name``.
    """
    step1 = _CAMEL_ACRONYM_BOUNDARY.sub(r"\1_\2", name)
    step2 = _CAMEL_LOWER_BOUNDARY.sub(r"\1_\2", step1).lower()
    step3 = _NON_IDENT_RUN.sub("_", step2)
    return _UNDERSCORE_RUN.sub("_", step3).strip("_")


@dataclasses.dataclass
class A2AAgent(BaseAgent[Any]):
    """A remote A2A endpoint treated as a ``BaseAgent`` peer.

    Pure config — no execution methods. To dispatch a remote call, use
    :class:`troopai.adk.a2a.a2a_runner.A2ARunner`. To wrap the agent as
    a tool callable from a parent local agent's LLM, use
    :meth:`A2AAgent.as_tool`.

    Attributes:
        name: Agent identifier (inherited from :class:`BaseAgent`).
        description: Human-readable description of what the agent does,
            used by :meth:`as_tool` when no explicit ``tool_description``
            is supplied (inherited from :class:`BaseAgent`).
        tools: Inherited from :class:`BaseAgent` but ignored at run
            time — it describes the *local* side of the orchestration
            peer relationship and is surfaced only to keep API parity
            with :class:`Agent`. A remote agent's own tools are
            entirely opaque to this side.
        url: Base URL of the remote A2A endpoint. Required at
            construction time; an empty URL raises ``ValueError``.
        agent_card: Optional pre-fetched :class:`a2a.types.AgentCard`.
            When ``None``, the card is fetched lazily on first call
            via the well-known URL.
        timeout: Per-call HTTP timeout in seconds. Applied to the
            client's auto-constructed ``httpx.AsyncClient`` when no
            ``client_config`` carries an external one.
        interceptors: List of :class:`a2a.client.ClientCallInterceptor`
            instances inserted into every outbound request. The
            standard place to plug in authentication
            (:class:`a2a.client.AuthInterceptor`).
        client_config: Optional :class:`a2a.client.ClientConfig` for
            full pass-through to the underlying protocol client. Use
            this for connection-pool sharing (set ``httpx_client`` on
            the config), wire-transport selection (set
            ``supported_protocol_bindings``), and every other
            protocol-library knob the a2a SDK exposes. The framework
            does not duplicate these surfaces as flat fields —
            ``ClientConfig`` IS the surface.
        poll_interval: Seconds between polls when waiting on a
            background task — framework-side knob the a2a SDK does not
            surface.
        max_stream_chunks: Maximum streaming chunks before raising
            :class:`A2AProtocolError` — framework-side volume cap.
        max_stream_bytes: Maximum total streamed bytes before raising
            :class:`A2AProtocolError` — framework-side byte cap.
    """

    url: str = ""
    """Base URL of the remote A2A endpoint.

    Required at construction time despite the ``""`` dataclass-field
    default — the default exists so ``BaseAgent``'s defaulted fields
    can come before A2A-specific ones in the generated ``__init__``;
    ``__post_init__`` enforces non-empty.
    """

    agent_card: AgentCard | None = None
    """Optional pre-fetched :class:`a2a.types.AgentCard`.

    When ``None``, the card is fetched lazily on first call via
    :class:`a2a.client.A2ACardResolver` against the well-known URL.
    Supplying it explicitly skips the discovery round trip and is
    useful when the card is already known to the caller (e.g.
    cached from a registry).
    """

    timeout: float = 30.0
    """Per-call HTTP timeout in seconds.

    Applied via the auto-constructed :class:`httpx.AsyncClient` when
    ``client_config`` does not carry one; otherwise the caller's
    ``client_config.httpx_client`` governs and this field is
    ignored.
    """

    interceptors: list[ClientCallInterceptor] = dataclasses.field(default_factory=list)
    """List of :class:`a2a.client.ClientCallInterceptor` instances.

    Inserted into every outbound A2A request. The standard place
    to plug in authentication (:class:`a2a.client.AuthInterceptor`).
    Pure pass-through to the underlying protocol client — no new
    auth surface in this ADK.
    """

    client_config: ClientConfig | None = None
    """Pass-through configuration for the underlying A2A client.

    Set this to a fully-constructed :class:`a2a.client.ClientConfig`
    to control low-level knobs not surfaced as flat fields above:
    ``httpx_client`` (connection pool / shared client),
    ``supported_protocol_bindings`` (wire transport — JSON-RPC vs
    REST vs gRPC), ``streaming``, ``polling``, ``grpc_channel_factory``,
    ``use_client_preference``, ``accepted_output_modes``,
    ``push_notification_config``.

    When ``None`` (default), the underlying client builds a minimal
    config with an auto-constructed httpx client and protocol
    defaults for everything else. When set without an
    ``httpx_client``, the framework splices in its own so card
    resolution and the protocol client share a connection pool.

    .. warning::
       When supplying a ``ClientConfig`` with a custom
       ``httpx_client`` (e.g. one configured for mTLS or a custom
       trust store), the **caller is responsible for that
       client's TLS configuration** — the framework does not
       inspect, override, or warn about ``verify=False`` or
       similar verification-disabling settings. A misconfigured
       httpx client silently weakens transport security. Audit
       any ``httpx.AsyncClient`` you pass through this surface.
    """

    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS
    """Seconds between polls when waiting on a background task.

    Threaded through to the underlying :class:`A2AClient`; reads
    the value when callers drive their own polling loop via
    :meth:`A2ARunner.poll_task`. Lower this for tasks expected to
    complete quickly; raise it for long-running tasks where
    polling overhead matters.
    """

    max_stream_chunks: int = DEFAULT_MAX_STREAM_CHUNKS
    """Maximum streaming chunks before raising :class:`A2AProtocolError`.

    Framework-side bound on the per-call streaming hot path. A
    runaway server that streams forever without a terminal state
    is bounded by this cap so it cannot exhaust memory or hang the
    caller indefinitely.
    """

    max_stream_bytes: int = DEFAULT_MAX_STREAM_BYTES
    """Maximum total streamed bytes before raising :class:`A2AProtocolError`.

    Framework-side bound paired with :attr:`max_stream_chunks`. A
    server emitting huge artifacts is cut off at this byte cap.
    """

    _client: A2AClient | None = dataclasses.field(default=None, init=False, repr=False)
    """Lazily-constructed underlying :class:`A2AClient` — internal cache.

    External code MUST NOT read or write this slot directly; the
    public accessor :meth:`get_client` is the only supported entry
    point. The accessor performs the lazy construction step that
    bare attribute access skips — ``self._client`` is ``None`` until
    the first call, so a direct read on a freshly-constructed agent
    returns ``None``.

    Why the leading underscore on top of ``init=False``? ``init=False``
    only blocks constructor-time passing; it does NOT prevent
    post-init direct mutation or reads. A public name (``client``)
    would invite both. The underscore is the conventional Python
    signal for "do not touch from outside" and is the missing piece
    that turns ``init=False`` into a real privacy guarantee.

    ``repr=False`` keeps the dataclass ``repr`` clean. Assignment
    uses plain ``self._client = ...`` because the dataclass is not
    frozen — no ``object.__setattr__`` workaround needed.
    """

    def __post_init__(self) -> None:
        # Explicit boundary check rather than `assert` (which
        # `python -O` strips). Fails fast on misconfigured A2AAgent
        # before any network traffic is attempted.
        if len(self.url) == 0:
            raise ValueError(
                f"A2AAgent(name={self.name!r}).url MUST be a non-empty URL pointing at a remote A2A endpoint."
            )
        if self.client_config is not None and self.client_config.httpx_client is not None:
            logger.debug(
                "A2AAgent(name=%r, url=%r) constructed with caller-supplied ClientConfig.httpx_client",
                self.name,
                self.url,
            )

    # ------------------------------------------------------------------
    # Internal client lifecycle
    # ------------------------------------------------------------------

    async def get_client(self) -> A2AClient:
        """Return the underlying :class:`A2AClient`, constructing on first use.

        Public accessor for the private :attr:`_client` lazy-init slot.
        Internal state (the cached client) is exposed only via this
        method, never as a direct attribute read — that contract lets
        the lazy-init step run on first access and prevents callers
        from observing the pre-init ``None``. Callers that need to
        swap interceptors, inspect connection state, or share the
        client across requests go through this accessor.

        Returns:
            The lazily-constructed :class:`A2AClient` for this agent.
        """
        if self._client is not None and self._client is not _CLOSING:
            return self._client  # type: ignore[return-value]
        if self._client is _CLOSING:
            raise RuntimeError(f"A2AAgent(name={self.name!r}) is being closed; cannot acquire client.")
        self._client = A2AClient(
            url=self.url,
            agent_card=self.agent_card,
            timeout=self.timeout,
            interceptors=self.interceptors,
            client_config=self.client_config,
            poll_interval=self.poll_interval,
            max_stream_chunks=self.max_stream_chunks,
            max_stream_bytes=self.max_stream_bytes,
        )
        return self._client

    async def close(self) -> None:
        """Release the underlying transport.

        Safe to call multiple times. Idempotent. Closes the internal
        :class:`httpx.AsyncClient` only if the agent was constructed
        without an externally-supplied one.
        """
        if self._client is None or self._client is _CLOSING:
            return
        # Mark as closing BEFORE yielding, so concurrent get_client() calls
        # see the sentinel and raise instead of returning a being-closed client.
        client = self._client
        self._client = _CLOSING  # type: ignore[assignment]
        await client.close()
        self._client = None

    async def __aenter__(self) -> A2AAgent:
        """Async context-manager entry — returns ``self`` unchanged."""
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Async context-manager exit — calls :meth:`close`."""
        del exc
        await self.close()

    # ------------------------------------------------------------------
    # as_tool — mirror of Agent.as_tool()
    # ------------------------------------------------------------------

    def as_tool(
        self,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
        max_result_tokens: int | None = None,
        timeout: float | None = None,
        guardrails: ToolGuardrails | None = None,
        enabled: bool | Callable[[RunContext[Any]], Any] = True,
    ) -> FunctionTool:
        """Wrap this remote agent as a :class:`FunctionTool`.

        Mirrors :meth:`Agent.as_tool`. Returns a tool whose
        ``on_invoke`` dispatches via :meth:`A2ARunner.arun`
        (non-streaming) and returns the result text. Drops cleanly
        into any ``Agent.tools`` list and participates in the
        standard middleware / tracing / hooks plumbing.

        Args:
            tool_name: Override the snake-cased default derived from
                ``self.name``.
            tool_description: Override the auto-generated description
                (falls back to ``self.description`` then to a generic
                "Delegate a task to..." string).
            max_result_tokens: Per-call result-truncation cap.
            timeout: Per-call timeout in seconds wrapping the remote
                call in :func:`asyncio.wait_for`. Distinct from
                ``self.timeout`` (the HTTP client's timeout).
            guardrails: Optional :class:`ToolGuardrails` for the
                wrapping tool — applies from the parent agent's
                perspective; does NOT affect the remote agent's own
                guardrails.
            enabled: Standard ``FunctionTool.enabled`` — boolean or
                a callable receiving the run context.

        Returns:
            A :class:`FunctionTool` whose ``on_invoke`` dispatches to
            this remote agent via :meth:`A2ARunner.arun`.
        """
        # Lazy import: FunctionTool's module pulls a wide tree of
        # framework symbols.
        from troopai.adk.tools.function_tool import FunctionTool

        return FunctionTool(
            name=tool_name if tool_name is not None else _snake_case(self.name),
            schema=_DEFAULT_AS_TOOL_SCHEMA,
            description=self._resolve_tool_description(tool_description),
            guardrails=guardrails,
            enabled=enabled,
            max_result_tokens=max_result_tokens,
            timeout=timeout,
            on_invoke=self._build_tool_on_invoke(),
        )

    def _resolve_tool_description(self, override: str | None) -> str:
        """Resolve the tool description for :meth:`as_tool`.

        Order: explicit override → ``self.description`` (if
        non-empty) → generic "Delegate a task to..." fallback.

        Args:
            override: Caller-supplied description, or ``None`` to use
                the auto-resolved fallback.

        Returns:
            The resolved description string.
        """
        if override is not None:
            return override
        if self.description is not None and len(self.description) > 0:
            return self.description
        return f"Delegate a task to the remote A2A agent {self.name!r}."

    def _build_tool_on_invoke(self) -> Callable[[ToolContext, str], Awaitable[str]]:
        """Build the closure that becomes :class:`FunctionTool.on_invoke`.

        Captures ``self`` so the tool callback can reach the remote
        agent without holding a typed reference on the
        :class:`FunctionTool` itself. Dispatches via
        :class:`A2ARunner.arun` (the canonical execution entry
        point for ``A2AAgent``).

        Returns:
            An async callable ``(ToolContext, str) -> str`` suitable
            for :class:`FunctionTool.on_invoke`.
        """
        # Local import — avoids the import cycle ``a2a_runner``
        # imports ``a2a_agent`` for type narrowing.
        from troopai.adk.a2a.a2a_runner import A2ARunner

        agent_ref = self

        async def _on_invoke_tool(ctx: ToolContext, raw_input: str) -> str:
            del ctx  # The remote agent owns its own context.
            prompt = _extract_prompt_from_raw_input(raw_input)
            result = await A2ARunner.arun(agent_ref, prompt)
            return result.text

        return _on_invoke_tool


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


_DEFAULT_AS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "The prompt to send to the remote agent.",
        },
    },
    "required": ["input"],
    "additionalProperties": False,
}
"""Default JSON schema for ``A2AAgent.as_tool()`` — single-string ``input``."""


def _extract_prompt_from_raw_input(raw_input: str) -> str:
    """Pull ``input`` out of the LLM-supplied JSON tool argument string.

    Mirrors the behaviour of ``Agent.as_tool()`` — the LLM serialises
    a JSON object matching the tool schema, and we extract the single
    ``input`` field. Falls back to the raw text when the JSON is
    malformed or missing the ``input`` key, so a model that emits a
    bare string doesn't break the call.

    Args:
        raw_input: The raw JSON string supplied by the LLM as the tool
            call argument.

    Returns:
        The extracted prompt string, or ``raw_input`` verbatim on
        parse failure.
    """
    if len(raw_input) == 0:
        return ""
    try:
        parsed = json.loads(raw_input)
    except (json.JSONDecodeError, ValueError):
        return raw_input
    if isinstance(parsed, dict):
        candidate = parsed.get("input")
        if isinstance(candidate, str):
            return candidate
    if isinstance(parsed, str):
        return parsed
    return raw_input
