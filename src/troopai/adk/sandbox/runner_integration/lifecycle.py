"""``sandbox_run_context`` — bracket the agent loop with a sandbox session.

The Runner calls this at the top of ``arun`` when it detects a
``SandboxAgent`` (or a non-None ``RunConfig.sandbox``). Inside the
context manager:

1. The concurrency guard is acquired (raises immediately if a
   second concurrent run targets the same agent).
2. The session is resolved per the documented priority order:
   ``run_config.sandbox.session`` →
   ``run_config.sandbox.session_state`` →
   ``run_config.sandbox.client`` + ``manifest`` →
   ``run_config.sandbox.selector`` + ``candidates``.
3. Cloned capabilities are bound to the live session.
4. Manifest is processed sequentially through capabilities.
5. The agent loop runs.
6. On exit, the session is released (``aclose`` when the runner
   owns the session; left running when the caller injected one).

The function is an ``@asynccontextmanager`` for the common path;
use ``__aexit__`` semantics for cleanup on success and failure.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from troopai.adk.sandbox.observability.audit_sink import SandboxAuditEvent
from troopai.adk.sandbox.observability.observability import SandboxObservability
from troopai.adk.sandbox.runner_integration.capability_lifecycle import (
    apply_command_policy,
    bind_capabilities,
    clone_capabilities,
    process_manifest_through_capabilities,
    validate_required_capability_types,
)
from troopai.adk.sandbox.runner_integration.iac_runner import apply_iac, destroy_iac
from troopai.adk.types.sandbox.cost import SandboxRequirements
from troopai.adk.types.sandbox.usage import SandboxUsage

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.context import RunContext
    from troopai.adk.sandbox.capabilities.base import SandboxCapability
    from troopai.adk.sandbox.clients.base import BaseSandboxClient
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.sandbox.config import SandboxRunConfig
    from troopai.adk.sandbox.runner_integration.concurrency_guard import (
        SandboxConcurrencyGuard,
    )
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.permissions import User

__all__ = ["SandboxLifecycleHandle", "sandbox_run_context"]

logger = logging.getLogger(__name__)


class SandboxLifecycleHandle:
    """Per-run handle yielded by ``sandbox_run_context``.

    Carries the live session + cloned capabilities + processed
    manifest. The Runner consumes these to plug capability tools
    into the agent's tool list and to compose the system prompt.

    Attributes:
        session: The live ``BaseSandboxSession``.
        capabilities: Per-run cloned + bound capabilities.
        manifest: Manifest after capability processing (or None when
            no manifest was provided).
        observability: Run-scoped observability handle (or ``None``
            when no hooks/context were supplied).
        runner_owns_session: True when the runner created the session
            (and therefore must close it on exit); False when the
            caller injected ``session=`` and retains ownership.
        iac_env: The IaC output → env-var mapping from
            ``config.iac`` (empty dict when no IaC bundle was
            configured). The Runner injects these into the agent's
            environment so provisioned-infra endpoints are reachable.
    """

    __slots__ = ("capabilities", "iac_env", "manifest", "observability", "runner_owns_session", "session")

    def __init__(
        self,
        *,
        session: BaseSandboxSession,
        capabilities: list[SandboxCapability],
        manifest: Manifest | None,
        runner_owns_session: bool,
        iac_env: dict[str, str],
        observability: SandboxObservability | None = None,
    ) -> None:
        self.session = session
        self.capabilities = capabilities
        self.manifest = manifest
        self.runner_owns_session = runner_owns_session
        self.iac_env = iac_env
        self.observability = observability


async def _resolve_session(
    config: SandboxRunConfig,
) -> tuple[BaseSandboxSession, bool, BaseSandboxClient | None]:
    """Return (session, runner_owns_session, resolved_client) by priority.

    ``resolved_client`` is the client that owns the session's backend
    identity — the explicit ``config.client``, the resume client, or the
    selector-chosen client. It is ``None`` only for a caller-injected
    session with no client. Resolution order: session > session_state >
    explicit client > selector + candidates.
    """
    # 1. Caller-provided live session — runner does NOT own it.
    if config.session is not None:
        return config.session, False, config.client
    # 2. Session state — runner calls client.resume; runner owns it.
    if config.session_state is not None:
        if config.client is None:
            raise ValueError("SandboxRunConfig.session_state requires a non-None client to resume the session")
        session = await config.client.resume(config.session_state)
        return session, True, config.client
    # 3. Fresh create. Explicit client beats the selector (explicit > automatic).
    client = config.client
    options = config.options
    if client is None and config.selector is not None and config.candidates is not None:
        requirements = config.requirements if config.requirements is not None else SandboxRequirements()
        chosen = config.selector.select(config.candidates, requirements)
        client = chosen.client
        options = chosen.options
    if client is None:
        raise ValueError("SandboxRunConfig requires session, session_state, or client")
    # Pass options only when set; omitting lets a backend apply its own
    # default (or raise its clear required-options error). Backends with a
    # required options= param must receive a real value here. snapshot /
    # snapshot_store / manifest are forwarded only when non-None so backends
    # get no spurious None argument in place of their default.
    extra: dict[str, Any] = {}
    if config.snapshot is not None:
        extra["snapshot"] = config.snapshot
    if config.snapshot_store is not None:
        extra["snapshot_store"] = config.snapshot_store
    if config.manifest is not None:
        extra["manifest"] = config.manifest
    if options is not None:
        extra["options"] = options
    session = await client.create(**extra)
    return session, True, client


async def _provision_iac(config: SandboxRunConfig) -> dict[str, str]:
    """Apply ``config.iac`` (if any) and return its output → env map.

    Returns an empty dict when no IaC bundle is configured. A
    provisioning failure raises ``SandboxConfigurationError`` (from
    ``apply_iac``) — surfaced through the session-owned ``try`` so
    the runner still tears the session down.
    """
    if config.iac is None:
        return {}
    return await apply_iac(config.iac)


async def _emit_audit(
    config: SandboxRunConfig,
    *,
    event_type: Literal["start", "stop", "error"],
    agent_name: str | None,
    session: BaseSandboxSession,
    resolved_client: BaseSandboxClient | None,
    error: str | None = None,
) -> None:
    """Best-effort sandbox lifecycle audit emission.

    No-op when ``config.audit_sink`` is None. Any ``Exception`` from
    ``AuditSink.emit`` is suppressed — an audit-backend hiccup MUST
    NOT take down the run nor mask the agent-loop exception (same
    ``suppress(Exception)`` contract as the teardown steps;
    ``BaseException`` such as cancellation still propagates).
    ``backend_id`` is the resolved client's stable id, or
    ``"injected"`` for a caller-supplied session (no client owns its
    backend identity); ``agent_name`` falls back to ``"<unknown>"``
    (the Runner always passes ``agent.name``).
    """
    sink = config.audit_sink
    if sink is None:
        return
    event = SandboxAuditEvent(
        event_type=event_type,
        agent_name=agent_name if agent_name is not None else "<unknown>",
        backend_id=resolved_client.backend_id if resolved_client is not None else "injected",
        session_id=session.session_id,
        error=error,
    )
    try:
        await sink.emit(event)
    except Exception:
        # Best-effort: a sink hiccup must not fail the run nor mask the
        # loop exception. Logged at DEBUG so the drop is diagnosable.
        logger.debug("sandbox lifecycle audit emit failed; suppressed (event_type=%s)", event_type, exc_info=True)


def _build_observability(
    config: SandboxRunConfig,
    *,
    session: BaseSandboxSession,
    resolved_client: BaseSandboxClient | None,
    tracing_enabled: bool,
    hooks: RunHooks[Any] | None,
    run_context: RunContext[Any] | None,
    agent: Agent[Any] | None,
) -> SandboxObservability:
    """Construct the run-scoped observability handle from resolved run state."""
    return SandboxObservability(
        backend_id=resolved_client.backend_id if resolved_client is not None else "injected",
        tracing_enabled=tracing_enabled,
        usage=SandboxUsage(),
        session_id=session.session_id,
        audit_sink=config.audit_sink,
        cost=resolved_client.cost if resolved_client is not None else None,
        hooks=hooks,
        context=run_context,
        agent=agent,
    )


async def _prepare_handle(
    config: SandboxRunConfig,
    cloned: list[SandboxCapability],
    *,
    run_as: User | None,
    session: BaseSandboxSession,
    runner_owns_session: bool,
    resolved_client: BaseSandboxClient | None,
    agent_name: str | None,
    observability: SandboxObservability,
) -> SandboxLifecycleHandle:
    """Bind capabilities, process the manifest, start + provision, audit.

    Called INSIDE the caller's session-owned ``try`` so any failure
    here (``session.start``, ``_provision_iac``) still routes through
    the ``finally`` that tears the session down — no orphaned session.
    Emits the ``"start"`` audit event AFTER provisioning succeeds so
    the audit reflects a fully-ready run.
    """
    bind_capabilities(cloned, session=session, run_as=run_as, observability=observability)
    processed_manifest = (
        process_manifest_through_capabilities(cloned, config.manifest) if config.manifest is not None else None
    )
    if runner_owns_session:
        await session.start()
        if config.manifest is not None and config.session_state is None:
            # Materialize the declared workspace AFTER backend resources
            # are up and BEFORE the agent loop, so File/Dir/LocalFile/
            # LocalDir/GitRepo entries exist when the agent runs. Inside
            # this session-owned try → a materialization failure
            # (e.g. LocalArtifactError / GitArtifactError /
            # ExecNonZeroError / UnsupportedManifestEntryError, or an
            # untranslated backend SandboxError from the sandbox-side
            # write) propagates loudly through the finally teardown; it
            # is NOT wrapped as SandboxStartFailed (a bad workspace is
            # semantically distinct from a backend that would not
            # start). Injected sessions (runner_owns_session
            # False) are the caller's to populate and are skipped. The
            # `session_state is None` clause restricts materialization
            # to the FRESH create path: per the Manifest contract a
            # resumed session keeps its existing workspace state and
            # the manifest is fresh-session-only, so it is NOT
            # re-materialized on resume (no pointless no-op call, no
            # spurious hosted-bridge POST).
            await session.apply_manifest(only_ephemeral=False)
    iac_env = await _provision_iac(config)
    await observability.on_start(session)
    await _emit_audit(
        config, event_type="start", agent_name=agent_name, session=session, resolved_client=resolved_client
    )
    return SandboxLifecycleHandle(
        session=session,
        capabilities=cloned,
        manifest=processed_manifest,
        runner_owns_session=runner_owns_session,
        iac_env=iac_env,
        observability=observability,
    )


async def _teardown(
    session: BaseSandboxSession,
    config: SandboxRunConfig,
    *,
    runner_owns_session: bool,
    resolved_client: BaseSandboxClient | None,
    observability: SandboxObservability,
    agent_name: str | None = None,
    exc: BaseException | None = None,
) -> None:
    """Best-effort run finalization — never masks the loop exception.

    Runs in the lifecycle ``finally``. When ``capture_live_cost`` is set,
    live billing is fetched first (best-effort / suppressed — a billing
    endpoint failure must never fail the run) so the stop hook sees
    ``billed_cost_usd`` on the usage accumulator. Order: billing →
    ``on_stop`` → stop audit → ``aclose``.

    ``exc`` is the in-flight agent-loop exception (or None on a clean
    exit), captured by the caller in the generator ``finally`` itself
    and passed in explicitly — NOT re-derived here via
    ``sys.exc_info()``. The generator frame is the direct ``athrow``
    target where the exception is unambiguously set; reading it across
    the ``await`` into this coroutine would depend on CPython
    exc-info-propagation behaviour that is the subject of an open,
    version-spanning upstream issue. Explicit passing makes the
    ``"error"``/``"stop"`` selection correct on every supported
    interpreter. Then closes the session (only when the runner owns
    it) and destroys IaC infra (only when configured AND
    ``IaCBundle.destroy_on_exit``). Audit emission and each teardown
    step are independently exception-suppressed so none can hide the
    loop exception.
    """
    if config.capture_live_cost and resolved_client is not None:
        try:
            record = await resolved_client.fetch_billing(session)
            if record is not None:
                observability.usage.billed_cost_usd = record.cost_usd
        except Exception:
            # Best-effort: a billing-endpoint failure must never fail the
            # run. Logged at DEBUG so a developer who opted into
            # capture_live_cost can tell a thrown error apart from a
            # backend that simply reports no per-sandbox cost (None).
            logger.debug(
                "sandbox live billing fetch failed; suppressed (backend=%s)",
                resolved_client.backend_id,
                exc_info=True,
            )
    try:
        await observability.on_stop(session)
    except Exception:
        # Best-effort: a developer on_sandbox_stop hook must not mask the
        # loop exception. Logged at DEBUG so a buggy hook is diagnosable.
        logger.debug("sandbox on_stop hook failed; suppressed (backend=%s)", observability.backend_id, exc_info=True)
    await _emit_audit(
        config,
        event_type="error" if exc is not None else "stop",
        agent_name=agent_name,
        session=session,
        resolved_client=resolved_client,
        error=str(exc) if exc is not None else None,
    )
    if runner_owns_session:
        with contextlib.suppress(Exception):
            await session.aclose()
    if config.iac is not None and config.iac.destroy_on_exit:
        with contextlib.suppress(Exception):
            await destroy_iac(config.iac)


@asynccontextmanager
async def sandbox_run_context(
    *,
    config: SandboxRunConfig,
    capabilities: list[SandboxCapability],
    run_as: User | None,
    concurrency_guard: SandboxConcurrencyGuard | None,
    agent_name: str | None = None,
    agent: Agent[Any] | None = None,
    run_context: RunContext[Any] | None = None,
    hooks: RunHooks[Any] | None = None,
    tracing_enabled: bool = False,
) -> AsyncIterator[SandboxLifecycleHandle]:
    """Bracket an agent loop run with a sandbox session lifecycle.

    Args:
        config: The ``RunConfig.sandbox`` value (must be non-None).
        capabilities: The SandboxAgent's capabilities (will be cloned
            per-run before binding).
        run_as: Optional model-facing user identity bound to every
            cloned capability.
        concurrency_guard: Per-agent ``SandboxConcurrencyGuard``;
            ``None`` skips the guard (single-call test paths).
        agent_name: Stamped on audit events; the Runner passes
            ``agent.name``. ``None`` → ``"<unknown>"``.
        agent: Active agent forwarded to the observability handle.
        run_context: Active ``RunContext`` forwarded to the observability handle.
        hooks: Composed run hooks forwarded to the observability handle.
        tracing_enabled: Mirror of ``RunConfig.tracing_enabled``.

    Yields:
        A ``SandboxLifecycleHandle`` carrying the live session,
        cloned-and-bound capabilities, and the post-process manifest.

    The context-manager exit releases the concurrency guard and runs
    ``_teardown`` (exit audit + ``session.aclose`` when the runner
    owns the session + IaC destroy), propagating any exception that
    bubbled out of the agent loop.
    """
    if concurrency_guard is not None:
        await concurrency_guard.acquire()
    try:
        validate_required_capability_types(capabilities)
        cloned = clone_capabilities(capabilities)
        apply_command_policy(cloned, config.command_policy)
        session, runner_owns_session, resolved_client = await _resolve_session(config)
        observability = _build_observability(
            config,
            session=session,
            resolved_client=resolved_client,
            tracing_enabled=tracing_enabled,
            hooks=hooks,
            run_context=run_context,
            agent=agent,
        )
        try:
            handle = await _prepare_handle(
                config,
                cloned,
                run_as=run_as,
                session=session,
                runner_owns_session=runner_owns_session,
                resolved_client=resolved_client,
                agent_name=agent_name,
                observability=observability,
            )
            yield handle
        finally:
            # exc captured HERE (the athrow target — version-safe);
            # never re-derived inside _teardown. See _teardown docstring.
            await _teardown(
                session,
                config,
                runner_owns_session=runner_owns_session,
                resolved_client=resolved_client,
                observability=observability,
                agent_name=agent_name,
                exc=sys.exc_info()[1],
            )
    finally:
        if concurrency_guard is not None:
            concurrency_guard.release()
