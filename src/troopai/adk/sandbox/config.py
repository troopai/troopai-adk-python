"""Per-run sandbox configuration.

``SandboxRunConfig`` lives on ``RunConfig.sandbox``. It bundles the
fields the Runner consults to acquire and configure a sandbox
session for a single ``Runner.arun`` call.

Every field is typed against the Layer-1 sandbox types; fields that
reference runtime types (``BaseSandboxClient``, ``BaseSandboxSession``,
``SandboxCommandGuardrail``, ``AuditSink``, ``SnapshotStore``) are
typed ``Any`` to avoid load-coupling.

Resolution order (consumed by ``sandbox_run_context``):
1. ``session`` — caller provides a live session; the runner reuses it.
2. ``session_state`` — runner resumes via ``client.resume(state)``.
3. ``client`` + ``manifest`` — runner creates a fresh session.
4. ``selector`` + ``candidates`` — runner picks the cheapest eligible
   backend and creates a fresh session from it.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.sandbox.selector import SandboxCandidate, SandboxSelector
    from troopai.adk.types.sandbox.cost import SandboxRequirements
    from troopai.adk.types.sandbox.iac import IaCBundle
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.network import NetworkPolicy
    from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits
    from troopai.adk.types.sandbox.session_state import SandboxSessionState
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

__all__ = ["SandboxRunConfig"]


@dataclasses.dataclass(kw_only=True)
class SandboxRunConfig:
    """Per-run sandbox configuration attached to ``RunConfig.sandbox``.

    Attributes:
        client: Backend client used to create / resume / delete the
            sandbox session.
        options: Backend-specific client options (e.g. Docker image
            spec, K8s pod template). Opaque to the framework.
        session: Live sandbox session provided by the caller. When
            set, the runner reuses it and does NOT close it on exit.
        session_state: Serialized state the runner reconnects via
            ``client.resume(state)``. Used when ``session`` is None.
        manifest: Fresh-session workspace contract. Used when the
            runner creates a new session (no live session, no resume).
        snapshot: Snapshot spec accepted by the lifecycle for interface
            conformance. Current backends warn and ignore it.
        snapshot_store: Pluggable snapshot store for direct use. Current
            backends reject run-level wiring with
            ``UnsupportedSnapshotFeatureError``.
        resource_limits: CPU / memory / disk / time / process /
            egress caps the backend enforces.
        network_policy: Declarative network access policy. Each
            backend translates to its wire format.
        command_policy: Optional ``SandboxCommandGuardrail`` checked
            before each shell command.
        audit_sink: Pluggable audit sink the runner emits lifecycle +
            command events to.
        iac: Optional IaC bundle the runner applies before the session
            and destroys after.
        selector: Cost-aware backend selector, consulted only when no
            explicit session / session_state / client is provided.
        candidates: Candidate (client, options) pairs the selector ranks.
        requirements: Constraints the selector matches against each
            backend's capabilities.
        capture_live_cost: Opt-in flag to query the chosen client's
            fetch_billing after the run (off by default).
    """

    client: Any | None = None
    """Backend client used to create / resume / delete the session."""

    options: Any | None = None
    """Backend-specific client options (Docker image, K8s pod tmpl, …)."""

    session: Any | None = None
    """Live sandbox session provided by the caller (``BaseSandboxSession``)."""

    session_state: SandboxSessionState | None = None
    """Serialized state for ``client.resume(state)``."""

    manifest: Manifest | None = None
    """Fresh-session workspace contract."""

    snapshot: SnapshotSpec | None = None
    """Snapshot spec accepted by current backends but not restored or persisted."""

    snapshot_store: Any | None = None
    """Pluggable snapshot store; run-level backend wiring raises if set."""

    resource_limits: SandboxResourceLimits | None = None
    """CPU / memory / disk / time / process / egress caps."""

    network_policy: NetworkPolicy | None = None
    """Declarative network access policy."""

    command_policy: Any | None = None
    """Optional ``SandboxCommandGuardrail`` for the shell tool."""

    audit_sink: Any | None = None
    """Pluggable ``AuditSink`` for lifecycle + command events."""

    iac: IaCBundle | None = None
    """IaC bundle the runner applies/destroys around the session."""

    selector: SandboxSelector | None = None
    """Cost-aware backend selector; consulted only when no explicit
    ``session`` / ``session_state`` / ``client`` is set."""

    candidates: list[SandboxCandidate] | None = None
    """Candidate (client, options) pairs the ``selector`` chooses among."""

    requirements: SandboxRequirements | None = None
    """Constraints the selector matches against backend capabilities."""

    capture_live_cost: bool = False
    """Opt-in: query the chosen client's ``fetch_billing`` after the run.
    Off by default — the only network cost the developer must opt into."""

    def __post_init__(self) -> None:
        has_selector_path = self.selector is not None and self.candidates is not None and len(self.candidates) > 0
        if self.session is None and self.session_state is None and self.client is None and not has_selector_path:
            raise ValueError(
                "SandboxRunConfig requires one of: session=, session_state=, "
                "client= (with optional manifest=), or selector= with "
                "candidates= to acquire a sandbox session"
            )
        if self.selector is not None and (self.candidates is None or len(self.candidates) == 0):
            raise ValueError("SandboxRunConfig.selector requires a non-empty candidates= list")
