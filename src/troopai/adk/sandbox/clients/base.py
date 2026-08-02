"""``BaseSandboxClient`` — the abstract sandbox backend contract.

Every concrete backend (``LocalSubprocessSandboxClient``,
``DockerSandboxClient``, ``K8sPodSandboxClient``,
``RemoteVMSandboxClient`` and its hosted-bridge subclasses) inherits
from this ABC and implements four abstract methods: ``create``,
``delete``, ``resume``, ``deserialize_session_state``. A concrete
default ``serialize_session_state`` covers the common JSON path.

The ``ClientOptionsT`` type parameter binds the client to its
options dataclass — Docker pins ``DockerSandboxClientOptions``,
K8s pins ``K8sSandboxClientOptions``, etc. ``BaseSandboxClientOptions``
is the Pydantic frozen base every concrete option set inherits from
so JSON round-trip is uniform.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from troopai.adk.exceptions.exceptions import UnsupportedSnapshotFeatureError
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.cost import SandboxBillingRecord, SandboxCostDescriptor
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.session_state import SandboxSessionState
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

__all__ = [
    "BaseSandboxClient",
    "BaseSandboxClientOptions",
    "ClientOptionsT",
    "reject_unsupported_snapshot_store",
    "warn_discarded_snapshot",
]


ClientOptionsT = TypeVar("ClientOptionsT")
"""Type parameter binding a client to its options dataclass."""


def reject_unsupported_snapshot_store(snapshot_store: object, backend_id: str) -> None:
    """Raise if a backend was handed a ``snapshot_store`` it cannot honor.

    No sandbox backend implements snapshot-store persistence. Every
    backend's ``create()`` calls this BEFORE doing any work: a
    configured store silently discarded would be a data-durability
    lie, so a non-None ``snapshot_store`` raises
    ``UnsupportedSnapshotFeatureError``. This is the single source of
    the no-store-persistence contract — when a backend gains real
    store support it stops calling this (or a future variant passes
    ``supported_backends``).
    """
    if snapshot_store is not None:
        raise UnsupportedSnapshotFeatureError("snapshot_store", backend_id)


def warn_discarded_snapshot(
    snapshot: object,
    backend_id: str,
    logger: logging.Logger,
) -> None:
    """Log a warning when a configured ``snapshot`` is being discarded.

    No sandbox backend implements snapshot restore. Unlike
    ``snapshot_store`` (which raises), ``snapshot`` is forwarded by
    the lifecycle, so a non-None ``snapshot`` is accepted for
    ABC conformance and explicitly discarded — but NOT silently: the
    backend logs a warning so a configured-but-ignored snapshot is
    operator-visible. (Raising would break callers who set
    ``config.snapshot`` expecting the current no-op.) Single source of the
    snapshot-discard contract.
    """
    if snapshot is not None:
        logger.warning(
            "backend %r does not implement snapshot restore; the "
            "configured snapshot is discarded (select a backend that "
            "supports it, or remove config.snapshot)",
            backend_id,
        )


class BaseSandboxClientOptions(BaseModel):
    """Polymorphic base for sandbox client option dataclasses.

    Concrete option classes (``DockerSandboxClientOptions``,
    ``K8sSandboxClientOptions``, …) inherit from this so JSON
    serialization / deserialization works uniformly across backends.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class BaseSandboxClient(abc.ABC, Generic[ClientOptionsT]):
    """Abstract base every concrete sandbox backend extends.

    Subclasses MUST set ``backend_id`` to a unique stable string
    identifying the backend in tracing + audit + session-state
    payloads.

    Class attributes:
        backend_id: Stable identifier used in tracing + session
            state. Subclasses MUST override.
    """

    backend_id: str = "base"
    """Backend identifier (subclasses MUST override)."""

    cost: SandboxCostDescriptor | None = None
    """Static rate card for cost-aware selection. ``None`` means unpriced
    (the selector treats unpriced backends as more expensive than any
    priced one)."""

    capabilities: SandboxBackendCapabilities = SandboxBackendCapabilities()
    """What this backend can do, matched against ``SandboxRequirements``
    during selector filtering. The conservative default declares no
    network and an ephemeral workspace; each backend overrides it in its
    class body with its real capability surface."""

    @abc.abstractmethod
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: ClientOptionsT,
    ) -> BaseSandboxSession:
        """Create and return a new sandbox session.

        Args:
            snapshot: Optional snapshot spec to restore the workspace
                from at session-start time.
            snapshot_store: Optional pluggable snapshot store (a
                ``SnapshotStore``) the backend reads + writes snapshot
                bytes through. ``None`` ⇒ the backend performs no
                snapshot persistence.
            manifest: Optional fresh-session workspace contract. When
                provided AND no snapshot restore is available, the
                backend materializes this manifest during start.
            options: Backend-specific options (image, namespace, …).

        Returns:
            A live ``BaseSandboxSession`` ready for ``await session.start()``
            (or use as async context manager).
        """

    @abc.abstractmethod
    async def delete(self, session: BaseSandboxSession) -> BaseSandboxSession:
        """Release the backend resources for ``session``.

        Returns the session in its post-delete state so callers can
        inspect final usage / timings before discarding the handle.
        """

    @abc.abstractmethod
    async def resume(self, state: SandboxSessionState) -> BaseSandboxSession:
        """Re-attach to a session previously persisted as ``state``.

        Providers first try to reach the original backend sandbox
        identified by ``state``. If the resource is gone, providers
        MAY recreate the backend sandbox and hydrate it from
        ``state.snapshot`` during ``session.start``.
        """

    def serialize_session_state(self, state: SandboxSessionState) -> dict[str, Any]:
        """Serialize backend-specific state into a JSON-compatible payload.

        Default implementation calls ``state.model_dump(mode="json")``.
        Override only when the backend needs custom JSON shaping
        beyond Pydantic's default.
        """
        return state.model_dump(mode="json")

    async def fetch_billing(
        self,
        session: BaseSandboxSession,
    ) -> SandboxBillingRecord | None:
        """Return provider-reported cost for ``session``, or ``None``.

        The default returns ``None`` (no live billing). Hosted providers
        override this to query their usage endpoint; the sandbox lifecycle
        calls it only when live-cost capture is enabled on the run config.
        """
        del session
        return None

    @abc.abstractmethod
    def deserialize_session_state(self, payload: dict[str, Any]) -> SandboxSessionState:
        """Inverse of ``serialize_session_state``."""
