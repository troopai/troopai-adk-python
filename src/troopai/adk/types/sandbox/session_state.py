"""Serialized sandbox-session state for cross-process / cross-run resume.

A backend can serialize the state of a live sandbox session into a
``SandboxSessionState`` so the runtime (or a separate worker
process) can reconnect to the same backend resource later. The base
model carries only the provider identifier and the snapshot
reference; provider subclasses add backend-specific routing data
(container id, hosted-provider session token, kube pod name, …).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from troopai.adk.types.sandbox.snapshot import SnapshotRef

__all__ = ["SandboxSessionState"]


class SandboxSessionState(BaseModel):
    """Provider-extensible serialized state of a sandbox session.

    Backends MAY subclass this model to carry backend-specific
    routing data. The base form is sufficient for in-memory handoff
    between two runs in the same process when the backend's ``create``
    is idempotent against an existing resource.

    Attributes:
        backend_id: Backend that produced this state (``"unix_local"``,
            ``"docker"``, ``"k8s_pod"``, hosted-provider name, …).
            Used by the runtime to dispatch to the right
            ``client.resume(state)`` implementation.
        snapshot: Optional address of the workspace snapshot the
            session was last persisted to. The backend's ``resume``
            uses this to re-hydrate the workspace.
        provider_payload: Provider-specific routing data. Opaque to
            the framework — only the matching backend deserializes it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend_id: str
    """Backend that produced this state."""

    snapshot: SnapshotRef | None = None
    """Address of the workspace snapshot the session was last persisted to."""

    provider_payload: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific routing data; framework MUST NOT introspect."""
