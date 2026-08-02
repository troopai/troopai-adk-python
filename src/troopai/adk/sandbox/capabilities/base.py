"""``SandboxCapability`` — base for composable sandbox extensions.

Pydantic ``BaseModel`` matching OpenAI's ``Capability``. Every
concrete capability (Shell, Filesystem, Skills, Memory, Compaction)
inherits from this and overrides one or more of the no-op hooks:
``tools()``, ``process_manifest()``, ``instructions()``,
``sampling_params()``, ``process_context()``.

Per-run isolation is provided by ``clone()`` — the Runner clones
every capability before binding the live session so two concurrent
runs of the same agent never share session state.
"""

from __future__ import annotations

import asyncio
import copy as _copy
import threading
from typing import TYPE_CHECKING, Any, Self

# Cached once at import time — avoids allocating a throwaway Lock on every
# _clone_value call when detecting threading.Lock fields.
_THREADING_LOCK_TYPE = type(threading.Lock())

from pydantic import BaseModel, ConfigDict, Field

from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.permissions import User

if TYPE_CHECKING:
    from troopai.adk.tools.function_tool import FunctionTool

__all__ = ["SandboxCapability"]


def _clone_value(value: Any) -> Any:
    """Per-field clone with special-case handling.

    Sandbox sessions, asyncio primitives, and threading primitives
    are NOT recursively cloned — they MUST come fresh from the bind
    step (or be None on a clone). Tools and other Pydantic-managed
    values use ``copy.deepcopy``; everything else falls back to
    deepcopy with error suppression for un-serializable values
    (kept as shared references).
    """
    if value is None:
        return None
    # Sandbox session — opaque, never deep-copied.
    if hasattr(value, "_TROOPAI_SANDBOX_SESSION_MARKER"):
        return None
    # asyncio primitives — fresh instances per clone.
    if isinstance(value, (asyncio.Lock, asyncio.Event, asyncio.Semaphore, asyncio.Condition)):
        return type(value)()
    # threading primitives — fresh instances per clone.
    if isinstance(value, threading.Event):
        return threading.Event()
    if isinstance(value, _THREADING_LOCK_TYPE):
        return threading.Lock()
    # Recursive containers.
    if isinstance(value, list):
        return [_clone_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(v) for v in value)
    if isinstance(value, set):
        return {_clone_value(v) for v in value}
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, bytearray):
        return bytearray(value)
    # Generic fallback — best-effort recursive copy.
    try:
        return _copy.deepcopy(value)
    except (TypeError, ValueError):
        # Un-serializable value (file handle, lock-like, ...) — share ref.
        return value


class SandboxCapability(BaseModel):
    """Base Pydantic model for sandbox capabilities.

    Every concrete capability declares a unique ``type`` literal so
    the framework can dispatch by discriminator. The three excluded
    fields (``session``, ``run_as``, ``observability``) are bound by
    the Runner just before the agent loop and never serialized.

    Attributes:
        type: Discriminator string set by each concrete subclass.
        session: Live sandbox session bound by the Runner. Excluded
            from serialization. Typed ``Any`` because
            ``BaseSandboxSession`` lives in the client package;
            framework→client coupling stays loose.
        run_as: Optional user identity model-facing tools run as.
            Excluded from serialization.
        observability: Run-scoped observability handle bound by the
            Runner. Excluded from serialization. Typed ``Any`` so the
            capability base does not load-couple the observability
            package.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str
    """Discriminator string for subclass dispatch."""

    session: Any = Field(default=None, exclude=True)
    """Live sandbox session bound by the Runner; excluded from serialization."""

    run_as: User | None = Field(default=None, exclude=True)
    """Optional user identity model-facing tools run as; excluded."""

    observability: Any = Field(default=None, exclude=True)
    """Run-scoped ``SandboxObservability`` bound by the Runner; excluded.

    Typed ``Any`` for the same loose-coupling reason as ``session`` — the
    observability handle lives in the observability package and must not
    load-couple the capability base."""

    def clone(self) -> Self:
        """Return a per-run copy of this capability.

        Special-cased: asyncio + threading primitives become FRESH
        instances; sandbox sessions become ``None``; tools and other
        nested Pydantic models clone recursively; un-serializable
        values fall back to shared references with no exception
        leakage.
        """
        cloned = self.model_copy(deep=False)
        for name, value in self.__dict__.items():
            object.__setattr__(cloned, name, _clone_value(value))
        # Force-reset session on the clone so a stale bind from the
        # source never leaks into the per-run instance.
        object.__setattr__(cloned, "session", None)
        object.__setattr__(cloned, "observability", None)
        return cloned

    def bind(self, session: Any) -> None:
        """Bind a live sandbox session.

        Called by the Runner once per cloned capability, before any
        ``tools()`` / ``instructions()`` / ``process_manifest()`` calls
        that depend on the session.
        """
        self.session = session

    def bind_run_as(self, user: User | None) -> None:
        """Bind the model-facing user identity (or None)."""
        self.run_as = user

    def bind_observability(self, observability: Any) -> None:
        """Bind the run-scoped observability handle (or ``None``)."""
        self.observability = observability

    def required_capability_types(self) -> set[str]:
        """Return capability ``type`` discriminators that MUST also be present.

        Concrete capabilities use this to declare structural deps
        (e.g. ``MemoryCapability`` requires ``{"shell"}`` or
        ``{"shell", "filesystem"}`` depending on its read/live_update
        configuration). The Runner enforces dependencies at bind time.

        Default: no dependencies.
        """
        return set()

    def tools(self) -> list[FunctionTool]:
        """Return the FunctionTools this capability exposes.

        Capabilities that surface model-facing tools (Shell exposes
        the command-execution tool, Filesystem exposes apply_patch +
        view_image, Skills exposes load_skill) override this.

        Default: no tools.
        """
        return []

    def process_manifest(self, manifest: Manifest) -> Manifest:
        """Transform the manifest before the backend creates the session.

        Capabilities that materialize workspace state at session-start
        time (Skills.process_manifest reserves the skills_path
        namespace; Memory wires the sessions/ + memories/ dirs)
        override this. The Runner folds capabilities sequentially —
        later capabilities see earlier mutations.

        Default: pass-through.
        """
        return manifest

    async def instructions(self, manifest: Manifest | None) -> str | None:
        """Return an instruction fragment to append to the system prompt.

        Async so capabilities can read live workspace state at
        prompt-build time (Memory reads memory_summary.md; Skills
        renders the skill index from the manifest). ``manifest`` is
        ``None`` when no workspace contract was configured for the run.

        Default: no fragment.
        """
        _ = manifest
        return None

    def sampling_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return additional LLM sampling parameters.

        Capabilities that influence model behavior (Compaction
        adjusts compact_threshold based on the model's context
        window) override this. The Runner shallow-merges every
        capability's returned dict into the call's sampling params.

        Default: no extras.
        """
        _ = params
        return {}

    def process_context(self, context: list[Any]) -> list[Any]:
        """Transform the LLM input context before sampling.

        Capabilities that rewrite history (Compaction truncates
        before the most-recent compaction marker) override this.
        Folded sequentially — each capability sees the previous
        capability's output.

        Default: pass-through.
        """
        return context
