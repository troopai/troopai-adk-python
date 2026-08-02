"""Tests for ``troopai.adk.sandbox.capabilities.base.SandboxCapability``."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

import pytest
from pydantic import Field

from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.permissions import User


class _TestCapability(SandboxCapability):
    """Concrete capability used by the test suite."""

    type: Literal["test"] = "test"
    payload: dict[str, Any] = Field(default_factory=dict)


class _CapabilityWithLocks(SandboxCapability):
    """Capability holding asyncio + threading primitives (excluded from serialization)."""

    type: Literal["with_locks"] = "with_locks"
    asyncio_lock: Any = Field(default_factory=asyncio.Lock, exclude=True)
    asyncio_event: Any = Field(default_factory=asyncio.Event, exclude=True)
    threading_event: Any = Field(default_factory=threading.Event, exclude=True)
    threading_lock: Any = Field(default_factory=threading.Lock, exclude=True)


class TestBaseFields:
    def test_construction(self) -> None:
        c = _TestCapability()
        assert c.type == "test"
        assert c.session is None
        assert c.run_as is None

    def test_session_and_run_as_excluded_from_dump(self) -> None:
        c = _TestCapability()
        c.bind("fake_session")
        c.bind_run_as(User(name="alice"))
        dumped = c.model_dump()
        assert "session" not in dumped
        assert "run_as" not in dumped


class TestBindAndBindRunAs:
    def test_bind_stores_session(self) -> None:
        c = _TestCapability()
        sentinel = object()
        c.bind(sentinel)
        assert c.session is sentinel

    def test_bind_run_as_stores_user(self) -> None:
        c = _TestCapability()
        alice = User(name="alice")
        c.bind_run_as(alice)
        assert c.run_as is alice

    def test_bind_run_as_accepts_none(self) -> None:
        c = _TestCapability()
        c.bind_run_as(None)
        assert c.run_as is None


class TestClone:
    def test_clone_resets_session_to_none(self) -> None:
        c = _TestCapability()
        c.bind("session-1")
        cloned = c.clone()
        assert cloned.session is None
        # Original retains its bind.
        assert c.session == "session-1"

    def test_clone_creates_fresh_asyncio_lock(self) -> None:
        c = _CapabilityWithLocks()
        cloned = c.clone()
        assert cloned.asyncio_lock is not c.asyncio_lock
        # Both are still asyncio.Lock instances.
        assert isinstance(cloned.asyncio_lock, asyncio.Lock)

    def test_clone_creates_fresh_asyncio_event(self) -> None:
        c = _CapabilityWithLocks()
        c.asyncio_event.set()
        cloned = c.clone()
        # Fresh event is NOT set (state does not leak).
        assert not cloned.asyncio_event.is_set()

    def test_clone_creates_fresh_threading_event(self) -> None:
        c = _CapabilityWithLocks()
        c.threading_event.set()
        cloned = c.clone()
        assert not cloned.threading_event.is_set()

    def test_clone_creates_fresh_threading_lock(self) -> None:
        # Regression: _clone_value used threading.Lock().__class__ (allocates a
        # throwaway lock on every call) — should use _THREADING_LOCK_TYPE cached
        # at module level. Behaviour is identical; this confirms no regression.
        c = _CapabilityWithLocks()
        cloned = c.clone()
        assert cloned.threading_lock is not c.threading_lock
        assert isinstance(cloned.threading_lock, type(threading.Lock()))

    def test_clone_deep_copies_nested_dict(self) -> None:
        c = _TestCapability(payload={"k": [1, 2, 3]})
        cloned = c.clone()
        assert cloned.payload == c.payload
        cloned.payload["k"].append(4)
        # Original's payload is untouched.
        assert c.payload["k"] == [1, 2, 3]


class TestDefaultHooks:
    def test_required_capability_types_empty(self) -> None:
        c = _TestCapability()
        assert c.required_capability_types() == set()

    def test_tools_empty(self) -> None:
        c = _TestCapability()
        assert c.tools() == []

    def test_process_manifest_pass_through(self) -> None:
        c = _TestCapability()
        m = Manifest(root="/workspace")
        assert c.process_manifest(m) is m

    @pytest.mark.asyncio
    async def test_instructions_none(self) -> None:
        c = _TestCapability()
        m = Manifest()
        assert await c.instructions(m) is None

    def test_sampling_params_empty(self) -> None:
        c = _TestCapability()
        assert c.sampling_params({"model": "gpt-4"}) == {}

    def test_process_context_pass_through(self) -> None:
        c = _TestCapability()
        ctx = [{"role": "user", "content": "hi"}]
        assert c.process_context(ctx) == ctx


class TestRequiredCapabilityTypesOverride:
    """A subclass can override required_capability_types declaratively."""

    def test_override_returns_dependencies(self) -> None:
        class _DepCap(SandboxCapability):
            type: Literal["dep"] = "dep"

            def required_capability_types(self) -> set[str]:
                return {"shell", "filesystem"}

        c = _DepCap()
        assert c.required_capability_types() == {"shell", "filesystem"}
