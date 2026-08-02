"""Tests for ``troopai.adk.sandbox.session.dependencies``."""

from __future__ import annotations

import asyncio
from typing import override

import pytest

from troopai.adk.sandbox.session.dependencies import (
    Dependencies,
    DependenciesBindingError,
    DependenciesClosedError,
    DependenciesMissingDependencyError,
)


class TestBindValue:
    @pytest.mark.asyncio
    async def test_get_returns_bound_value(self) -> None:
        deps = Dependencies()
        deps.bind_value("db_client", "fake-client")
        assert await deps.get("db_client") == "fake-client"

    @pytest.mark.asyncio
    async def test_with_values_factory(self) -> None:
        deps = Dependencies.with_values({"a": 1, "b": 2})
        assert await deps.get("a") == 1
        assert await deps.get("b") == 2

    def test_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Dependencies().bind_value("", "x")

    def test_rejects_duplicate_without_overwrite(self) -> None:
        deps = Dependencies()
        deps.bind_value("k", 1)
        with pytest.raises(DependenciesBindingError, match="already bound"):
            deps.bind_value("k", 2)

    def test_overwrite_allowed_with_flag(self) -> None:
        deps = Dependencies()
        deps.bind_value("k", 1)
        deps.bind_value("k", 2, overwrite=True)
        # No exception is success.


class TestBindFactory:
    @pytest.mark.asyncio
    async def test_factory_resolves_sync(self) -> None:
        deps = Dependencies()
        deps.bind_factory("k", lambda _d: "produced")
        assert await deps.get("k") == "produced"

    @pytest.mark.asyncio
    async def test_factory_resolves_async(self) -> None:
        async def factory(_d: Dependencies) -> str:
            await asyncio.sleep(0)
            return "async-produced"

        deps = Dependencies()
        deps.bind_factory("k", factory)
        assert await deps.get("k") == "async-produced"

    @pytest.mark.asyncio
    async def test_factory_cached_by_default(self) -> None:
        counter = {"n": 0}

        def factory(_d: Dependencies) -> int:
            counter["n"] += 1
            return counter["n"]

        deps = Dependencies()
        deps.bind_factory("k", factory)
        assert await deps.get("k") == 1
        assert await deps.get("k") == 1  # cached
        assert counter["n"] == 1

    @pytest.mark.asyncio
    async def test_factory_uncached_runs_every_call(self) -> None:
        counter = {"n": 0}

        def factory(_d: Dependencies) -> int:
            counter["n"] += 1
            return counter["n"]

        deps = Dependencies()
        deps.bind_factory("k", factory, cache=False)
        assert await deps.get("k") == 1
        assert await deps.get("k") == 2
        assert counter["n"] == 2

    @pytest.mark.asyncio
    async def test_factory_sees_container(self) -> None:
        deps = Dependencies()
        deps.bind_value("base", 10)

        async def factory(container: Dependencies) -> int:
            base = await container.get("base")
            assert isinstance(base, int)
            return base * 2

        deps.bind_factory("derived", factory)
        assert await deps.get("derived") == 20


class TestRequire:
    @pytest.mark.asyncio
    async def test_returns_value_when_bound(self) -> None:
        deps = Dependencies.with_values({"k": "v"})
        assert await deps.require("k") == "v"

    @pytest.mark.asyncio
    async def test_raises_with_consumer_context(self) -> None:
        deps = Dependencies()
        with pytest.raises(DependenciesMissingDependencyError, match=r"`db_client`.*for OrderMaterializer"):
            await deps.require("db_client", consumer="OrderMaterializer")


class TestClone:
    @pytest.mark.asyncio
    async def test_clone_shares_bindings_but_not_cache(self) -> None:
        counter = {"n": 0}

        def factory(_d: Dependencies) -> int:
            counter["n"] += 1
            return counter["n"]

        parent = Dependencies()
        parent.bind_factory("k", factory)
        assert await parent.get("k") == 1

        child = parent.clone()
        # Clone has its own cache — factory fires again.
        assert await child.get("k") == 2
        assert counter["n"] == 2

    @pytest.mark.asyncio
    async def test_clone_is_independent_for_bindings(self) -> None:
        parent = Dependencies()
        parent.bind_value("k", "parent")
        child = parent.clone()
        child.bind_value("k", "child", overwrite=True)
        # Observable behavior: each container resolves to its own value.
        assert await parent.get("k") == "parent"
        assert await child.get("k") == "child"


class TestAclose:
    @pytest.mark.asyncio
    async def test_owns_result_closes_value(self) -> None:
        closed = {"n": 0}

        class _Closeable:
            async def aclose(self) -> None:
                closed["n"] += 1

        deps = Dependencies()
        deps.bind_factory("k", lambda _d: _Closeable(), owns_result=True)
        await deps.get("k")
        await deps.aclose()
        assert closed["n"] == 1

    @pytest.mark.asyncio
    async def test_aclose_idempotent(self) -> None:
        closed = {"n": 0}

        class _Closeable:
            async def aclose(self) -> None:
                closed["n"] += 1

        deps = Dependencies()
        deps.bind_factory("k", lambda _d: _Closeable(), owns_result=True)
        await deps.get("k")
        await deps.aclose()
        await deps.aclose()
        assert closed["n"] == 1

    @pytest.mark.asyncio
    async def test_owns_result_false_skips_close(self) -> None:
        closed = {"n": 0}

        class _Closeable:
            async def aclose(self) -> None:
                closed["n"] += 1

        deps = Dependencies()
        deps.bind_factory("k", lambda _d: _Closeable())
        await deps.get("k")
        await deps.aclose()
        assert closed["n"] == 0

    @pytest.mark.asyncio
    async def test_failing_close_doesnt_block_other_closes(self) -> None:
        order: list[str] = []

        class _Bad:
            @override
            def __repr__(self) -> str:
                return "_Bad"

            async def aclose(self) -> None:
                order.append("bad")
                raise RuntimeError("boom")

        class _Good:
            async def aclose(self) -> None:
                order.append("good")

        deps = Dependencies()
        deps.bind_factory("a", lambda _d: _Bad(), owns_result=True)
        deps.bind_factory("b", lambda _d: _Good(), owns_result=True)
        await deps.get("a")
        await deps.get("b")
        await deps.aclose()
        # Closes happen in LIFO order; the bad close MUST NOT prevent good close.
        assert order == ["good", "bad"]

    @pytest.mark.asyncio
    async def test_bind_factory_after_close_raises(self) -> None:
        # aclose() promises subsequent bind_* calls raise; bind_factory
        # must honor that symmetrically with bind_value, so a teardown-
        # racing task can't register an owned resource the idempotent,
        # already-run aclose() will never drain.
        deps = Dependencies()
        await deps.aclose()
        with pytest.raises(DependenciesClosedError):
            deps.bind_factory("k", lambda _d: object(), owns_result=True)

    @pytest.mark.asyncio
    async def test_bind_value_after_close_raises(self) -> None:
        deps = Dependencies()
        await deps.aclose()
        with pytest.raises(DependenciesClosedError):
            deps.bind_value("k", "v")
