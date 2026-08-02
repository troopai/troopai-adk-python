"""Session-scoped dependency-injection container.

Sandbox clients hold a template ``Dependencies`` instance —
populated with value bindings (long-lived clients, configuration
objects) and factory bindings (lazy per-session producers). For
each new ``BaseSandboxSession`` the client ``clone()``s the
template so every session gets its own resolution cache and
owned-resource lifecycle without recreating the bindings table.

Materialization callbacks (``LocalFile.apply()``,
``S3Mount.apply()``, etc.) call ``await deps.require("name")`` to
read the injected client. ``owns_result=True`` on a factory makes
the container responsible for awaiting ``aclose()`` (or
``close()``) on the produced value when ``aclose()`` runs.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Self, cast

logger = logging.getLogger(__name__)

__all__ = [
    "Dependencies",
    "DependenciesBindingError",
    "DependenciesClosedError",
    "DependenciesError",
    "DependenciesMissingDependencyError",
    "FactoryFn",
]

DependencyKey = str
"""Bindings are keyed by a non-empty string."""

FactoryFn = Callable[["Dependencies"], object | Awaitable[object]]
"""Factory signature: receives the container, returns a value (sync or async)."""


class DependenciesError(RuntimeError):
    """Common base for dependency-container errors."""


class DependenciesClosedError(DependenciesError):
    """Raised when a bind / resolve call lands after ``aclose()``."""


class DependenciesBindingError(DependenciesError, ValueError):
    """Raised when a bind operation conflicts with an existing binding."""


class DependenciesMissingDependencyError(DependenciesError, LookupError):
    """Raised by ``require()`` when no binding satisfies the requested key."""


@dataclass(slots=True)
class _ValueBinding:
    """A static dependency binding for a pre-constructed object."""

    value: object


@dataclass(slots=True)
class _FactoryBinding:
    """A dynamic dependency binding that produces an object on demand."""

    factory: FactoryFn
    cache: bool
    owns_result: bool


_Binding = _ValueBinding | _FactoryBinding


async def _close_best_effort(value: object) -> None:
    """Call ``aclose()`` then ``close()`` on ``value``, swallowing failures.

    Used in the ``aclose()`` teardown path so a buggy close-method
    on one owned resource doesn't prevent the next resource from
    being released. Suppression is narrowed to ``Exception`` so
    ``KeyboardInterrupt`` / ``SystemExit`` still propagate.
    """
    aclose = getattr(value, "aclose", None)
    if aclose is not None:
        try:
            result = aclose()
            if inspect.isawaitable(result):
                # ``isawaitable`` already proved the runtime type; the cast
                # narrows from ``object`` so the static checker accepts the
                # ``await`` — no other invariant is being asserted here.
                await cast(Awaitable[object], result)
            return
        except Exception as exc:
            logger.warning(
                "Dependencies: %r.aclose() raised: %s",
                type(value).__name__,
                exc,
                exc_info=True,
            )
            return

    close = getattr(value, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            # Same invariant as the aclose path above.
            await cast(Awaitable[object], result)
    except Exception as exc:
        logger.warning(
            "Dependencies: %r.close() raised: %s",
            type(value).__name__,
            exc,
            exc_info=True,
        )


class Dependencies:
    """Session-scoped dependency container.

    Sandbox clients build a template ``Dependencies`` once at
    construction time and ``clone()`` it for each created /
    resumed session — the clone shares the binding shape but
    starts with an empty cache and an empty owned-resources list,
    so each session's lifecycle stays isolated.
    """

    def __init__(self) -> None:
        self._bindings: dict[DependencyKey, _Binding] = {}
        self._cache: dict[DependencyKey, object] = {}
        self._owned_results: list[object] = []
        self._closed = False

    @classmethod
    def with_values(
        cls,
        values: Mapping[DependencyKey, object],
    ) -> Dependencies:
        """Construct a container pre-populated with value bindings."""
        dependencies = cls()
        for key, value in values.items():
            dependencies.bind_value(key, value)
        return dependencies

    def bind_value(
        self,
        key: DependencyKey,
        value: object,
        *,
        overwrite: bool = False,
    ) -> Self:
        """Bind ``key`` to a concrete value; chainable for fluent setup."""
        self._raise_if_closed()
        if len(key) == 0:
            raise ValueError("Dependency key must be non-empty")
        self._bind(key, _ValueBinding(value=value), overwrite=overwrite)
        return self

    def bind_factory(
        self,
        key: DependencyKey,
        factory: FactoryFn,
        *,
        cache: bool = True,
        overwrite: bool = False,
        owns_result: bool = False,
    ) -> Self:
        """Bind ``key`` to a (possibly async) factory function.

        Args:
            key: Dependency key.
            factory: Callable receiving the container and returning
                the resolved value (sync or async).
            cache: When True, the factory runs at most once per
                container — subsequent ``get`` calls return the
                cached value.
            overwrite: Pass True to replace an existing binding.
            owns_result: Pass True to make the container responsible
                for awaiting ``aclose()`` / ``close()`` on the
                factory's result when ``aclose()`` runs.
        """
        self._raise_if_closed()
        if len(key) == 0:
            raise ValueError("Dependency key must be non-empty")
        self._bind(
            key,
            _FactoryBinding(factory=factory, cache=cache, owns_result=owns_result),
            overwrite=overwrite,
        )
        return self

    def clone(self) -> Dependencies:
        """Return a fresh container with the same bindings but no cache state.

        Used by sandbox clients when spawning a new session — each
        session needs its own resolution cache + owned-resource
        list so per-session teardown can release one session's
        resources without touching another's.
        """
        cloned = Dependencies()
        for key, binding in self._bindings.items():
            if isinstance(binding, _ValueBinding):
                cloned._bindings[key] = _ValueBinding(value=binding.value)
            else:
                cloned._bindings[key] = _FactoryBinding(
                    factory=binding.factory,
                    cache=binding.cache,
                    owns_result=binding.owns_result,
                )
        return cloned

    def _bind(self, key: DependencyKey, binding: _Binding, *, overwrite: bool) -> None:
        if not overwrite and key in self._bindings:
            raise DependenciesBindingError(f"Dependency `{key}` is already bound")
        self._bindings[key] = binding
        self._cache.pop(key, None)

    async def get(self, key: DependencyKey) -> object | None:
        """Resolve ``key`` to a value, or ``None`` when unbound.

        Note: a factory may legitimately produce ``None``; callers
        that need to distinguish "unbound" from "bound to None"
        MUST use ``has(key)`` or ``require(key)`` instead.
        """
        self._raise_if_closed()
        binding = self._bindings.get(key)
        if binding is None:
            return None
        return await self._resolve(key, binding)

    def has(self, key: DependencyKey) -> bool:
        """True iff ``key`` has a binding (even one whose value is ``None``)."""
        return key in self._bindings

    async def require(
        self,
        key: DependencyKey,
        *,
        consumer: str | None = None,
    ) -> object:
        """Resolve ``key`` or raise ``DependenciesMissingDependencyError``.

        Returns the bound value — including ``None`` when a factory
        legitimately produced ``None``. Only raises when the key is
        not bound at all. Pass ``consumer="<caller name>"`` so the
        error message names the entry that needed the dependency.
        """
        self._raise_if_closed()
        binding = self._bindings.get(key)
        if binding is None:
            consumer_part = f" for {consumer}" if consumer is not None and len(consumer) > 0 else ""
            raise DependenciesMissingDependencyError(
                f"Missing dependency `{key}`{consumer_part}. "
                "Bind it on a Dependencies instance and pass it as "
                "`dependencies=` when constructing the sandbox client."
            )
        return await self._resolve(key, binding)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise DependenciesClosedError("Dependencies container is closed; bind/resolve calls are rejected.")

    async def _resolve(self, key: DependencyKey, binding: _Binding) -> object:
        if isinstance(binding, _ValueBinding):
            return binding.value

        # _Binding union — the only other variant is _FactoryBinding.
        if binding.cache and key in self._cache:
            return self._cache[key]

        produced = binding.factory(self)
        # isawaitable proved the runtime type; cast narrows from object so
        # the static checker accepts the await.
        value = await cast(Awaitable[object], produced) if inspect.isawaitable(produced) else produced

        if binding.cache:
            self._cache[key] = value
        if binding.owns_result:
            self._owned_results.append(value)
        return value

    async def aclose(self) -> None:
        """Release every ``owns_result=True`` value in LIFO order; idempotent.

        After ``aclose()`` returns, subsequent ``get`` / ``require``
        / ``bind_*`` calls raise ``DependenciesClosedError`` — a
        background task that tries to resolve a dependency post-
        teardown gets a loud failure instead of silently leaking
        a freshly-produced owned resource.

        Cancellation safety: every owned resource gets a chance to
        release even when an earlier close raises (including
        ``BaseException`` subclasses like ``CancelledError``). Any
        secondary exceptions are recorded; the first one is
        re-raised after the loop completes so the original
        cancellation signal still propagates.
        """
        if self._closed:
            return
        self._closed = True

        seen_ids: set[int] = set()
        deferred: BaseException | None = None
        for value in reversed(self._owned_results):
            value_id = id(value)
            if value_id in seen_ids:
                continue
            seen_ids.add(value_id)
            try:
                await _close_best_effort(value)
            except BaseException as exc:
                # Continue draining so we don't leak resources further
                # down the LIFO list when one close is cancelled or
                # raises a BaseException subclass.
                logger.warning(
                    "Dependencies.aclose: close of %r aborted: %s",
                    type(value).__name__,
                    exc,
                    exc_info=True,
                )
                if deferred is None:
                    deferred = exc
        if deferred is not None:
            raise deferred
