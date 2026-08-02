"""Wrapper class for decorated Flow methods.

When a method is decorated with ``@flow_start`` / ``@flow_listen`` / ``@flow_router``,
the decorator wraps it in a :class:`FlowStep` instance. The wrapper:

1. Carries decorator markers (``__flow_role__`` and
   ``__flow_triggers__``) read by :class:`FlowMeta` at class creation.
2. Implements the descriptor protocol (``__get__``) so the wrapped
   method behaves like a regular bound async method when called on an
   instance: ``await flow.step()`` invokes the wrapped function with
   the instance pre-bound.
3. Implements ``__or__`` and ``__and__`` so combinator gates compose
   fluently::

       @flow_listen(method_a | method_b)             # Or gate
       @flow_listen(method_a & method_b & method_c)  # And gate (left-assoc, flattens)
       @flow_listen(method_a | "route_label")        # Mixed FlowStep / str

4. Preserves ``__name__``, ``__qualname__``, ``__doc__``, ``__module__``,
   and ``__wrapped__`` from the wrapped function so introspection
   (pytest collection, IDE go-to-definition, signature inspection)
   behaves identically to a plain async method.

CrewAI uses a similar pattern (one wrapper class per role:
``StartMethod`` / ``ListenMethod`` / ``RouterMethod`` all extending
``FlowMethod[P, R]``). This codebase uses a single :class:`FlowStep`
class with a ``__flow_role__`` data attribute — the role is data, not
type — to keep the wrapper plumbing small.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, override

FlowRole = Literal["start", "listen", "router"]
"""Discriminator string identifying the decorator that produced a :class:`FlowStep`.

Stored on every :class:`FlowStep` instance as ``__flow_role__``. The
:class:`FlowMeta` metaclass reads this to classify each decorated
method into the registry's ``starts`` / ``listeners`` / ``routers``
buckets. The :class:`troopai.adk.flows.executor.FlowExecutor` reads it
to decide whether to capture a router's return value as a route label.
"""

if TYPE_CHECKING:
    from troopai.adk.flows.approval_policy import FlowApprovalPolicy
    from troopai.adk.flows.combinators import And, Or
    from troopai.adk.flows.step_cache_policy import FlowStepCachePolicy
    from troopai.adk.flows.step_context import FlowStepGate
    from troopai.adk.flows.step_guardrails import FlowStepGuardrails
    from troopai.adk.flows.step_rate_limit import FlowStepRateLimit


def name_of(obj: Any) -> str:
    """Extract a step-name string from a :class:`FlowStep`, callable, or string.

    Args:
        obj: A :class:`FlowStep` wrapper (the class-body value of a
            decorated method), a method reference, or a step-name
            string.

    Returns:
        The step name.

    Raises:
        ValueError: When ``obj`` is an empty string.
        TypeError: When ``obj`` is none of the supported forms.
    """
    if isinstance(obj, str):
        if len(obj) == 0:
            raise ValueError("Step name string must be non-empty.")
        return obj
    if isinstance(obj, FlowStep):
        return obj.__name__
    if callable(obj) and hasattr(obj, "__name__"):
        name = obj.__name__
        if not isinstance(name, str):
            raise TypeError(
                f"Callable {obj!r} has non-string __name__: {name!r}",
            )
        return name
    raise TypeError(
        f"Cannot extract step name from {type(obj).__name__}: {obj!r}. "
        f"Expected a method reference, a step-name string, or a FlowStep wrapper.",
    )


class FlowStep:
    """Descriptor wrapping a decorated Flow method.

    Built by :func:`troopai.adk.flows.decorators.flow_start` /
    :func:`troopai.adk.flows.decorators.flow_listen` /
    :func:`troopai.adk.flows.decorators.flow_router`. The wrapper is:

    - **Callable**: implements ``__call__`` so ``await flow.step()``
      invokes the wrapped async function with the bound instance.
    - **A descriptor**: ``__get__`` returns a new bound
      :class:`FlowStep` instance when accessed via an instance,
      enabling normal Python method-call semantics.
    - **Composable**: ``__or__`` / ``__and__`` build :class:`Or` /
      :class:`And` gates from method references in the class body.

    Args:
        fn: The async function being wrapped.
        role: Decorator role — one of ``"start"`` / ``"listen"`` /
            ``"router"``.
        triggers: Trigger specs declared by the decorator. Strings,
            :class:`Or` instances, or :class:`And` instances. ``@flow_start``
            has no triggers.
        description: Optional human-readable blurb describing what
            the step does. Mirrors :attr:`FunctionTool.description`.
            Read by the visualisation emitters to label nodes; falls
            back to the method name when ``None``.
    """

    _fn: Callable[..., Any]
    """The wrapped async function (private; access via :attr:`wrapped_function`)."""

    _instance: Any
    """Bound instance for descriptor-protocol method binding; ``None`` when unbound."""

    __flow_role__: FlowRole
    """Decorator role identifier — discriminated literal type."""

    __flow_triggers__: tuple[Any, ...]
    """Triggers declared by the decorator; strings, :class:`Or`, or :class:`And` instances."""

    __flow_description__: str | None
    """Optional human-readable blurb attached by the decorator (``description=`` kwarg)."""

    __flow_requires_approval__: FlowStepGate
    """HITL gate; see :data:`FlowStepGate` for the typed shape."""

    __flow_approval_policy__: FlowApprovalPolicy | None
    """Optional declarative approval policy attached to this step's deferral."""

    __flow_enabled__: FlowStepGate
    """Dynamic-skip gate; see :data:`FlowStepGate` for the typed shape."""

    __flow_max_retries__: int | None
    """Extra retry budget on body exceptions; ``None`` ⇒ no retries."""

    __flow_timeout__: float | None
    """``asyncio.wait_for`` ceiling on the step body in seconds; ``None`` ⇒ no timeout."""

    __flow_rate_limit__: FlowStepRateLimit | None
    """Optional per-step sliding-window rate limit."""

    __flow_guardrails__: FlowStepGuardrails | None
    """Optional pre/post guardrail bundle attached to the step."""

    __flow_cache__: FlowStepCachePolicy | None
    """Optional per-step result-cache policy."""

    __name__: str
    """Wrapped function's name; copied from ``fn.__name__``."""

    __qualname__: str
    """Wrapped function's qualified name; copied from ``fn.__qualname__``."""

    __wrapped__: Callable[..., Any]
    """Reference to the wrapped function (``functools`` convention)."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        role: FlowRole,
        triggers: tuple[Any, ...] = (),
        description: str | None = None,
        requires_approval: FlowStepGate = False,
        approval_policy: FlowApprovalPolicy | None = None,
        enabled: FlowStepGate = True,
        max_retries: int | None = None,
        timeout: float | None = None,
        rate_limit: FlowStepRateLimit | None = None,
        guardrails: FlowStepGuardrails | None = None,
        cache: FlowStepCachePolicy | None = None,
    ) -> None:
        if max_retries is not None and max_retries < 0:
            raise ValueError(f"FlowStep.max_retries must be >= 0 when set, got {max_retries}")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"FlowStep.timeout must be > 0 when set, got {timeout}")
        self._fn = fn
        self._instance = None
        self.__flow_role__ = role
        self.__flow_triggers__ = triggers
        self.__flow_description__ = description
        self.__flow_requires_approval__ = requires_approval
        self.__flow_approval_policy__ = approval_policy
        self.__flow_enabled__ = enabled
        self.__flow_max_retries__ = max_retries
        self.__flow_timeout__ = timeout
        self.__flow_rate_limit__ = rate_limit
        self.__flow_guardrails__ = guardrails
        self.__flow_cache__ = cache
        self.__name__ = getattr(fn, "__name__", "")
        self.__qualname__ = getattr(fn, "__qualname__", self.__name__)
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__module__ = getattr(fn, "__module__", "")
        self.__wrapped__ = fn
        if inspect.iscoroutinefunction(fn):
            self._is_coroutine = asyncio.coroutines._is_coroutine  # type: ignore[attr-defined]  # Private but stable since Py3.8; signals "this is awaitable" to asyncio.

    def __get__(
        self,
        instance: Any,
        owner: type | None = None,
    ) -> FlowStep:
        """Descriptor protocol — bind to ``instance`` when accessed via ``flow.step``.

        Accessing the step on the class (``MyFlow.step``) returns
        ``self`` so ``__or__`` / ``__and__`` and :class:`FlowMeta`
        introspection work directly. Accessing via an instance returns
        a new bound :class:`FlowStep` whose ``_instance`` is set so
        ``__call__`` invokes the wrapped function with ``self``
        pre-bound.

        Args:
            instance: The Flow instance, or ``None`` for class access.
            owner: The Flow class (unused).

        Returns:
            ``self`` for class access; a bound :class:`FlowStep` for
            instance access.
        """
        del owner
        if instance is None:
            return self
        bound = FlowStep.__new__(FlowStep)
        bound._fn = self._fn
        bound._instance = instance
        bound.__flow_role__ = self.__flow_role__
        bound.__flow_triggers__ = self.__flow_triggers__
        bound.__flow_description__ = self.__flow_description__
        bound.__flow_requires_approval__ = self.__flow_requires_approval__
        bound.__flow_approval_policy__ = self.__flow_approval_policy__
        bound.__flow_enabled__ = self.__flow_enabled__
        bound.__flow_max_retries__ = self.__flow_max_retries__
        bound.__flow_timeout__ = self.__flow_timeout__
        bound.__flow_rate_limit__ = self.__flow_rate_limit__
        bound.__flow_guardrails__ = self.__flow_guardrails__
        bound.__flow_cache__ = self.__flow_cache__
        bound.__name__ = self.__name__
        bound.__qualname__ = self.__qualname__
        bound.__doc__ = self.__doc__
        bound.__module__ = self.__module__
        bound.__wrapped__ = self.__wrapped__
        if hasattr(self, "_is_coroutine"):
            bound._is_coroutine = self._is_coroutine  # type: ignore[attr-defined]  # Private but stable since Py3.8.
        return bound

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped async function.

        When bound (``_instance is not None``), invokes
        ``self._fn(self._instance, *args, **kwargs)``. When unbound,
        invokes ``self._fn(*args, **kwargs)`` — useful for tests that
        call the wrapper directly with a manually-constructed self.

        Args:
            *args: Positional arguments forwarded to the wrapped function.
            **kwargs: Keyword arguments forwarded to the wrapped function.

        Returns:
            The wrapped function's return value. ``None`` for ``@flow_start``
            and ``@flow_listen``; ``str`` for ``@flow_router``.
        """
        if self._instance is not None:
            return await self._fn(self._instance, *args, **kwargs)
        return await self._fn(*args, **kwargs)

    def __or__(self, other: Any) -> Or:
        """Build an :class:`Or` gate combining this step with ``other``.

        Args:
            other: A :class:`FlowStep`, an :class:`Or` (whose triggers
                merge), a step-name string, or any callable with
                ``__name__``.

        Returns:
            An :class:`Or` gate with the union of trigger names.

        Raises:
            TypeError: When ``other`` is an :class:`And` — mixed-type
                operator chains are rejected.
        """
        from troopai.adk.flows.combinators import And, Or

        if isinstance(other, And):
            raise TypeError(
                "Cannot combine FlowStep with And via |. Construct Or(...) directly or restructure the chain.",
            )
        if isinstance(other, Or):
            extra = tuple(t for t in other.triggers if t != self.__name__)
            return Or(triggers=(self.__name__, *extra))
        other_name = name_of(other)
        if other_name == self.__name__:
            raise ValueError(
                f"Cannot Or-combine step {self.__name__!r} with itself.",
            )
        return Or(triggers=(self.__name__, other_name))

    def __and__(self, other: Any) -> And:
        """Build an :class:`And` gate combining this step with ``other``.

        Args:
            other: A :class:`FlowStep`, an :class:`And` (whose triggers
                merge), a step-name string, or any callable with
                ``__name__``.

        Returns:
            An :class:`And` gate with the union of trigger names.

        Raises:
            TypeError: When ``other`` is an :class:`Or` — mixed-type
                operator chains are rejected.
        """
        from troopai.adk.flows.combinators import And, Or

        if isinstance(other, Or):
            raise TypeError(
                "Cannot combine FlowStep with Or via &. Construct And(...) directly or restructure the chain.",
            )
        if isinstance(other, And):
            extra = tuple(t for t in other.triggers if t != self.__name__)
            return And(triggers=(self.__name__, *extra))
        other_name = name_of(other)
        if other_name == self.__name__:
            raise ValueError(
                f"Cannot And-combine step {self.__name__!r} with itself.",
            )
        return And(triggers=(self.__name__, other_name))

    @override
    def __repr__(self) -> str:
        """Return a debug-friendly repr distinguishing bound vs. unbound."""
        bound_marker = f" of {self._instance!r}" if self._instance is not None else ""
        return f"<FlowStep[{self.__flow_role__}] {self.__qualname__}{bound_marker}>"

    @property
    def wrapped_function(self) -> Callable[..., Any]:
        """Return the underlying wrapped async function.

        Public accessor used by :class:`FlowMeta` to validate the step's
        signature without reaching into private state. Equivalent to
        ``self.__wrapped__`` for callers preferring the dunder form.
        """
        return self._fn

    @property
    def description(self) -> str | None:
        """Return the optional description set by the decorator.

        Public accessor used by the visualisation emitters and other
        introspection tools to read the human-readable blurb attached
        via the ``description=`` decorator kwarg. ``None`` when no
        description was supplied.
        """
        return self.__flow_description__

    @property
    def role(self) -> FlowRole:
        """Decorator role — one of ``"start"`` / ``"listen"`` / ``"router"``.

        Public accessor over ``__flow_role__`` (see :data:`FlowRole`)
        for introspection tools that should not touch the dunder.
        """
        return self.__flow_role__

    @property
    def triggers(self) -> tuple[Any, ...]:
        """Trigger specs declared by the decorator.

        Tuple of step-name strings, :class:`Or`, or :class:`And`
        instances; empty for ``@flow_start`` steps. Public accessor
        over ``__flow_triggers__``.
        """
        return self.__flow_triggers__

    @property
    def max_retries(self) -> int | None:
        """Optional step-body retry count (``None`` ⇒ no retries)."""
        return self.__flow_max_retries__

    @property
    def timeout(self) -> float | None:
        """Optional ``asyncio.wait_for`` ceiling for the step body."""
        return self.__flow_timeout__

    @property
    def rate_limit(self) -> FlowStepRateLimit | None:
        """Optional per-step sliding-window rate-limit configuration."""
        return self.__flow_rate_limit__

    @property
    def guardrails(self) -> FlowStepGuardrails | None:
        """Optional pre/post guardrail bundle attached to the step."""
        return self.__flow_guardrails__

    @property
    def cache(self) -> FlowStepCachePolicy | None:
        """Optional per-step result-cache policy."""
        return self.__flow_cache__

    @property
    def approval_policy(self) -> FlowApprovalPolicy | None:
        """Optional declarative approval policy for this step's deferral.

        Attached to the :class:`FlowDeferredStep` the executor captures
        when the ``requires_approval`` gate fires, so the out-of-band
        approval driver can enforce quorum / role / deadline semantics.
        ``None`` (default) is the bare single-approver case.
        """
        return self.__flow_approval_policy__

    async def check_enabled(self, ctx: Any) -> bool:
        """Evaluate the ``enabled`` gate against ``ctx``.

        Mirrors :meth:`FunctionTool.check_requires_approval` /
        ``check_enabled`` semantics at the step layer:

        - bool literal → returned as-is.
        - sync callable → invoked with ``ctx``, return coerced to bool.
        - async callable → awaited, return coerced to bool.

        Args:
            ctx: The :class:`FlowStepContext` for the step about to
                fire. Typed as ``Any`` here to avoid a circular import
                with :mod:`troopai.adk.flows.step_context`; callers
                pass a real :class:`FlowStepContext`.

        Returns:
            ``True`` if the step should run; ``False`` to skip it.
        """
        return await _evaluate_gate(self.__flow_enabled__, ctx, default=True)

    async def check_requires_approval(self, ctx: Any) -> bool:
        """Evaluate the ``requires_approval`` gate against ``ctx``.

        Same evaluator contract as :meth:`check_enabled` but returns
        ``True`` when the step requires human approval (deferral),
        ``False`` to proceed without approval.

        Args:
            ctx: The :class:`FlowStepContext` for the step about to
                fire.

        Returns:
            ``True`` to defer the step; ``False`` to run it
            immediately.
        """
        return await _evaluate_gate(self.__flow_requires_approval__, ctx, default=False)


async def _evaluate_gate(gate: Any, ctx: Any, *, default: bool) -> bool:
    """Resolve a :data:`FlowStepGate` value to a bool.

    bool → returned; sync callable → invoked then coerced; async
    callable → awaited then coerced.

    Anything else raises :class:`troopai.adk.flows.exceptions.FlowDefinitionError`
    — silently defaulting on garbage values would fail open on
    safety-critical gates (e.g. a misconfigured
    ``requires_approval`` would silently elevate to "no approval
    needed"). ``default`` documents the cost-conservative semantic
    on each gate field but is NOT used as a runtime fallback.
    """
    if isinstance(gate, bool):
        return gate
    if callable(gate):
        result = gate(ctx)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    from troopai.adk.flows.exceptions import FlowDefinitionError

    raise FlowDefinitionError(
        f"Flow step gate must be bool or Callable[[FlowStepContext], MaybeAwaitable[bool]]; "
        f"got {type(gate).__name__} (value={gate!r}). The cost-conservative semantic for this "
        f"gate is {default}, but silent fallback would mask a misconfigured safety control — "
        f"fix the gate declaration.",
    )
