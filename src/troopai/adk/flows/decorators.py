"""Decorators — ``@flow_start``, ``@flow_listen``, ``@flow_router``.

Each decorator wraps the decorated method in a :class:`FlowStep`
instance. The wrapper carries the decorator role and trigger spec as
data, exposes ``__or__`` / ``__and__`` for fluent combinator
construction, and forwards calls via the descriptor protocol so
``await flow.step()`` works identically to a plain async method.

What this module deliberately does NOT do:

- NEVER injects step-method arguments — step methods take ONLY ``self``.
  CrewAI's :func:`_execute_single_listener` inspects parameter counts
  and injects the previous step's return value when present
  (``lib/crewai/src/crewai/flow/flow.py:3117-3139``). We reject this.
- NEVER auto-persists state — the developer drives
  :class:`FlowCheckpoint` explicitly.
- NEVER treats a bare ``str`` return from a non-router as a route name —
  only ``@flow_router`` returns drive dispatch; other returns are ignored.
- NEVER generates code at runtime — wrappers are constructed via the
  normal class-construction path with no ``exec`` / ``compile``.

Combinator construction is operator-only — there are NO ``or_()`` /
``and_()`` helper functions. Use the ``|`` and ``&`` operators on
:class:`FlowStep` instances::

    @flow_listen(method_a | method_b)             # Or gate
    @flow_listen(method_a & method_b & method_c)  # And gate (flattens left-assoc)
    @flow_listen(method_a | "route_label")        # Mixed FlowStep / str
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypedDict, overload

from troopai.adk.flows.combinators import And, Or
from troopai.adk.flows.flow_wrappers import FlowRole, FlowStep

if TYPE_CHECKING:
    from troopai.adk.flows.approval_policy import FlowApprovalPolicy
    from troopai.adk.flows.step_cache_policy import FlowStepCachePolicy
    from troopai.adk.flows.step_context import FlowStepGate
    from troopai.adk.flows.step_guardrails import FlowStepGuardrails
    from troopai.adk.flows.step_rate_limit import FlowStepRateLimit


class _GateKwargs(TypedDict):
    """Shared decorator kwargs forwarded to :class:`FlowStep`.

    Keeps the three decorator entry points (``flow_start`` /
    ``flow_listen`` / ``flow_router``) under the function-length cap
    by factoring the gate-argument bundle into a single typed
    structure.

    Attributes:
        description: Optional human-readable blurb describing the step.
            Read by the visualisation emitters to label diagram nodes.
        requires_approval: HITL gate — ``False`` to proceed; ``True`` to
            always defer; or a callable receiving a
            :class:`FlowStepContext` and returning a bool.
        approval_policy: Optional declarative approval policy attached
            to the deferral when the gate fires. ``None`` ⇒ bare
            single-approver deferral.
        enabled: Dynamic-skip gate — ``True`` to run; ``False`` to skip;
            or a callable receiving a :class:`FlowStepContext`.
        max_retries: Optional count of extra retry attempts on body
            exception. ``None`` ⇒ no retries.
        timeout: Optional ``asyncio.wait_for`` ceiling in seconds.
            ``None`` ⇒ no timeout.
        rate_limit: Optional per-step sliding-window rate-limit
            configuration. ``None`` ⇒ no rate limit.
        guardrails: Optional pre/post guardrail bundle.
            ``None`` ⇒ no guardrails.
        cache: Optional per-step result-cache policy.
            ``None`` ⇒ no caching.
    """

    description: str | None
    requires_approval: FlowStepGate
    approval_policy: FlowApprovalPolicy | None
    enabled: FlowStepGate
    max_retries: int | None
    timeout: float | None
    rate_limit: FlowStepRateLimit | None
    guardrails: FlowStepGuardrails | None
    cache: FlowStepCachePolicy | None


def _build_step(
    fn: Callable[..., Any],
    *,
    role: FlowRole,
    triggers: tuple[Any, ...],
    description: str | None,
    requires_approval: FlowStepGate,
    approval_policy: FlowApprovalPolicy | None,
    enabled: FlowStepGate,
    max_retries: int | None,
    timeout: float | None,
    rate_limit: FlowStepRateLimit | None,
    guardrails: FlowStepGuardrails | None,
    cache: FlowStepCachePolicy | None,
) -> FlowStep:
    """Construct a :class:`FlowStep` from the decorator-supplied kwargs.

    Single construction site for every decorator branch keeps each
    decorator under the function-length cap and centralises the gate
    propagation contract.
    """
    return FlowStep(
        fn,
        role=role,
        triggers=triggers,
        description=description,
        requires_approval=requires_approval,
        approval_policy=approval_policy,
        enabled=enabled,
        max_retries=max_retries,
        timeout=timeout,
        rate_limit=rate_limit,
        guardrails=guardrails,
        cache=cache,
    )


FlowTriggerSpec = str | Callable[..., Any] | Or | And
"""A single trigger spec accepted by ``@flow_listen`` / ``@flow_router``.

- ``str`` — a step name or route label.
- ``Callable`` — a method reference (typically an unbound method on the
  flow class); the wrapper normalizes it to ``__name__``.
- :class:`Or` / :class:`And` — combinator gate produced by the ``|`` /
  ``&`` operators on :class:`FlowStep` instances.
"""


def _normalize_trigger(trigger: FlowTriggerSpec) -> str | Or | And:
    """Normalize a trigger spec to its stable internal form.

    Strings and callables / :class:`FlowStep` instances flatten to the
    step name string. :class:`Or` / :class:`And` instances pass through
    so the executor can interpret the gate semantics.

    Args:
        trigger: One trigger spec.

    Returns:
        Either a step-name ``str`` or a gate dataclass.

    Raises:
        ValueError / TypeError: See :func:`troopai.adk.flows.flow_wrappers.name_of`.
    """
    if isinstance(trigger, (Or, And)):
        return trigger
    from troopai.adk.flows.flow_wrappers import name_of

    return name_of(trigger)


@overload
def flow_start(fn: Callable[..., Any], /) -> FlowStep: ...


@overload
def flow_start(
    *,
    description: str | None = ...,
    requires_approval: FlowStepGate = ...,
    approval_policy: FlowApprovalPolicy | None = ...,
    enabled: FlowStepGate = ...,
    max_retries: int | None = ...,
    timeout: float | None = ...,
    rate_limit: FlowStepRateLimit | None = ...,
    guardrails: FlowStepGuardrails | None = ...,
    cache: FlowStepCachePolicy | None = ...,
) -> Callable[[Callable[..., Any]], FlowStep]: ...


def flow_start(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    description: str | None = None,
    requires_approval: FlowStepGate = False,
    approval_policy: FlowApprovalPolicy | None = None,
    enabled: FlowStepGate = True,
    max_retries: int | None = None,
    timeout: float | None = None,
    rate_limit: FlowStepRateLimit | None = None,
    guardrails: FlowStepGuardrails | None = None,
    cache: FlowStepCachePolicy | None = None,
) -> FlowStep | Callable[[Callable[..., Any]], FlowStep]:
    """Mark a method as a Flow entry point.

    Multiple ``@flow_start`` methods are allowed on one Flow class; all of
    them fire in parallel when the flow begins execution. Each method
    MUST be ``async def`` taking only ``self`` — :class:`FlowMeta`
    enforces this at class-definition time.

    Supports both call forms::

        @flow_start
        async def kickoff(self) -> None: ...


        @flow_start(description="Seed the topic from input.")
        async def kickoff(self) -> None: ...

    Args:
        fn: The async method to mark as a start point. Pass ``None``
            (the parenthesised form) when supplying keyword-only
            attributes.
        description: Optional human-readable blurb describing what
            the step does. Read by the visualisation emitters to
            label diagram nodes.
        requires_approval: HITL gate. ``False`` (default) ⇒ no gate;
            ``True`` ⇒ always defer the step; a callable receives a
            :class:`FlowStepContext` and returns a bool (sync or
            async). When the gate fires, the executor captures the
            step into the :class:`FlowCheckpoint` and halts.
        approval_policy: Optional :class:`FlowApprovalPolicy` attached
            to the captured :class:`FlowDeferredStep` when
            ``requires_approval`` fires. The executor does NOT evaluate
            quorum / roles / deadline — the out-of-band approval driver
            does (see the :class:`FlowApprovalPolicy` module docstring).
            ``None`` (default) ⇒ bare single-approver deferral.
        enabled: Dynamic step skip. ``True`` (default) ⇒ always run;
            ``False`` ⇒ never run (silently skipped); a callable
            receives a :class:`FlowStepContext` and returns a bool.
        max_retries: Optional retry-on-exception count for the step
            body. ``None`` (default) ⇒ no retries.
        timeout: Optional ``asyncio.wait_for`` ceiling (seconds) on
            the step body. ``None`` (default) ⇒ no timeout.
        rate_limit: Optional :class:`FlowStepRateLimit` sliding-window
            cap for the step body. ``None`` (default) ⇒ no rate limit.
        guardrails: Optional :class:`FlowStepGuardrails` pre/post
            verdict chain around the step body. ``None`` (default) ⇒
            no guardrails.
        cache: Optional :class:`FlowStepCachePolicy` result cache for
            the step. ``None`` (default) ⇒ no caching.

    Returns:
        Bare form: a :class:`FlowStep` wrapping ``fn`` with
        ``__flow_role__ = "start"``. Parenthesised form: a decorator
        producing the same wrapper. The wrapper supports ``|`` and
        ``&`` operators for building gates that reference this step.
    """
    gates: _GateKwargs = {
        "description": description,
        "requires_approval": requires_approval,
        "approval_policy": approval_policy,
        "enabled": enabled,
        "max_retries": max_retries,
        "timeout": timeout,
        "rate_limit": rate_limit,
        "guardrails": guardrails,
        "cache": cache,
    }
    if fn is not None:
        # Runtime guard for users who write @flow_start("something") thinking
        # the string becomes `description`. The overload narrows `fn` to
        # Callable, so pyright treats this branch as unreachable — but at
        # runtime Python happily binds the misplaced positional to `fn`.
        if not callable(fn):  # pyright: ignore[reportUnreachable]
            raise TypeError(
                f"@flow_start: positional argument must be the decorated async function; "
                f"got {type(fn).__name__}. Did you mean @flow_start(description=...)?",
            )
        return _build_step(fn, role="start", triggers=(), **gates)

    def decorator(target: Callable[..., Any]) -> FlowStep:
        return _build_step(target, role="start", triggers=(), **gates)

    return decorator


def flow_listen(
    trigger: FlowTriggerSpec,
    *,
    description: str | None = None,
    requires_approval: FlowStepGate = False,
    approval_policy: FlowApprovalPolicy | None = None,
    enabled: FlowStepGate = True,
    max_retries: int | None = None,
    timeout: float | None = None,
    rate_limit: FlowStepRateLimit | None = None,
    guardrails: FlowStepGuardrails | None = None,
    cache: FlowStepCachePolicy | None = None,
) -> Callable[[Callable[..., Any]], FlowStep]:
    """Mark a method as a listener fired on ``trigger`` arrival.

    The trigger may be:

    - A step name string (``"research"``) — fires when that step
      completes OR when a router returns the string as its label.
    - A method reference (``MyFlow.research``) — resolved to the
      method's ``__name__``.
    - A :class:`FlowStep` (the wrapped method on the class body) —
      same as the method reference form.
    - An :class:`Or` gate built via ``method_a | method_b`` — fires
      ONCE on first arrival in the run.
    - An :class:`And` gate built via ``method_a & method_b`` — fires
      ONCE after every required trigger has arrived.

    Step methods receive only ``self``; return values are IGNORED.
    Only ``@flow_router`` returns drive downstream dispatch.

    Args:
        trigger: One trigger spec — see above.
        description: Optional human-readable blurb describing what the
            step does. Read by the visualisation emitters.
        requires_approval: HITL gate (see :func:`flow_start`).
        approval_policy: Optional :class:`FlowApprovalPolicy` for the
            captured deferral (see :func:`flow_start`). The executor
            does NOT evaluate quorum / roles / deadline — the approval
            driver does.
        enabled: Dynamic skip gate (see :func:`flow_start`).
        max_retries: Step-body retry-on-exception count.
        timeout: ``asyncio.wait_for`` ceiling in seconds for the step body.
        rate_limit: Optional per-step sliding-window rate limit (see
            :func:`flow_start`).
        guardrails: Optional pre/post guardrail bundle (see
            :func:`flow_start`).
        cache: Optional per-step result-cache policy (see
            :func:`flow_start`).

    Returns:
        A decorator that wraps the target function as a
        :class:`FlowStep` with ``__flow_role__ = "listen"`` and the
        trigger stored on ``__flow_triggers__``.

    Raises:
        ValueError: When the trigger string is empty.
        TypeError: When the trigger is not a supported form.
    """
    normalized = _normalize_trigger(trigger)
    gates: _GateKwargs = {
        "description": description,
        "requires_approval": requires_approval,
        "approval_policy": approval_policy,
        "enabled": enabled,
        "max_retries": max_retries,
        "timeout": timeout,
        "rate_limit": rate_limit,
        "guardrails": guardrails,
        "cache": cache,
    }

    def decorator(fn: Callable[..., Any]) -> FlowStep:
        return _build_step(fn, role="listen", triggers=(normalized,), **gates)

    return decorator


def flow_router(
    trigger: FlowTriggerSpec,
    *,
    description: str | None = None,
    requires_approval: FlowStepGate = False,
    approval_policy: FlowApprovalPolicy | None = None,
    enabled: FlowStepGate = True,
    max_retries: int | None = None,
    timeout: float | None = None,
    rate_limit: FlowStepRateLimit | None = None,
    guardrails: FlowStepGuardrails | None = None,
    cache: FlowStepCachePolicy | None = None,
) -> Callable[[Callable[..., Any]], FlowStep]:
    """Mark a method as a router fired on ``trigger`` arrival.

    A router runs after the trigger arrives, executes its body, and
    returns a non-empty ``str`` route label. Downstream methods
    decorated with ``@flow_listen("label")`` then fire on that label.

    Routers are the ONLY way to express conditional dispatch in a
    Flow. A plain ``@flow_listen`` method's return value is dropped — this
    structural rule prevents CrewAI's hidden "bare string return becomes
    next step's name" behavior. If you need routing, write a ``@flow_router``.

    Args:
        trigger: One trigger spec — see :func:`flow_listen` for valid forms.
        description: Optional human-readable blurb (see :func:`flow_start`).
        requires_approval: HITL gate (see :func:`flow_start`).
        approval_policy: Optional :class:`FlowApprovalPolicy` for the
            captured deferral (see :func:`flow_start`). The executor
            does NOT evaluate quorum / roles / deadline — the approval
            driver does.
        enabled: Dynamic skip gate (see :func:`flow_start`).
        max_retries: Step-body retry-on-exception count.
        timeout: ``asyncio.wait_for`` ceiling in seconds for the step body.
        rate_limit: Optional per-step sliding-window rate limit (see
            :func:`flow_start`).
        guardrails: Optional pre/post guardrail bundle (see
            :func:`flow_start`).
        cache: Optional per-step result-cache policy (see
            :func:`flow_start`).

    Returns:
        A decorator that wraps the target function as a
        :class:`FlowStep` with ``__flow_role__ = "router"`` and the
        trigger stored on ``__flow_triggers__``.

    Raises:
        ValueError: When the trigger string is empty.
        TypeError: When the trigger is not a supported form.
    """
    normalized = _normalize_trigger(trigger)
    gates: _GateKwargs = {
        "description": description,
        "requires_approval": requires_approval,
        "approval_policy": approval_policy,
        "enabled": enabled,
        "max_retries": max_retries,
        "timeout": timeout,
        "rate_limit": rate_limit,
        "guardrails": guardrails,
        "cache": cache,
    }

    def decorator(fn: Callable[..., Any]) -> FlowStep:
        return _build_step(fn, role="router", triggers=(normalized,), **gates)

    return decorator
