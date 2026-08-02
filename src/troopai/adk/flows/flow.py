"""`Flow[StateT]` base class + :class:`FlowMeta` metaclass.

A :class:`Flow` is the declarative configuration of a multi-step
orchestration over a typed state object. Subclasses declare their
steps via the ``@flow_start`` / ``@flow_listen`` / ``@flow_router`` decorators in
:mod:`troopai.adk.flows.decorators`.

A ``Flow`` is configuration; the Runner executes it. So ``Flow``
carries NO ``run()`` / ``arun()`` method. Execution lives on
:meth:`troopai.adk.run.runner.Runner.arun_flow` /
:meth:`troopai.adk.run.runner.Runner.arun_flow_streamed` /
:meth:`troopai.adk.run.runner.Runner.arun_flow_from_checkpoint`.

:class:`FlowMeta` is a minimal metaclass that runs ONCE at class
creation, walks ``cls.__dict__`` (NOT inherited methods — see "Inheritance"
below), and:

1. Validates each decorated method's signature (must be ``async def``,
   must take only ``self``, must not be a ``classmethod`` / ``staticmethod``).
2. Collects ``@flow_start`` / ``@flow_listen`` / ``@flow_router`` registrations into a
   frozen :class:`troopai.adk.flows.registry.FlowStepRegistry`.
3. Stamps the registry on the class as ``cls.__flow_registry__``.

What the metaclass deliberately does NOT do:

- NEVER injects methods.
- NEVER mutates user methods.
- NEVER runs at instance creation — only at class creation.
- NEVER produces code at runtime.

**Inheritance**: ``FlowMeta`` walks ``cls.__dict__`` only — inherited
decorated methods are NOT registered in the child's registry. This is
deliberate: inheritance of step methods adds subtle ordering and
override semantics, undermining the framework's contract that step
registration is explicit at class-definition time. If composition is
needed, build a ``FlowExecutable`` adapter and nest a Flow inside a
Graph.

**State**: ``self.state`` is the developer's mutable typed object. The
developer constructs the state instance and passes it via
``initial_state=`` — either the state class (auto-instantiated with
zero args) or a pre-built instance. The framework NEVER auto-instantiates
state from the generic type parameter — that is one of the seven hidden
behaviors of CrewAI Flow we explicitly forbid.

**Parallel mutation contract**: When multiple listeners fire in
parallel (e.g., via ``a & b`` gates or multiple ``@flow_start`` methods),
they share the same ``self.state`` reference. The framework adds NO
hidden lock — concurrent writes are the developer's responsibility.
Either write to disjoint fields, or wrap mutations in an explicit
``asyncio.Lock``. This mirrors the ``TaskGroup`` hook concurrency
contract.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast, override

from troopai.adk.flows.exceptions import FlowDefinitionError
from troopai.adk.flows.flow_wrappers import FlowRole, FlowStep
from troopai.adk.flows.registry import FlowStepRegistry, TriggerSpec

if TYPE_CHECKING:
    from troopai.adk.flows.deferred import FlowApprovalDecision
    from troopai.adk.flows.definition import FlowDefinition
    from troopai.adk.run.context import RunContext
    from troopai.adk.visualization.dot import DotRankdir
    from troopai.adk.visualization.mermaid import MermaidDirection


class FlowMeta(type):
    """Metaclass that collects Flow decorator registrations at class creation.

    Runs once per class definition, walks ``cls.__dict__`` (NOT inherited
    members), validates signatures, and builds a frozen
    :class:`FlowStepRegistry` stamped on the class as ``__flow_registry__``.

    The metaclass distinguishes the abstract :class:`Flow` base from
    concrete subclasses by checking whether any base class already
    carries a ``__flow_registry__`` attribute. The abstract base sets
    ``__flow_registry__ = None`` at class-creation time. Concrete
    subclasses (those whose bases include the base ``Flow``) must
    declare at least one ``@flow_start`` — enforced by
    :class:`FlowStepRegistry.__post_init__`.
    """

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Collect Flow registrations from ``namespace`` at class creation.

        Args:
            name: Class name being created.
            bases: Tuple of base classes.
            namespace: The class body's local namespace (``cls.__dict__``).
            **kwargs: Extra metaclass keyword arguments (currently unused).

        Raises:
            FlowDefinitionError: When any decorated method has an invalid
                signature, when a decorated method is a classmethod /
                staticmethod, or when the registry validation fails
                (e.g., no ``@flow_start``).
        """
        super().__init__(name, bases, namespace, **kwargs)
        if not _has_flow_ancestor(bases):
            cls.__flow_registry__ = None
            return
        starts, listeners, routers = _collect_decorations(namespace)
        cls.__flow_registry__ = FlowStepRegistry(
            starts=frozenset(starts),
            listeners=listeners,
            routers=routers,
        )


def _has_flow_ancestor(bases: tuple[type, ...]) -> bool:
    """Return True iff any base class already carries ``__flow_registry__``.

    The abstract :class:`Flow` base sets ``__flow_registry__ = None`` at
    its own class-creation time. Any class inheriting from :class:`Flow`
    (directly or transitively) will therefore have the attribute via
    inheritance — that is the marker for "concrete subclass."

    Args:
        bases: Tuple of base classes being inspected.

    Returns:
        ``True`` if at least one base carries the marker;
        ``False`` for the abstract ``Flow`` class itself.
    """
    return any(hasattr(base, "__flow_registry__") for base in bases)


def _collect_decorations(
    namespace: dict[str, Any],
) -> tuple[
    set[str],
    dict[str, tuple[TriggerSpec, ...]],
    dict[str, tuple[TriggerSpec, ...]],
]:
    """Walk ``namespace`` and collect ``@flow_start`` / ``@flow_listen`` / ``@flow_router`` methods.

    Validates every decorated method's signature and rejects
    ``classmethod`` / ``staticmethod`` descriptors that wrap a decorated
    function (the decorator order ``@classmethod @flow_start`` is a common
    mistake that would produce surprising runtime errors otherwise).

    Args:
        namespace: Class body namespace from :meth:`FlowMeta.__init__`.

    Returns:
        Triple ``(starts, listeners, routers)`` ready to construct a
        :class:`FlowStepRegistry`.

    Raises:
        FlowDefinitionError: On any signature problem or decorator misuse.
    """
    starts: set[str] = set()
    listeners: dict[str, tuple[TriggerSpec, ...]] = {}
    routers: dict[str, tuple[TriggerSpec, ...]] = {}
    for attr_name, obj in namespace.items():
        if isinstance(obj, (classmethod, staticmethod)):
            if isinstance(obj.__func__, FlowStep):
                raise FlowDefinitionError(
                    f"Flow step {attr_name!r} must be a plain async method; "
                    f"got {type(obj).__name__}. Remove @classmethod / @staticmethod.",
                )
            continue
        if not isinstance(obj, FlowStep):
            continue
        _validate_step_signature(obj, attr_name)
        triggers: tuple[TriggerSpec, ...] = obj.__flow_triggers__
        role: FlowRole = obj.__flow_role__
        if role == "start":
            starts.add(attr_name)
        elif role == "listen":
            listeners[attr_name] = triggers
        else:
            # role == "router" — exhaustive on the Literal["start","listen","router"] union.
            routers[attr_name] = triggers
    return starts, listeners, routers


def _validate_step_signature(step: FlowStep, name: str) -> None:
    """Validate that the wrapped method is ``async def`` and takes only ``self``.

    Strict enforcement at class-definition time makes the explicit-only
    step contract structural: a developer who tries to accept the
    previous step's return value via a second parameter gets a clear
    error at class-load, not a confusing runtime failure.

    Args:
        step: The :class:`FlowStep` wrapping the decorated method.
        name: The attribute name (for error messages).

    Raises:
        FlowDefinitionError: On any signature violation — non-async,
            missing self, or extra parameters.
    """
    fn = step.wrapped_function
    if not inspect.iscoroutinefunction(fn):
        raise FlowDefinitionError(
            f"Flow step {name!r} must be 'async def'; got a synchronous function. Step bodies run under asyncio.",
        )
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) == 0 or params[0].name != "self":
        raise FlowDefinitionError(
            f"Flow step {name!r} must take 'self' as its first parameter.",
        )
    if len(params) > 1:
        extras = [p.name for p in params[1:]]
        raise FlowDefinitionError(
            f"Flow step {name!r} must take ONLY 'self' "
            f"(no auto-injected args). Got extras: {extras}. "
            f"This rule prevents CrewAI's hidden behavior of injecting "
            f"the previous step's return value into the listener.",
        )


def collect_step_descriptions(namespace: Mapping[str, Any]) -> dict[str, str | None]:
    """Extract step description strings from a class namespace.

    Walks ``namespace`` and reads ``__flow_description__`` from every
    :class:`FlowStep` found. Used by :meth:`Flow.get_definition` to
    propagate ``description=`` decorator kwarg values into the pure-data
    :class:`~troopai.adk.flows.definition.FlowDefinition` without holding
    callable references.

    Args:
        namespace: A class body namespace (``cls.__dict__`` or the
            namespace passed to :meth:`FlowMeta.__init__``).

    Returns:
        Mapping from step method name to its description string or
        ``None`` when the decorator was called without ``description=``.
    """
    result: dict[str, str | None] = {}
    for attr_name, obj in namespace.items():
        if isinstance(obj, FlowStep):
            result[attr_name] = obj.__flow_description__
    return result


class Flow[StateT](metaclass=FlowMeta):
    """Base class for declarative multi-step orchestration over typed state.

    Subclasses decorate methods with ``@flow_start`` / ``@flow_listen``
    / ``@flow_router`` and access shared state via ``self.state``.
    The :class:`FlowMeta` metaclass collects the decorations at class
    creation; the executor runs them at :meth:`Runner.arun_flow` time.

    State initialization is mandatory and explicit. Two construction
    paths, both developer-declared: pass ``initial_state=`` (the state
    class, auto-instantiated with zero args, or a pre-built instance),
    or declare a :attr:`state_factory` class attribute on the subclass.
    An explicit ``initial_state`` always wins when both are present.
    The framework NEVER auto-instantiates state from the
    ``Flow[StateT]`` generic parameter — that is one of CrewAI's hidden
    behaviors we explicitly forbid.

    Visualisation: :meth:`to_mermaid` / :meth:`to_dot` emit the flow
    topology as a Mermaid or Graphviz DOT string for embedding in
    pull-request descriptions, documentation, or live dashboards.
    Both methods walk the immutable :class:`FlowStepRegistry`
    transition table — pure functions of class metadata, no run
    required.

    Example::

        from pydantic import BaseModel
        from troopai.adk.flows import Flow, flow_start, flow_listen, flow_router


        class ResearchState(BaseModel):
            topic: str = ""
            research: str = ""


        class ResearchFlow(Flow[ResearchState]):
            @flow_start
            async def kickoff(self) -> None:
                self.state.topic = "climate"

            @flow_listen(kickoff)
            async def research(self) -> None:
                # ... build prompt from self.state, call Runner.arun
                ...


        # Three equivalent constructions:
        flow = ResearchFlow(ResearchState)  # class — zero-arg instantiation
        flow = ResearchFlow(ResearchState(topic="ml"))  # instance — used directly


        class FactoryFlow(Flow[ResearchState]):
            state_factory = ResearchState  # class attribute — zero-arg construction

            @flow_start
            async def kickoff(self) -> None: ...


        flow = FactoryFlow()  # state_factory() builds a fresh ResearchState()
        result = await Runner.arun_flow(flow)

    Args:
        initial_state: The state class (auto-instantiated with zero
            args) OR a pre-built instance. Mandatory unless the
            subclass declares :attr:`state_factory`; an explicit value
            always wins over the factory.

    Raises:
        FlowDefinitionError: When ``initial_state`` is ``None`` and the
            subclass declares no usable :attr:`state_factory`.
    """

    __flow_registry__: ClassVar[FlowStepRegistry | None]
    """Registry stamped by :class:`FlowMeta` at class creation.

    ``None`` on the abstract :class:`Flow` base class itself;
    a frozen :class:`FlowStepRegistry` on every concrete subclass.
    Read by the executor at run time.
    """

    state_factory: ClassVar[Callable[[], Any] | None] = None
    """Optional zero-arg factory used to build ``self.state``.

    The second explicit state path: declare ``state_factory = MyState``
    on the subclass and construct with zero arguments — each instance
    calls the factory for a fresh state object. An explicit
    ``initial_state=`` argument always wins when both are present.
    ``None`` (default) means the subclass relies on ``initial_state=``.
    The framework still NEVER infers state from the ``Flow[StateT]``
    generic parameter — both paths are developer-declared.

    Declared as ``Callable[[], Any]`` rather than ``Callable[[], StateT]``:
    PEP 526 forbids type variables inside ``ClassVar`` (pyright enforces
    this), so the ``StateT`` return contract is documented here instead
    of encoded in the annotation.
    """

    state: StateT
    """The flow's typed state, accessible from every step method."""

    flow_id: str
    """Stable identifier for this flow instance (``flow-<8-hex>``)."""

    run_context: RunContext[Any] | None = None
    """Populated by :class:`FlowExecutor` for the duration of a run.

    Step bodies that want their inner :meth:`Runner.arun` calls to share
    a cumulative-usage accumulator pass ``context=self.run_context`` to
    those calls. This is OPT-IN — the framework never auto-injects the
    context into any call. ``None`` outside of a run.
    """

    _pending_approvals: dict[str, FlowApprovalDecision]
    """Resume-time approval decisions keyed by deferred step name.

    Populated by :meth:`Runner.arun_flow_from_checkpoint` before the
    executor starts; read by :class:`FlowExecutor` via
    :meth:`get_pending_approval`. Empty for cold-start runs.
    """

    _pending_agent_resolutions: dict[str, str]
    """Resume-time agent-bridge resolutions keyed by ``defer_key``.

    Maps a deferred ``defer_key`` to the serialised :class:`RunState`
    JSON the consumer recorded decisions onto (via
    :meth:`RunState.approve` / :meth:`RunState.reject`). Read by
    :func:`arun_flow_agent` on resume so the inner agent run picks
    up where it deferred. Empty for cold-start runs.
    """

    def __init__(self, initial_state: StateT | type[StateT] | None = None) -> None:
        """Initialize the flow with explicit state.

        Args:
            initial_state: A state instance OR the state class itself.
                When a class is passed, it is instantiated with zero
                arguments (``initial_state()``); when an instance is
                passed, it is used directly. Mirrors CrewAI's
                ``Flow.initial_state`` field which accepts the same
                dual shape. When omitted, the subclass's
                :attr:`state_factory` class attribute is used instead;
                an explicit ``initial_state`` always wins over the
                factory. The framework NEVER infers state from the
                ``Flow[StateT]`` generic parameter.

        Raises:
            FlowDefinitionError: When ``initial_state`` is ``None`` and
                no :attr:`state_factory` is declared, or when
                :attr:`state_factory` is not callable.
        """
        if initial_state is None:
            # getattr indirection: pyright method-binds a callable-typed
            # ClassVar accessed as ``type(self).state_factory`` (treats
            # it as an unbound method missing ``self``), so the lookup
            # goes through getattr into an explicitly annotated local.
            factory: Callable[[], Any] | None = getattr(type(self), "state_factory", None)
            if factory is not None and not callable(factory):
                raise FlowDefinitionError(
                    f"{type(self).__name__}.state_factory must be a zero-arg callable "
                    f"returning a state instance; got {factory!r}.",
                )
            if factory is None:
                raise FlowDefinitionError(
                    f"{type(self).__name__}() requires initial_state — pass either "
                    f"the state class (auto-instantiated) or an instance, or declare "
                    f"a state_factory class attribute on the subclass. "
                    f"The framework NEVER infers state from the generic parameter.",
                )
            # Same zero-arg call contract as the initial_state=MyState
            # form; the factory's return type is the subclass's
            # responsibility (declared Callable[[], Any] — see the
            # state_factory attribute docstring).
            self.state = factory()
        elif isinstance(initial_state, type):
            # type[StateT]() returns StateT at runtime; pyright widens
            # the call to "Any | object" because the metaclass result
            # is structurally unknown. cast is the narrow + documented
            # escape — the runtime invariant pyright can't see.
            self.state = cast("StateT", initial_state())
        else:
            self.state = initial_state
        self.flow_id = f"flow-{uuid.uuid4().hex[:8]}"
        self._pending_approvals = {}
        self._pending_agent_resolutions = {}

    @override
    def __repr__(self) -> str:
        """Compact, human-readable repr — never dumps ``self.state``.

        The default object repr hides which flow this is; a state dump
        would leak arbitrary payloads into logs. Shows the flow id, the
        registered step / router counts, and the state's type name.
        """
        parts: list[str] = []
        parts.append(f"flow_id={self.flow_id!r}")
        registry = type(self).__flow_registry__
        if registry is not None:
            parts.append(f"steps={len(registry.step_names())}")
            parts.append(f"routers={len(registry.routers)}")
        parts.append(f"state={type(self.state).__name__}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def set_pending_approvals(self, approvals: dict[str, FlowApprovalDecision]) -> None:
        """Public setter used by :class:`Runner` on the checkpoint-resume path.

        Replaces the entire pending-approvals mapping. Called once
        before the executor starts. Routed through this accessor
        rather than direct attribute write so external modules never
        touch ``_pending_approvals`` directly.

        Args:
            approvals: Mapping from deferred step name to the
                corresponding :class:`FlowApprovalDecision`. Empty
                mapping clears the table.
        """
        self._pending_approvals = dict(approvals)

    def get_pending_approval(self, step_name: str) -> FlowApprovalDecision | None:
        """Return the resume-time approval decision for ``step_name``, if any.

        Non-consuming read — useful for inspection in hooks / tests.
        The executor uses :meth:`consume_pending_approval` instead so
        a cyclic re-fire of the same step does NOT auto-approve.

        Args:
            step_name: Method name of the candidate step.

        Returns:
            The pending :class:`FlowApprovalDecision`, or ``None`` if
            no decision was pre-queued for this step.
        """
        return self._pending_approvals.get(step_name)

    def set_pending_agent_resolutions(self, resolutions: dict[str, str]) -> None:
        """Public setter used by :class:`Runner` on the agent-bridge resume path.

        Replaces the entire pending-agent-resolutions mapping with a
        copy of ``resolutions`` (``defer_key`` → serialised
        :class:`RunState` JSON). Called once before the executor
        starts. Routed through this accessor rather than direct
        attribute write so external modules never touch
        ``_pending_agent_resolutions`` directly.

        Args:
            resolutions: Mapping from ``defer_key`` to the JSON
                payload the consumer recorded decisions onto. Empty
                mapping clears the table.
        """
        self._pending_agent_resolutions = dict(resolutions)

    def consume_pending_agent_resolution(self, defer_key: str) -> str | None:
        """Pop the resume-time :class:`RunState` JSON for ``defer_key``, if any.

        Used by :func:`arun_flow_agent` so each pre-queued resolution
        applies to exactly one inner agent invocation. Mirrors the
        consume-on-use pattern of :meth:`consume_pending_approval`.

        Args:
            defer_key: The agent-bridge key passed to
                :func:`arun_flow_agent`.

        Returns:
            The popped JSON payload (suitable for
            :meth:`RunState.from_dict` ``+`` ``json.loads``), or
            ``None`` if no resolution was pre-queued.
        """
        return self._pending_agent_resolutions.pop(defer_key, None)

    def pending_agent_resolution_keys(self) -> list[str]:
        """Return sorted list of unconsumed agent-resolution ``defer_key`` values.

        Public accessor so :class:`FlowExecutor` can inspect leftover
        resolutions at run completion without reaching into the private
        ``_pending_agent_resolutions`` attribute.

        Returns:
            Sorted list of ``defer_key`` strings that were supplied via
            :meth:`set_pending_agent_resolutions` but never consumed by
            :func:`arun_flow_agent`.
        """
        return sorted(self._pending_agent_resolutions.keys())

    def consume_pending_approval(self, step_name: str) -> FlowApprovalDecision | None:
        """Pop the resume-time approval decision for ``step_name``, if any.

        Used by :class:`FlowExecutor` so each pre-queued decision
        applies to exactly one step invocation. Mirrors the tool-layer
        contract on :attr:`RunState.approved_tools` / ``rejected_tools``
        — they are consumed on use; a one-time approval never
        elevates into a persistent bypass.

        Args:
            step_name: Method name of the candidate step.

        Returns:
            The popped :class:`FlowApprovalDecision`, or ``None`` if
            no decision was pre-queued for this step.
        """
        return self._pending_approvals.pop(step_name, None)

    def get_registry(self) -> FlowStepRegistry:
        """Return the frozen :class:`FlowStepRegistry` for this Flow's class.

        Public accessor for the registry stamped by :class:`FlowMeta`.
        Used by the executor at run start to build the transition table.

        Returns:
            The class-level :class:`FlowStepRegistry`.

        Raises:
            FlowDefinitionError: When called on the abstract :class:`Flow`
                base class (no registry was built). Concrete subclasses
                always have a registry stamped by the metaclass.
        """
        registry = type(self).__flow_registry__
        if registry is None:
            raise FlowDefinitionError(
                f"{type(self).__name__} is abstract — no @flow_start/@flow_listen/@flow_router "
                f"declarations were found. Subclass Flow and decorate methods.",
            )
        return registry

    def get_definition(self) -> FlowDefinition:
        """Return a frozen :class:`FlowDefinition` for this Flow's class.

        Extracts the step registry stamped by :class:`FlowMeta` and
        compiles it into a pure-data :class:`FlowDefinition` with no
        callable references. The definition includes per-step description
        strings sourced from the ``description=`` decorator kwarg on each
        :class:`FlowStep` in the class body.

        Calling this method multiple times always returns structurally
        equivalent objects (same field values). The method does NOT cache
        the definition — the cost is one dict walk and is negligible.

        Returns:
            A frozen :class:`FlowDefinition` capturing steps, roles,
            triggers, gate topology, and router trigger mappings.

        Raises:
            FlowDefinitionError: When called on the abstract
                :class:`Flow` base class (no registry was built).
        """
        from troopai.adk.flows.definition import build_flow_definition

        registry = self.get_registry()
        descs = collect_step_descriptions(type(self).__dict__)
        return build_flow_definition(registry, descriptions=descs)

    def to_mermaid(self, *, direction: MermaidDirection = "LR") -> str:
        """Return a Mermaid ``flowchart`` rendering of this Flow's topology.

        Thin ergonomic wrapper around
        :func:`troopai.adk.visualization.flow_to_mermaid`. Pure function:
        no I/O, idempotent. Walks the immutable class registry — no
        run is triggered.

        Args:
            direction: Mermaid layout direction. ``"LR"`` (default)
                reads left-to-right; ``"TD"`` / ``"TB"`` reads top-down.
                Any value in the :data:`MermaidDirection` Literal is
                valid; invalid values raise ``ValueError`` at runtime.

        Returns:
            A complete Mermaid ``flowchart`` block ready to paste into
            GitHub Markdown or any Mermaid renderer.
        """
        from troopai.adk.visualization.mermaid import flow_to_mermaid

        return flow_to_mermaid(self, direction=direction)

    def to_dot(self, *, rankdir: DotRankdir = "LR") -> str:
        """Return a Graphviz DOT rendering of this Flow's topology.

        Thin ergonomic wrapper around
        :func:`troopai.adk.visualization.flow_to_dot`. Pure function:
        no I/O. Walks the immutable class registry — no run is
        triggered.

        Args:
            rankdir: Graphviz layout direction. ``"LR"`` (default)
                reads left-to-right; ``"TB"`` reads top-down. Invalid
                values raise ``ValueError`` at runtime.

        Returns:
            A complete DOT digraph string ready for the ``dot`` CLI or
            any DOT-aware renderer.
        """
        from troopai.adk.visualization.dot import flow_to_dot

        return flow_to_dot(self, rankdir=rankdir)
