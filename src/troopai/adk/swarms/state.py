"""SwarmState — serializable, resumable per-run state.

Tracks everything the swarm driver needs between turns: which agent
is active, the full shared history (for observability + strategies
that broadcast), per-agent scratch (for the ``SCOPED`` default),
cumulative usage, and the last yield signal.

Serialization follows the same pattern as ``RunState``:

- ``to_dict()`` / ``from_dict()`` round-trip the runtime-mutable
  fields; ``from_dict`` validates ``current_agent_name`` and every
  ``per_agent_scratch`` key against the supplied ``Swarm`` roster
  and raises ``ValueError`` on a non-member.
- ``to_json()`` is ``json.dumps(self.to_dict())``; ``from_json()``
  is ``from_dict(json.loads(raw))``. No version key, no envelope:
  an older persisted payload either round-trips or fails loudly on
  a hard key access — never a half-populated object.

The ``Swarm`` config is NOT serialized on the state — ``from_json``
requires the caller to provide the same ``Swarm`` instance (or an
equivalent) because a ``Swarm`` contains ``Agent`` objects,
``SwarmPolicy`` subclasses, and ``TerminationCondition`` subclasses
which may not be JSON-representable in general. This matches the
OpenAI agents SDK approach to ``RunState``: the structure of
agents/policies/conditions is code; the run state is data.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, TypeVar

from troopai.adk.graphs.interrupt import NESTED_AGENT_TOOL_APPROVAL_KIND, Interrupt, NestedAgentInterrupt
from troopai.adk.run.state import RunState
from troopai.adk.swarms.yield_signal import SwarmDone, SwarmHandoff, SwarmYieldSignal
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem


class _CumulativeUsageDict(TypedDict):
    """Serialized shape of :class:`LLMUsage` inside :class:`SwarmStateDict`.

    Kept TypedDict rather than re-deriving from ``LLMUsage`` so the
    swarm-state schema is self-contained — bumping the usage dataclass
    cannot silently migrate state payloads.

    Attributes:
        requests: Total number of LLM API requests made during the run.
        total_tokens: Sum of input and output tokens across all requests.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens generated.
    """

    requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int


class _HandoffYieldDict(TypedDict):
    """Serialized shape of a :class:`SwarmHandoff` yield.

    Attributes:
        kind: Discriminator constant — always ``"handoff"``.
        target: Name of the target member agent.
        message: Explicit handoff content passed to the target.
    """

    kind: Literal["handoff"]
    target: str
    message: str


class _DoneYieldDict(TypedDict):
    """Serialized shape of a :class:`SwarmDone` yield.

    ``final_output`` is intentionally ``Any | None`` — the structured
    output is whatever the member agent returned (Pydantic model,
    primitive, nested dict, …). That shape lives on the agent's
    ``output_schema``, not on the swarm-state schema.

    Attributes:
        kind: Discriminator constant — always ``"done"``.
        reason: Human-readable termination reason supplied by the
            LLM via the ``swarm_done`` tool.
        final_output: The terminal agent's last output. Type depends on
            whether the agent uses a structured ``output_schema``.
    """

    kind: Literal["done"]
    reason: str
    final_output: Any


_LastYieldDict = _HandoffYieldDict | _DoneYieldDict


class SwarmStateDict(TypedDict):
    """Serialized shape of :class:`SwarmState`.

    Produced by :meth:`SwarmState.to_dict` and consumed by
    :meth:`SwarmState.from_dict`. The bare serialized dict — no
    version key, no envelope.

    ``last_yield`` may be ``None`` before the first turn has yielded.

    Attributes:
        current_agent_name: Name of the agent whose turn is next (or
            was last active). Resolved against ``Swarm.members`` on
            load.
        shared_history: Full audit trail of every Layer 1 item produced
            by every member during the run, in chronological order.
        per_agent_scratch: Per-agent private history, keyed by agent
            name. Each list contains the Layer 1 items from that
            agent's turns.
        handoff_count: Number of agent switches that have occurred.
        total_turns: 1-indexed count of member turns completed or in
            progress.
        cumulative_usage: Aggregate LLM token usage across all member
            turns, serialized as a :class:`_CumulativeUsageDict`.
        last_yield: The most recent explicit yield signal (handoff or
            done), or ``None`` before the first yield.
        pending_interrupts: Serialized interrupt payloads awaiting a
            human reply, keyed by member name.
        nested_agent_snapshots: Serialized mid-execution sub-agent
            state for members paused on a nested-agent interrupt, keyed
            by member name.
        status: Lifecycle tag: ``"running"`` / ``"completed"`` /
            ``"failed"`` / ``"interrupted"``.
        error: Failure message when ``status == "failed"``; ``None``
            otherwise.
        swarm_id: Stable run identifier for tracing correlation; absent
            (not required) on older persisted payloads.
        resume_counts: Per-member counter of resume entries; absent on
            fresh runs.
    """

    current_agent_name: str
    shared_history: list[LLMInputContentItem]
    per_agent_scratch: dict[str, list[LLMInputContentItem]]
    handoff_count: int
    total_turns: int
    cumulative_usage: _CumulativeUsageDict
    last_yield: _LastYieldDict | None
    pending_interrupts: dict[str, dict[str, Any]]
    nested_agent_snapshots: dict[str, dict[str, Any]]
    status: NotRequired[str]
    """Lifecycle tag: ``"running"`` / ``"completed"`` / ``"failed"`` /
    ``"interrupted"``.  Absent on payloads persisted before this field was
    added; loaders default to ``"running"`` in that case."""
    error: str | None
    swarm_id: NotRequired[str | None]
    resume_counts: NotRequired[dict[str, int]]
    initial_input_items: NotRequired[list[LLMInputContentItem]]
    """The run's opening user prompt as Layer 1 params. Absent on a payload
    persisted before this field existed; loaders default to an empty list."""


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


@dataclass
class SwarmState[TContext]:
    """Per-run mutable state of a swarm execution.

    Held by the driver loop across turn boundaries. Passed (read-only
    in practice) to ``SwarmPolicy.select_next``,
    ``TerminationCondition.should_stop``, and ``SwarmHooks`` so those
    collaborators can inspect progress without touching the driver.

    Serialization note: ``swarm`` and ``current_agent`` are NOT
    serialized. ``to_dict()`` emits only the runtime-mutable fields;
    ``from_dict()`` requires the caller to re-supply the ``Swarm``
    config and will resolve ``current_agent`` by name.

    Attributes:
        swarm: The :class:`~troopai.adk.swarms.swarm.Swarm` config this
            state belongs to. Not serialized.
        current_agent: The agent whose turn is next (or in progress).
            Not serialized directly — resolved from
            ``current_agent_name`` on load.
        current_agent_name: Serializable mirror of
            ``current_agent.name``. Updated whenever ``current_agent``
            changes.
        shared_history: Full Layer 3 audit trail of every item produced
            by every member during the swarm run.
        per_agent_scratch: Per-agent private history, keyed by agent
            name. Used by the ``SCOPED`` shared-context strategy.
        handoff_count: Number of agent switches that have occurred.
        total_turns: 1-indexed count of member turns completed (or in
            progress).
        cumulative_usage: Cumulative LLM usage across all member turns
            in this swarm run.
        last_yield: The most recent explicit yield signal emitted by a
            member turn. ``None`` before the first yield.
        pending_interrupts: Interrupts awaiting a human reply, keyed by
            swarm member name.
        nested_agent_snapshots: Mid-execution sub-agent state for
            members paused on a :class:`NestedAgentInterrupt`, keyed
            by member name.
        status: Lifecycle tag: ``"running"`` / ``"completed"`` /
            ``"failed"`` / ``"interrupted"``.
        error: Failure message when ``status == "failed"``. ``None``
            otherwise.
        swarm_id: Stable id for this swarm run, used for tracing
            correlation. ``None`` when absent from a loaded checkpoint.
        resume_counts: Per-member counter of resume entries. Empty dict
            for fresh runs.
        initial_input_items: The run's opening user prompt as Layer 3 items,
            recorded once on the first turn so cross-agent broadcast
            strategies keep the original question visible.
        last_structured_output: Parsed structured output from the most
            recent member turn. Not serialized.
    """

    swarm: Swarm[TContext]
    """The ``Swarm`` config this state belongs to. Not serialized."""

    current_agent: Agent[TContext]
    """The agent whose turn is next (or in progress). Not serialized
    directly — resolved from ``current_agent_name`` on load."""

    current_agent_name: str
    """Serializable mirror of ``current_agent.name``. Updated whenever
    ``current_agent`` changes. ``from_dict`` uses this to re-resolve
    the ``Agent`` instance from ``Swarm.members``."""

    shared_history: list[RunItem] = field(default_factory=list)
    """Full Layer 3 audit trail of every item produced by every
    member during the swarm run. Used by
    ``SharedContextStrategy.FULL_BROADCAST`` /``LAST_N`` /
    ``SUMMARIZED`` to build each next turn's input. ``SCOPED``
    ignores this and reads from ``per_agent_scratch``."""

    per_agent_scratch: dict[str, list[RunItem]] = field(default_factory=dict)
    """Per-agent private history, keyed by agent name. Used by the
    ``SCOPED`` shared-context strategy (default). Each agent's
    scratch accumulates items from its own turns plus any items
    explicitly handed to it via ``SwarmHandoff.message``."""

    handoff_count: int = 0
    """Number of agent switches that have occurred. Incremented each
    time the driver resolves a ``SwarmHandoff``. Compared against
    ``SwarmConfig.max_handoffs`` hard guard."""

    total_turns: int = 0
    """1-indexed count of member turns completed (or in progress).
    Incremented at the top of each turn. Distinct from
    ``RunConfig.max_total_turns`` which counts LLM calls, not member
    turns — they typically differ when a member uses tools."""

    cumulative_usage: LLMUsage = field(default_factory=LLMUsage)
    """Cumulative LLM usage across all member turns in this swarm run.
    Updated at the end of each turn by adding the per-turn
    ``RunResult.context.usage`` delta. Compared against
    ``SwarmConfig.max_total_tokens``."""

    last_yield: SwarmYieldSignal | None = None
    """The most recent explicit yield signal emitted by a member turn.
    ``None`` before the first yield. Read by
    ``TerminationCondition`` implementations (e.g.
    ``ExplicitDoneTermination`` fires when this is a ``SwarmDone``)."""

    pending_interrupts: dict[str, Interrupt] = field(default_factory=dict)
    """Interrupts awaiting a human reply, keyed by swarm member name.

    Populated by the swarm loop when a member's turn raises
    :class:`~troopai.adk.graphs.interrupt.InterruptException` (HITL via
    ``request_human_input``) or when a member's tool defers
    (:class:`AgentToolDeferral` lifted to
    :class:`NestedAgentInterrupt`). Cleared by the resume path once
    the caller supplies a :class:`SwarmResume`."""

    nested_agent_snapshots: dict[str, RunState] = field(default_factory=dict)
    """Mid-execution sub-agent state for members paused on a
    :class:`NestedAgentInterrupt`. Serialised via :meth:`RunState.to_dict`;
    rehydrated via :meth:`RunState.from_dict` on load. The cross-reference
    invariant in :meth:`from_dict` requires every NestedAgentInterrupt
    entry in :attr:`pending_interrupts` to have a matching snapshot here."""

    status: str = "running"
    """Lifecycle tag: ``"running"`` / ``"completed"`` / ``"failed"`` /
    ``"interrupted"``. Updated by the swarm loop."""

    error: str | None = None
    """Failure message when ``status == "failed"``. Not an Exception
    instance so the state remains JSON-safe."""

    swarm_id: str | None = None
    """Stable id for this swarm run.

    Generated at ``Runner.arun_swarm`` entry as a UUID and persisted so
    ``Runner.arun_swarm_from_checkpoint`` can reuse it — the same
    ``troopai.swarm.id`` attribute appears on suspend-side and resume-side
    spans, letting tracing dashboards correlate the full lifecycle.
    ``None`` when absent from a loaded checkpoint; the runner regenerates
    a fresh UUID in that case with a warning log."""

    resume_counts: dict[str, int] = field(default_factory=dict)
    """Per-member counter of resume entries.

    Bumped at the entry of the deep-resume helpers in
    ``run/swarm_resume.py``; read by the per-turn span to set
    ``troopai.swarm.turn.resume_attempt``. Empty dict for fresh runs."""

    initial_input_items: list[RunItem] = field(default_factory=list)
    """The run's opening user prompt, recorded as Layer 3 items on turn 1.

    ``shared_history`` accumulates only the items each member *produces*, so
    the original question is never in it — leaving every cross-agent strategy
    that reads ``shared_history`` (``FULL_BROADCAST`` / ``LAST_N`` /
    ``SUMMARIZED``) blind to what the user actually asked on turn 2 onward.
    The driver records the prompt here once at the start of the first turn and
    the broadcast strategies prepend it, so the question stays visible without
    duplicating it into ``shared_history`` (which would double it into
    ``SwarmRunResult.new_items`` and the session store). Serialized so a
    resumed broadcast run keeps the question. ``SCOPED`` ignores this."""

    last_structured_output: Any | None = None
    """Parsed structured output from the most recent member turn.

    Populated by the swarm driver after each turn when the active agent
    has an ``output_schema`` — the value is the parsed Pydantic model
    (commonly an :class:`~troopai.adk.types.intents.Intent` subclass).
    Read by
    :class:`~troopai.adk.swarms.policy.StructuredRoutingPolicy` to
    dispatch via the wrapped ``HandoffRoute``. ``None`` when the
    active agent does not have a structured-output schema or when the
    last turn produced free-form text.

    Intentionally NOT serialized: this is a transient routing artifact
    that is recomputed on the next turn. A resumed swarm re-runs the
    active agent's turn before the next routing decision, so serializing
    stale structured output would bias the first routing decision after
    resume."""

    def advance_to(self, agent: Agent[TContext]) -> None:
        """Switch the active agent and mirror the name field.

        Keeps ``current_agent`` and ``current_agent_name`` in sync so
        the serialization path never drifts. Called by the driver on
        each turn transition.

        Args:
            agent: The agent that will take the next turn.
        """
        self.current_agent = agent
        self.current_agent_name = agent.name
        # Lazily seed the per-agent scratch so SCOPED strategy always
        # has a list to read from.
        if agent.name not in self.per_agent_scratch:
            self.per_agent_scratch[agent.name] = []

    def to_dict(self) -> SwarmStateDict:
        """Emit the serializable fields as a :class:`SwarmStateDict`.

        Intentionally omits ``swarm`` and ``current_agent`` (non-data
        references). ``last_yield`` is emitted as a tagged dict so
        ``from_dict`` can dispatch. ``RunItem`` instances are emitted
        via ``to_param()`` into Layer 1 TypedDicts for portability.

        Callers needing a JSON string for cross-process persistence
        should use ``to_json()``.

        Returns:
            A :class:`SwarmStateDict` containing all serializable
            fields of this state.

        Raises:
            TypeError: If ``last_yield`` is a :class:`SwarmYieldSignal`
                subtype not handled by this method.
        """
        from troopai.adk.types.items.items import ItemHelpers  # local to avoid cycle

        yield_payload: _LastYieldDict | None
        if self.last_yield is None:
            yield_payload = None
        elif isinstance(self.last_yield, SwarmHandoff):
            yield_payload = {
                "kind": "handoff",
                "target": self.last_yield.target,
                "message": self.last_yield.message,
            }
        elif isinstance(self.last_yield, SwarmDone):
            yield_payload = {
                "kind": "done",
                "reason": self.last_yield.reason,
                "final_output": _json_safe_final_output(self.last_yield.final_output),
            }
        else:
            raise TypeError(f"Unknown SwarmYieldSignal type: {type(self.last_yield).__name__}")

        return {
            "current_agent_name": self.current_agent_name,
            "shared_history": list(ItemHelpers.run_items_to_params(self.shared_history)),
            "per_agent_scratch": {
                name: list(ItemHelpers.run_items_to_params(items)) for name, items in self.per_agent_scratch.items()
            },
            "handoff_count": self.handoff_count,
            "total_turns": self.total_turns,
            "cumulative_usage": {
                "requests": self.cumulative_usage.requests,
                "total_tokens": self.cumulative_usage.total_tokens,
                "input_tokens": self.cumulative_usage.input_tokens,
                "output_tokens": self.cumulative_usage.output_tokens,
            },
            "last_yield": yield_payload,
            "pending_interrupts": {
                name: dataclasses.asdict(interrupt) for name, interrupt in self.pending_interrupts.items()
            },
            "nested_agent_snapshots": {name: snap.to_dict() for name, snap in self.nested_agent_snapshots.items()},
            "status": self.status,
            "error": self.error,
            "swarm_id": self.swarm_id,
            "resume_counts": dict(self.resume_counts),
            "initial_input_items": list(ItemHelpers.run_items_to_params(self.initial_input_items)),
        }

    @classmethod
    def from_dict(
        cls,
        data: SwarmStateDict,
        swarm: Swarm[TContext],
    ) -> SwarmState[TContext]:
        """Reconstruct a ``SwarmState`` from the output of ``to_dict``.

        Requires the caller to re-supply the ``Swarm`` config (which
        carries the ``Agent`` instances). The saved
        ``current_agent_name`` is resolved against ``swarm.members``.

        Args:
            data: The serialized state dict produced by
                :meth:`to_dict`.
            swarm: The :class:`~troopai.adk.swarms.swarm.Swarm` config
                this state belongs to. Provides the ``Agent`` instances
                needed to resolve ``current_agent_name``.

        Returns:
            A fully rehydrated :class:`SwarmState` ready for the
            driver to resume from.

        Raises:
            ValueError: If ``current_agent_name`` does not match any
                member in the supplied ``swarm``, or if
                ``per_agent_scratch`` contains a key that is not a
                member, or if ``status`` holds an unknown value, or if
                a ``NestedAgentInterrupt`` has no matching snapshot.
        """
        from troopai.adk.types.items.items import ItemHelpers  # local to avoid cycle

        current_name = data["current_agent_name"]
        current_agent: Agent[TContext] | None = None
        for m in swarm.members:
            if m.name == current_name:
                current_agent = m
                break
        if current_agent is None:
            raise ValueError(
                f"current_agent_name='{current_name}' does not match any "
                f"member in the supplied Swarm (members: "
                f"{[m.name for m in swarm.members]})."
            )

        # Reconstruct RunItems from Layer 1 params. Direct key access
        # throughout: ``SwarmStateDict`` is ``total=True``, so the
        # schema guarantees every field is present. Defensive
        # ``.get(key, default)`` reads here would silently paper over a
        # missing-field bug and defeat the TypedDict.
        shared_history_items = list(ItemHelpers.messages_to_run_items(data["shared_history"]))
        # Reject scratch keyed by a name that isn't in the roster.
        # A trusted round-trip never produces such keys; if they appear
        # we are deserializing an evolved or tampered payload and
        # silently importing phantom agent histories would mislead
        # policies (especially CustomPolicy) that read scratch by name.
        member_names = {m.name for m in swarm.members}
        per_agent_scratch: dict[str, list[RunItem]] = {}
        for name, raw_items in data["per_agent_scratch"].items():
            if name not in member_names:
                raise ValueError(
                    f"per_agent_scratch key {name!r} is not a member of "
                    f"the supplied Swarm (members: {sorted(member_names)}). "
                    "Refuse to import phantom agent scratch."
                )
            per_agent_scratch[name] = list(ItemHelpers.messages_to_run_items(raw_items))

        usage_data = data["cumulative_usage"]
        cumulative_usage = LLMUsage(
            requests=usage_data["requests"],
            total_tokens=usage_data["total_tokens"],
            input_tokens=usage_data["input_tokens"],
            output_tokens=usage_data["output_tokens"],
        )

        last_yield: SwarmYieldSignal | None = None
        y = data["last_yield"]
        if y is not None:
            # Narrow the tagged union via the ``kind`` discriminator.
            # Pyright's TypedDict narrowing drives the per-branch field
            # access below — using ``y["kind"]`` directly (not
            # ``y.get("kind")``) is what enables the narrowing.
            if y["kind"] == "handoff":
                last_yield = SwarmHandoff(target=y["target"], message=y["message"])
            elif y["kind"] == "done":
                last_yield = SwarmDone(reason=y["reason"], final_output=y["final_output"])
            else:
                raise ValueError(f"Unknown last_yield.kind={y['kind']!r} in SwarmState.")

        pending_interrupts = _rehydrate_swarm_pending_interrupts(data)
        nested_agent_snapshots = _rehydrate_swarm_nested_agent_snapshots(data)
        initial_input_items = list(ItemHelpers.messages_to_run_items(data.get("initial_input_items", [])))
        status_raw = data.get("status", "running")
        if status_raw not in _VALID_SWARM_STATUS_VALUES:
            raise ValueError(
                f"SwarmState.from_dict: status has unknown value {status_raw!r}. "
                f"Expected one of {sorted(_VALID_SWARM_STATUS_VALUES)}."
            )

        # Cross-reference: every NestedAgentInterrupt MUST have a matching
        # snapshot in nested_agent_snapshots, else the resume path would
        # deadlock. Mirror of GraphState.from_dict's invariant.
        for member_name, interrupt in pending_interrupts.items():
            if isinstance(interrupt, NestedAgentInterrupt) and member_name not in nested_agent_snapshots:
                raise ValueError(
                    f"SwarmState.from_dict: pending_interrupts[{member_name!r}] is a "
                    f"NestedAgentInterrupt but nested_agent_snapshots has no matching "
                    f"entry — checkpoint is inconsistent and would deadlock resume."
                )

        return cls(
            swarm=swarm,
            current_agent=current_agent,
            current_agent_name=current_name,
            shared_history=shared_history_items,
            per_agent_scratch=per_agent_scratch,
            handoff_count=data["handoff_count"],
            total_turns=data["total_turns"],
            cumulative_usage=cumulative_usage,
            last_yield=last_yield,
            pending_interrupts=pending_interrupts,
            nested_agent_snapshots=nested_agent_snapshots,
            status=status_raw,
            error=data.get("error"),
            swarm_id=data.get("swarm_id"),
            resume_counts=dict(data.get("resume_counts", {})),
            initial_input_items=initial_input_items,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string — ``json.dumps(self.to_dict())``.

        Use this for on-disk or cross-process persistence. For
        in-process handoff prefer ``to_dict()`` directly.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls,
        raw: str,
        swarm: Swarm[TContext],
    ) -> SwarmState[TContext]:
        """Deserialize from ``to_json()`` output — ``from_dict(json.loads(raw))``.

        Args:
            raw: A JSON string produced by :meth:`to_json`.
            swarm: The :class:`~troopai.adk.swarms.swarm.Swarm` config
                this state belongs to. Forwarded to :meth:`from_dict`
                for member-name resolution.

        Returns:
            A fully rehydrated :class:`SwarmState`.

        Raises:
            ValueError: If ``current_agent_name`` or any
                ``per_agent_scratch`` key is not a known member of the
                supplied ``swarm`` (semantic validation in
                ``from_dict``).
            json.JSONDecodeError: If ``raw`` is not valid JSON.
        """
        return cls.from_dict(json.loads(raw), swarm)


_VALID_SWARM_STATUS_VALUES: frozenset[str] = frozenset({"running", "completed", "failed", "interrupted"})
"""Allowlist of ``SwarmState.status`` values accepted on deserialisation.

Mirrors :data:`troopai.adk.graphs.state._VALID_STATUS_VALUES`. Prevents
arbitrary strings in a persisted payload from propagating into
``SwarmRunResult.status`` where consumer logic branches on it."""


def _json_safe_final_output(value: Any) -> Any:
    """Coerce a ``SwarmDone.final_output`` into a JSON-serialisable form.

    A terminal member with an ``output_schema`` yields a parsed Pydantic
    model as ``final_output``; the swarm loop stores that model verbatim on
    ``SwarmState.last_yield``. Emitting it raw would make ``to_dict`` produce
    a payload that ``json.dumps`` (used by ``to_json`` and every cross-process
    checkpointer) cannot serialise, aborting an otherwise-successful run at
    completion. Pydantic models are converted via ``model_dump(mode="json")``
    so the structured shape round-trips back through ``from_dict`` as a plain
    dict; any other already-JSON-safe value passes through unchanged, and the
    remaining non-serialisable cases fall back to ``str`` (mirroring the
    ``GraphState`` final-output convention).

    Args:
        value: The raw ``final_output`` carried on the ``SwarmDone`` signal.

    Returns:
        A JSON-serialisable representation of ``value``.
    """
    from pydantic import BaseModel  # local to avoid pulling pydantic into module import

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool, dict, list, type(None))):
        return value
    return str(value)


def _rehydrate_swarm_pending_interrupts(
    data: SwarmStateDict,
) -> dict[str, Interrupt]:
    """Rebuild SwarmState.pending_interrupts from serialised dicts.

    Reads the ``kind`` discriminator to select the concrete
    :class:`Interrupt` subclass — ``NestedAgentInterrupt`` for nested-
    agent deferrals, the base :class:`Interrupt` for plain HITL.
    Validates required NestedAgentInterrupt fields rather than
    surfacing opaque KeyErrors on partial payloads.

    Args:
        data: The raw :class:`SwarmStateDict` produced by
            :meth:`SwarmState.to_dict`.

    Returns:
        Mapping from member name to the rehydrated :class:`Interrupt`
        (or :class:`NestedAgentInterrupt` subtype).

    Raises:
        ValueError: When a ``NestedAgentInterrupt`` payload is missing
            ``agent_name`` or ``tool_call_ids``.
    """
    out: dict[str, Interrupt] = {}
    for name, payload in data.get("pending_interrupts", {}).items():
        node_id = payload.get("node_id", name)
        question = payload.get("question", "")
        kind = payload.get("kind", "generic")
        metadata = dict(payload.get("metadata") or {})
        if kind == NESTED_AGENT_TOOL_APPROVAL_KIND:
            agent_name = payload.get("agent_name")
            if not isinstance(agent_name, str) or len(agent_name) == 0:
                raise ValueError(
                    f"SwarmState.from_dict: pending_interrupts[{name!r}] is a "
                    f"NestedAgentInterrupt but agent_name is missing or empty."
                )
            tool_call_ids_raw = payload.get("tool_call_ids")
            if tool_call_ids_raw is None:
                raise ValueError(
                    f"SwarmState.from_dict: pending_interrupts[{name!r}] is a "
                    f"NestedAgentInterrupt but tool_call_ids is missing."
                )
            out[name] = NestedAgentInterrupt(
                node_id=node_id,
                question=question,
                kind=kind,
                metadata=metadata,
                agent_name=agent_name,
                tool_call_ids=tuple(tool_call_ids_raw),
            )
        else:
            out[name] = Interrupt(
                node_id=node_id,
                question=question,
                kind=kind,
                metadata=metadata,
            )
    return out


def _rehydrate_swarm_nested_agent_snapshots(
    data: SwarmStateDict,
) -> dict[str, RunState]:
    """Rebuild SwarmState.nested_agent_snapshots from serialised dicts.

    Args:
        data: The raw :class:`SwarmStateDict` produced by
            :meth:`SwarmState.to_dict`.

    Returns:
        Mapping from member name to the rehydrated :class:`RunState`.
    """
    return {name: RunState.from_dict(payload) for name, payload in data.get("nested_agent_snapshots", {}).items()}
