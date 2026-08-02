"""``GraphState`` — per-run mutable state of a graph execution.

Mirrors :class:`~troopai.adk.swarms.state.SwarmState` in shape and
serialisation patterns:

- ``to_dict()`` / ``from_dict()`` round-trip the runtime-mutable
  fields; ``from_dict`` requires a non-``None`` ``Graph`` and rejects
  an unknown node id against it with ``ValueError``.
- ``to_json()`` is ``json.dumps(self.to_dict())``; ``from_json()``
  is ``from_dict(json.loads(raw))``. No version key, no envelope:
  an older persisted payload either round-trips or fails loudly —
  never a half-populated object.

The :class:`Graph` config is NOT serialised on the state.
``from_json()`` requires the caller to supply the same ``Graph``
instance (or an equivalent with the same node ids) — the graph
encapsulates :class:`Executable`\\ s which are code, not data, and
will not round-trip through JSON. This matches how ``SwarmState``
handles its parent ``Swarm``.

``versions_seen`` is the LangGraph-inspired per-node, per-upstream
last-consumed version map. We track it at
``(node_id, upstream_node_id) -> superstep_number`` granularity so
that after a checkpoint restore the loop can decide whether a node
should re-fire. Without this, a restored cyclical graph can either
loop forever or never re-fire — LangGraph's
``_algo.py::prepare_next_tasks`` uses the same trick. ``produced_at``
records the superstep at which each node's current ``node_results``
entry was written; the loop compares the two maps after checkpoint
restore to decide which nodes must re-fire.

Per-node results live in ``node_results`` as a mapping
``node_id -> NodeResult``. Multiple firings of the same node (cycles)
overwrite the previous entry — the full history is recoverable from
``all_items``. This trades history granularity for readability; a
per-node ``firings: dict[str, list[NodeResult]]`` history map is not
currently tracked.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from troopai.adk.graphs.interrupt import (
    NESTED_AGENT_TOOL_APPROVAL_KIND,
    NESTED_GRAPH_INTERRUPT_KIND,
    Interrupt,
    NestedAgentInterrupt,
    NestedGraphInterrupt,
)
from troopai.adk.run.state import RunState
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.orchestration.executable import NodeResult
    from troopai.adk.types.items.items import RunItem


logger = logging.getLogger(__name__)


_VALID_STATUS_VALUES: frozenset[str] = frozenset(
    {"running", "completed", "failed", "interrupted", "max_supersteps", "max_tokens", "no_ready_nodes"}
)
"""Allowlist of ``GraphState.status`` values accepted on deserialisation.

Prevents arbitrary strings in a persisted payload from propagating into
``GraphRunResult.status`` where consumer logic branches on it. Must cover
every ``GraphRunStatus`` value plus the ``"running"`` initial sentinel — the
loop emits ``max_supersteps`` / ``max_tokens`` / ``no_ready_nodes`` terminal
statuses, so a checkpoint saved in those states must still reload.
"""


TContext = TypeVar("TContext")


@dataclass
class GraphState[TContext]:
    """Per-run mutable state of a graph execution.

    Held by :func:`~troopai.adk.run.graph_loop.run_graph_loop` across
    supersteps. Passed to :class:`GraphHooks` callbacks,
    :class:`Checkpointer`\\ s, and edge predicates so they can inspect
    progress without touching the driver.

    Attributes:
        graph: The owning :class:`Graph` config. Not serialised.
        thread_id: Caller-supplied identifier for a logical run. When
            a :class:`Checkpointer` is attached this is the key under
            which snapshots are stored. ``None`` when no checkpointer
            was configured. Auto-generated via ``uuid.uuid4().hex[:12]``
            on :meth:`GraphState.new` if the caller does not supply
            one, mirroring :meth:`Graph.new`'s id behaviour.
        superstep: Monotonically-increasing superstep counter
            (1-indexed once the first superstep starts). ``0`` before
            any node has fired.
        node_results: Latest :class:`NodeResult` per node id. Read by
            :class:`Merge` strategies (when a node has multiple
            incoming edges) and by edge predicates. In cycles, each
            re-firing of a node overwrites its prior entry.
        versions_seen: Per-(node, upstream) map of the last superstep
            at which ``node`` consumed input from ``upstream``. Used
            by :class:`Checkpointer` restores to decide whether a
            node should re-fire. See the module docstring.
        produced_at: Per-node superstep at which that node's current
            ``node_results`` entry was produced. Paired with
            ``versions_seen`` to drive selective re-fire on resume.
        all_items: Append-only Layer 3 audit trail of every item
            produced by every node, in completion order.
        cumulative_usage: Cumulative LLM usage across the whole graph
            run. Compared against :attr:`GraphConfig.max_total_tokens`.
        per_node_usage: Per-node-id usage attribution. Sum across
            nodes equals :attr:`cumulative_usage`.
        terminal_outputs: Outputs of terminal nodes as they fire,
            keyed by terminal node id. When every terminal has fired
            the loop exits; :attr:`final_output` is computed at that
            point.
        final_output: Aggregate output of the graph run. When there
            is exactly one terminal, this is that terminal's
            ``NodeResult.output``. When there are multiple terminals,
            this is the dict ``{terminal_id: output}``. Populated by
            the loop just before returning.
        status: Lifecycle tag — one of ``"running"`` / ``"completed"``
            / ``"failed"`` / ``"interrupted"``. Updated by the graph loop.
        error: Exception string when ``status == "failed"``. Not an
            ``Exception`` instance so the state remains JSON-safe.
        pending_interrupts: Interrupts awaiting a human reply, keyed by
            ``node_id``. Populated when a node raises
            :class:`~troopai.adk.graphs.interrupt.InterruptException`;
            cleared by the resume path.
        nested_agent_snapshots: Mid-execution sub-agent state for nodes
            paused on a :class:`~troopai.adk.graphs.interrupt.NestedAgentInterrupt`.
            Serialised via :meth:`RunState.to_dict`; rehydrated on load.
        nested_graph_snapshots: Mid-execution inner-graph state for nodes
            whose executable is a :class:`~troopai.adk.graphs.graph.Graph`
            and whose inner graph suspended. Serialised via
            :meth:`GraphState.to_dict` recursively; rehydrated on load.
        resume_counts: How many times each node has been resumed.
            Incremented in the graph loop when a node is dispatched
            with a resume-reply in its input metadata.
    """

    graph: Graph[TContext]
    """The parent graph. Not serialised."""

    thread_id: str | None = None
    """Checkpointer key for this logical run. ``None`` when unset."""

    superstep: int = 0
    """1-indexed superstep counter. ``0`` before the first superstep."""

    node_results: dict[str, NodeResult] = field(default_factory=dict)
    """Latest :class:`NodeResult` per node id."""

    versions_seen: dict[str, dict[str, int]] = field(default_factory=dict)
    """``node -> {upstream -> last_consumed_superstep}`` — LangGraph-style
    per-node-per-upstream last-consumed map. Enables selective re-fire
    after checkpoint restore."""

    produced_at: dict[str, int] = field(default_factory=dict)
    """``node_id -> superstep`` at which the current ``node_results``
    entry for that node was produced. Compared against
    :attr:`versions_seen` on checkpoint restore to decide which nodes
    must re-fire (LangGraph Pregel channel-version comparison)."""

    all_items: list[RunItem] = field(default_factory=list)
    """Append-only Layer 3 audit trail across the whole run."""

    cumulative_usage: LLMUsage = field(default_factory=LLMUsage)
    """Cumulative usage across every node's inner run."""

    per_node_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Per-node usage attribution — the cost feature neither LangGraph
    nor Strands exposes on the top-level Graph result."""

    terminal_outputs: dict[str, Any] = field(default_factory=dict)
    """Outputs of terminal nodes as they fire."""

    final_output: Any = None
    """Aggregate graph output. Populated at loop exit."""

    status: Literal[
        "running", "completed", "failed", "interrupted", "max_supersteps", "max_tokens", "no_ready_nodes"
    ] = "running"
    """``"running"`` / ``"completed"`` / ``"failed"`` / ``"interrupted"`` / ``"max_supersteps"`` / ``"max_tokens"`` / ``"no_ready_nodes"``."""

    error: str | None = None
    """Serialised error message. Set when ``status == "failed"``, and also
    when one or more nodes errored during a superstep that ALSO produced an
    interrupt (``status == "interrupted"``) — so a paused run still surfaces
    the failure instead of discarding it. Not an ``Exception`` instance so
    the state stays JSON-safe."""

    pending_interrupts: dict[str, Interrupt] = field(default_factory=dict)
    """Interrupts awaiting a human reply, keyed by ``node_id``.

    Populated when a node raises :class:`~troopai.adk.graphs.interrupt.InterruptException`.
    Cleared by the resume path once the caller supplies a
    :class:`~troopai.adk.graphs.interrupt.GraphResume`."""

    nested_agent_snapshots: dict[str, RunState] = field(default_factory=dict)
    """Mid-execution sub-agent state for nodes paused on a
    :class:`~troopai.adk.graphs.interrupt.NestedAgentInterrupt`. Serialized
    via :meth:`RunState.to_dict`; rehydrated via
    :meth:`RunState.from_dict` on load."""

    nested_graph_snapshots: dict[str, GraphState[Any]] = field(default_factory=dict)
    """Mid-execution inner-graph state for nodes whose executable is a
    :class:`~troopai.adk.graphs.graph.Graph` and whose inner graph
    suspended. Parallel to :attr:`nested_agent_snapshots` for the
    graph-backed-node case. Serialized via :meth:`GraphState.to_dict`
    recursively; rehydrated via :meth:`GraphState.from_dict` on load
    (requires the inner graph to be available in the parent graph's
    node executables). Populated by the BSP loop's
    :class:`InterruptException` catch when the raised exception
    carries a non-``None`` ``_nested_graph_state`` attribute set by
    :meth:`Graph.invoke`."""

    resume_counts: dict[str, int] = field(default_factory=dict)
    """How many times each node has been resumed.

    Incremented in the graph loop when a node is dispatched with an
    :class:`Interrupt` reply (``__resume_reply__``) or a nested-agent
    reply (``__nested_agent_reply__``) in its input metadata. The current value (after increment) is stamped onto the
    per-node observability span as
    ``troopai.graph.node.resume_attempt``. Original (non-resume) firings
    leave the counter at zero and omit the attribute."""

    def record(
        self,
        node_id: str,
        result: NodeResult,
    ) -> None:
        """Record a node's firing.

        Updates ``node_results`` and ``produced_at`` (overwriting any prior entry),
        appends ``result.new_items`` to ``all_items``, and accumulates
        usage both graph-wide and per-node. Called by the graph loop
        right after :meth:`Executable.invoke` returns.

        Args:
            node_id: Id of the node that fired.
            result: Its :class:`NodeResult`.
        """
        self.node_results[node_id] = result
        self.produced_at[node_id] = self.superstep
        self.all_items.extend(result.new_items)
        self.cumulative_usage = self.cumulative_usage + result.usage
        existing = self.per_node_usage.get(node_id)
        if existing is None:
            self.per_node_usage[node_id] = result.usage
        else:
            self.per_node_usage[node_id] = existing + result.usage

    def mark_version_consumed(
        self,
        node_id: str,
        upstream_id: str,
    ) -> None:
        """Record that ``node_id`` consumed input from ``upstream_id``
        at :attr:`superstep`.

        Used by the graph loop to populate ``versions_seen``. After a
        checkpoint restore, the loop compares this map against the
        graph's edge set to decide which nodes are still "up to date"
        vs. need to re-fire.
        """
        per_upstream = self.versions_seen.setdefault(node_id, {})
        per_upstream[upstream_id] = self.superstep

    def to_dict(self) -> dict[str, Any]:
        """Emit the serialisable fields as a plain dict.

        Every value is coerced JSON-safe via :func:`_json_safe` so the
        payload survives ``json.dumps`` unchanged — a durable checkpointer
        depends on this. Intentionally omits ``graph`` (non-data
        reference). ``node_results`` are serialised as a minimal dict per
        entry (``output`` and ``metadata`` are made JSON-safe, ``usage``
        is broken down into its four counters, ``new_items`` is converted
        to Layer 1 params). ``terminal_outputs`` and ``final_output`` are
        likewise made JSON-safe: primitives and nested dict/list structure
        are preserved, and any non-serialisable leaf (e.g. an
        :class:`LLMUsage` stamped into a nested-graph node's metadata) is
        ``str``-coerced rather than crashing the dump.
        """
        from troopai.adk.types.items.items import ItemHelpers

        return {
            "thread_id": self.thread_id,
            "superstep": self.superstep,
            "node_results": _serialise_node_results(self.node_results),
            "versions_seen": {k: dict(v) for k, v in self.versions_seen.items()},
            "produced_at": dict(self.produced_at),
            "all_items": list(ItemHelpers.run_items_to_params(self.all_items)),
            "cumulative_usage": _serialise_usage(self.cumulative_usage),
            "per_node_usage": {name: _serialise_usage(u) for name, u in self.per_node_usage.items()},
            "terminal_outputs": {k: _json_safe(v) for k, v in self.terminal_outputs.items()},
            "final_output": _json_safe(self.final_output),
            "status": self.status,
            "error": self.error,
            "pending_interrupts": self._serialise_pending_interrupts(),
            "nested_agent_snapshots": {nid: snap.to_dict() for nid, snap in self.nested_agent_snapshots.items()},
            "nested_graph_snapshots": {nid: inner.to_dict() for nid, inner in self.nested_graph_snapshots.items()},
            "resume_counts": dict(self.resume_counts),
        }

    def _serialise_pending_interrupts(self) -> dict[str, dict[str, Any]]:
        """Serialise :attr:`pending_interrupts` to plain dicts.

        Uses :func:`dataclasses.asdict` so subclass fields (``agent_name``,
        ``tool_call_ids`` on :class:`NestedAgentInterrupt`) are included
        automatically. The discriminator ``kind`` survives the round-trip
        and drives :meth:`_rehydrate_pending_interrupts` on load.
        """
        return {nid: dataclasses.asdict(interrupt) for nid, interrupt in self.pending_interrupts.items()}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        graph: Graph[TContext],
    ) -> GraphState[TContext]:
        """Reconstruct a :class:`GraphState` from :meth:`to_dict` output.

        Requires the caller to re-supply the :class:`Graph`. Validates
        that every ``node_id`` key referenced in the payload is a
        known node in the graph — phantom ids are rejected rather
        than silently imported (mirrors ``SwarmState.from_dict`` for
        consistency with project rules).

        Args:
            data: Payload from :meth:`to_dict`.
            graph: The parent :class:`Graph`.

        Returns:
            A reconstructed :class:`GraphState`.

        Raises:
            ValueError: When the payload references node ids not
                present in ``graph``, when ``status`` carries an
                unknown value, when a :class:`NestedAgentInterrupt`
                entry has missing/invalid required fields, or when a
                :class:`NestedAgentInterrupt` has no matching entry in
                ``nested_agent_snapshots``.
        """
        from troopai.adk.types.items.items import ItemHelpers

        # Residual debt: the body still tops 60 lines because the
        # node-id validation block runs three independent checks
        # (node_results / versions_seen / produced_at) and the final
        # ``cls(...)`` constructor enumerates every field. Both are
        # structural and cannot be hoisted without obscuring intent.
        known_ids = {n.id for n in graph.nodes}
        for key in data.get("node_results", {}):
            if key not in known_ids:
                raise ValueError(
                    f"GraphState.from_dict: node_results has unknown node id {key!r}. Known ids: {sorted(known_ids)}."
                )
        for key in data.get("versions_seen", {}):
            if key not in known_ids:
                raise ValueError(
                    f"GraphState.from_dict: versions_seen has unknown node id {key!r}. Known ids: {sorted(known_ids)}."
                )
        for key in data.get("produced_at", {}):
            if key not in known_ids:
                raise ValueError(
                    f"GraphState.from_dict: produced_at has unknown node id {key!r}. Known ids: {sorted(known_ids)}."
                )

        status_raw = data.get("status", "running")
        if status_raw not in _VALID_STATUS_VALUES:
            raise ValueError(
                f"GraphState.from_dict: status has unknown value "
                f"{status_raw!r}. Expected one of {sorted(_VALID_STATUS_VALUES)}."
            )

        pending_interrupts = _rehydrate_pending_interrupts(data)
        nested_agent_snapshots = _rehydrate_nested_agent_snapshots(data)
        nested_graph_snapshots = _rehydrate_nested_graph_snapshots(data, graph)

        # Cross-reference: every NestedAgentInterrupt MUST have a
        # matching snapshot in EITHER nested_agent_snapshots (agent-
        # backed inner) OR nested_graph_snapshots (graph-backed inner,
        # PA4). Without one the resume path would deadlock.
        for nid, interrupt in pending_interrupts.items():
            if isinstance(interrupt, NestedAgentInterrupt) and (
                nid not in nested_agent_snapshots and nid not in nested_graph_snapshots
            ):
                raise ValueError(
                    f"GraphState.from_dict: pending_interrupts[{nid!r}] is a "
                    f"NestedAgentInterrupt but neither nested_agent_snapshots nor "
                    f"nested_graph_snapshots has a matching entry — checkpoint is "
                    f"inconsistent and would deadlock resume."
                )
            if isinstance(interrupt, NestedGraphInterrupt) and nid not in nested_graph_snapshots:
                raise ValueError(
                    f"GraphState.from_dict: pending_interrupts[{nid!r}] is a "
                    f"NestedGraphInterrupt but nested_graph_snapshots has no matching "
                    f"entry — checkpoint is inconsistent and would deadlock resume."
                )

        return cls(
            graph=graph,
            thread_id=data.get("thread_id"),
            superstep=data.get("superstep", 0),
            node_results=_rehydrate_node_results(data),
            versions_seen={k: dict(v) for k, v in data.get("versions_seen", {}).items()},
            produced_at=dict(data.get("produced_at", {})),
            all_items=list(ItemHelpers.messages_to_run_items(data.get("all_items", []))),
            cumulative_usage=_rehydrate_usage(data.get("cumulative_usage", {})),
            per_node_usage={name: _rehydrate_usage(u) for name, u in data.get("per_node_usage", {}).items()},
            terminal_outputs=dict(data.get("terminal_outputs", {})),
            final_output=data.get("final_output"),
            status=status_raw,
            error=data.get("error"),
            pending_interrupts=pending_interrupts,
            nested_agent_snapshots=nested_agent_snapshots,
            nested_graph_snapshots=nested_graph_snapshots,
            resume_counts=dict(data.get("resume_counts", {})),
        )

    def to_json(self) -> str:
        """Serialise to a JSON string — ``json.dumps(self.to_dict())``.

        Use for on-disk or cross-process persistence. For in-process
        handoff prefer :meth:`to_dict`.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls,
        raw: str,
        graph: Graph[TContext],
    ) -> GraphState[TContext]:
        """Deserialise from :meth:`to_json` output — ``from_dict(json.loads(raw))``.

        Args:
            raw: JSON string produced by :meth:`to_json`.
            graph: The parent :class:`Graph` used to validate node ids
                (same contract as :meth:`from_dict`).

        Returns:
            A reconstructed :class:`GraphState`.

        Raises:
            ValueError: When an unknown node id appears in the payload
                (validated by ``from_dict`` against the supplied
                ``Graph``).
            json.JSONDecodeError: When ``raw`` is not valid JSON.
        """
        return cls.from_dict(json.loads(raw), graph)


def _serialise_usage(usage: LLMUsage) -> dict[str, int]:
    """Serialise an :class:`LLMUsage` to its four-counter wire form."""
    return {
        "requests": usage.requests,
        "total_tokens": usage.total_tokens,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _json_safe(value: Any) -> Any:
    """Coerce an arbitrary value into a JSON-serialisable shape.

    Primitives (``str`` / ``int`` / ``float`` / ``bool`` / ``None``) pass
    through untouched; ``dict`` and ``list`` / ``tuple`` recurse so a
    non-serialisable leaf is coerced in place; anything else (a dataclass
    such as :class:`LLMUsage`, a Pydantic model, an arbitrary object) is
    ``str``-coerced.

    Without this a durable checkpointer's ``json.dumps(state.to_dict())``
    crashes — and fails the whole run — on a value the runtime deposits
    inside :attr:`NodeResult.metadata` or :attr:`NodeResult.output`. A
    nested-graph node, for example, stamps ``per_node_usage`` onto its
    metadata as a ``dict[str, LLMUsage]``, and :class:`LLMUsage` is a
    dataclass that ``json`` cannot serialise.

    Dict keys are ``str``-coerced because JSON object keys are strings.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _serialise_node_results(node_results: dict[str, NodeResult]) -> dict[str, dict[str, Any]]:
    """Serialise :attr:`GraphState.node_results` to wire form.

    Each entry's ``output`` is coerced to ``str`` when not a JSON
    primitive; ``new_items`` are converted to Layer 1 params; ``usage``
    breaks down into the four counter fields.
    """
    from troopai.adk.types.items.items import ItemHelpers

    payload: dict[str, dict[str, Any]] = {}
    for node_id, r in node_results.items():
        payload[node_id] = {
            "output": _json_safe(r.output),
            "final_text": r.final_text,
            "usage": _serialise_usage(r.usage),
            "new_items": list(ItemHelpers.run_items_to_params(r.new_items)),
            "metadata": _json_safe(r.metadata),
        }
    return payload


def _rehydrate_usage(data: dict[str, Any]) -> LLMUsage:
    """Rebuild an :class:`LLMUsage` from its four-counter wire form."""
    return LLMUsage(
        requests=data.get("requests", 0),
        total_tokens=data.get("total_tokens", 0),
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
    )


def _rehydrate_node_results(data: dict[str, Any]) -> dict[str, NodeResult]:
    """Rebuild :attr:`GraphState.node_results` from a serialised payload."""
    from troopai.adk.orchestration.executable import NodeResult
    from troopai.adk.types.items.items import ItemHelpers

    node_results: dict[str, NodeResult] = {}
    for node_id, r in data.get("node_results", {}).items():
        node_results[node_id] = NodeResult(
            output=r.get("output"),
            new_items=list(ItemHelpers.messages_to_run_items(r.get("new_items", []))),
            usage=_rehydrate_usage(r.get("usage", {})),
            final_text=r.get("final_text"),
            metadata=dict(r.get("metadata", {})),
        )
    return node_results


def _rehydrate_pending_interrupts(data: dict[str, Any]) -> dict[str, Interrupt]:
    """Rebuild :attr:`GraphState.pending_interrupts` from a serialised payload.

    The discriminator ``kind`` selects between :class:`Interrupt` and
    :class:`NestedAgentInterrupt`. Nested-agent entries validate their
    required fields (``agent_name`` non-empty string, ``tool_call_ids``
    present) and raise :class:`ValueError` on any mismatch — silent
    fall-through would yield a degraded interrupt that the resume path
    cannot honour.

    Args:
        data: The :meth:`GraphState.to_dict` payload.

    Returns:
        A dict ``node_id -> Interrupt`` (or ``NestedAgentInterrupt``).

    Raises:
        ValueError: When a nested-agent entry is missing required
            fields or carries fields of the wrong type.
    """
    pending_interrupts: dict[str, Interrupt] = {}
    for nid, interrupt_data in data.get("pending_interrupts", {}).items():
        if interrupt_data.get("kind") == NESTED_AGENT_TOOL_APPROVAL_KIND:
            agent_name_raw = interrupt_data.get("agent_name")
            if not isinstance(agent_name_raw, str) or len(agent_name_raw) == 0:
                raise ValueError(
                    f"GraphState.from_dict: pending_interrupts[{nid!r}] has kind="
                    f"{NESTED_AGENT_TOOL_APPROVAL_KIND!r} but 'agent_name' is missing "
                    f"or not a non-empty string."
                )
            tool_call_ids_raw = interrupt_data.get("tool_call_ids")
            if tool_call_ids_raw is None:
                raise ValueError(
                    f"GraphState.from_dict: pending_interrupts[{nid!r}] has kind="
                    f"{NESTED_AGENT_TOOL_APPROVAL_KIND!r} but 'tool_call_ids' is missing."
                )
            pending_interrupts[nid] = NestedAgentInterrupt(
                node_id=interrupt_data.get("node_id", nid),
                question=interrupt_data.get("question", ""),
                metadata=dict(interrupt_data.get("metadata", {})),
                agent_name=agent_name_raw,
                tool_call_ids=tuple(tool_call_ids_raw),
            )
        elif interrupt_data.get("kind") == NESTED_GRAPH_INTERRUPT_KIND:
            # Lifted plain inner Interrupt — rehydrate as NestedGraphInterrupt
            # with NO agent_name guard (unlike the nested-agent branch above).
            # This is what keeps an outer graph resumable after a
            # lifted-plain-Interrupt checkpoint.
            pending_interrupts[nid] = NestedGraphInterrupt(
                node_id=interrupt_data.get("node_id", nid),
                question=interrupt_data.get("question", ""),
                metadata=dict(interrupt_data.get("metadata", {})),
            )
        else:
            pending_interrupts[nid] = Interrupt(
                node_id=interrupt_data.get("node_id", nid),
                question=interrupt_data.get("question", ""),
                kind=interrupt_data.get("kind", "generic"),
                metadata=dict(interrupt_data.get("metadata", {})),
            )
    return pending_interrupts


def _rehydrate_nested_graph_snapshots(
    data: dict[str, Any],
    graph: Graph[Any],
) -> dict[str, GraphState[Any]]:
    """Rebuild :attr:`GraphState.nested_graph_snapshots` from a serialised payload.

    Each entry is the inner :class:`GraphState` of a node whose
    executable is itself a :class:`Graph`. The inner graph instance is
    looked up off the parent ``graph``'s node executables so the
    rehydrate path can recurse into the inner state's own validations.

    Returns an empty dict when the payload omits the field — callers that
    carry no nested graph state load without error.
    """
    out: dict[str, GraphState[Any]] = {}
    for nid, payload in data.get("nested_graph_snapshots", {}).items():
        # The inner graph is the executable on the parent node. We need
        # it to validate node ids during inner-state rehydration.
        try:
            executable = graph.get_node(nid).executable
        except KeyError as exc:
            raise ValueError(
                f"GraphState.from_dict: nested_graph_snapshots entry {nid!r} is not a known node id"
            ) from exc
        from troopai.adk.graphs.graph import Graph as GraphCls

        if not isinstance(executable, GraphCls):
            raise ValueError(
                f"GraphState.from_dict: nested_graph_snapshots has entry for "
                f"node {nid!r} but its executable is "
                f"{type(executable).__name__}, not a Graph — checkpoint is "
                f"inconsistent."
            )
        out[nid] = GraphState.from_dict(payload, executable)
    return out


def _rehydrate_nested_agent_snapshots(data: dict[str, Any]) -> dict[str, RunState]:
    """Rebuild :attr:`GraphState.nested_agent_snapshots` from a serialised payload.

    Defers to :meth:`RunState.from_dict` for each entry. A missing
    field yields an empty dict — the cross-reference check in
    :meth:`GraphState.from_dict` is the authority for refusing a
    payload that has a nested-interrupt without its snapshot.
    """
    return {nid: RunState.from_dict(payload) for nid, payload in data.get("nested_agent_snapshots", {}).items()}


__all__ = [
    "GraphState",
]
