"""Resource-usage accumulator for sandbox sessions.

Mirrors the ``LLMUsage`` pattern in ``types/tokens/llm_usage.py``:
``SandboxSingleExecUsage`` captures per-command counters,
``SandboxUsage`` aggregates them with a list of per-command
breakdowns and supports ``__add__`` so the Runner can fold multiple
sessions (handoffs across SandboxAgents) into a single run-level
total.

Backends populate counters where they can; ``None``-tolerant
accumulation means a session that reports CPU time but not memory
peaks still gives meaningful aggregate.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

__all__ = ["SandboxSingleExecUsage", "SandboxUsage"]


@dataclass
class SandboxSingleExecUsage:
    """Per-command usage record produced by a backend.

    Attributes:
        command: The command string, truncated to 1024 chars for
            tracing safety. Backends MUST truncate or redact before
            constructing this record if the command is sensitive.
        exit_code: Process exit code.
        duration_ms: Wall-clock duration in milliseconds.
        cpu_ms: CPU time consumed by the command (user + system),
            in milliseconds. ``0`` when the backend cannot report.
        memory_peak_mb: Peak RSS observed during the command, MiB.
            ``0`` when unreported.
        bytes_read: Bytes the command read from the filesystem or
            stdin. ``0`` when unreported.
        bytes_written: Bytes the command wrote to the filesystem or
            stdout/stderr. ``0`` when unreported.
        cost_usd: Computed dollar cost for this command
            (``rate × duration``); ``0.0`` for a free backend, ``None``
            only when the backend declares no rate card (``client.cost``
            is ``None``).
    """

    command: str
    """The command string (truncated)."""

    exit_code: int
    """Process exit code."""

    duration_ms: int = 0
    """Wall-clock duration in milliseconds."""

    cpu_ms: int = 0
    """CPU time consumed (user + system), milliseconds."""

    memory_peak_mb: int = 0
    """Peak RSS observed during the command, MiB."""

    bytes_read: int = 0
    """Bytes read from the filesystem or stdin."""

    bytes_written: int = 0
    """Bytes written to the filesystem or stdout/stderr."""

    cost_usd: float | None = None
    """Computed dollar cost for this command (``rate × duration``); ``0.0``
    for a free backend, ``None`` only when the backend declares no rate card
    (``client.cost`` is ``None``)."""


@dataclass(kw_only=True)
class SandboxUsage:
    """Cumulative resource usage across one or more sandbox sessions.

    The Runner constructs one ``SandboxUsage`` per ``Runner.arun()``
    call and accumulates per-command records via ``add_exec`` (handoffs
    between SandboxAgents share that one accumulator — the session is
    bracketed once per run). ``__add__`` is available for explicitly
    aggregating usage across separate runs.

    Attributes:
        exec_count: Number of commands run.
        total_duration_ms: Sum of per-command wall-clock durations.
        cpu_ms: Sum of per-command CPU times.
        memory_peak_mb: Maximum ``memory_peak_mb`` observed across
            commands (max, NOT sum — memory peaks don't accumulate).
        bytes_read: Sum of per-command bytes read.
        bytes_written: Sum of per-command bytes written.
        computed_cost_usd: Sum of per-command ``cost_usd`` (rate-card
            estimate of session dollar cost).
        billed_cost_usd: Provider-reported session cost; set only when
            live billing retrieval ran (otherwise ``None``).
        executions: Per-command breakdown.
    """

    exec_count: int = 0
    """Number of commands run."""

    total_duration_ms: int = 0
    """Sum of per-command wall-clock durations (ms)."""

    cpu_ms: int = 0
    """Sum of per-command CPU times (ms)."""

    memory_peak_mb: int = 0
    """Maximum per-command peak RSS (MiB)."""

    bytes_read: int = 0
    """Sum of per-command bytes read."""

    bytes_written: int = 0
    """Sum of per-command bytes written."""

    computed_cost_usd: float = 0.0
    """Sum of per-command ``cost_usd`` (rate-card estimate)."""

    billed_cost_usd: float | None = None
    """Provider-reported session cost; set only when live billing ran."""

    executions: list[SandboxSingleExecUsage] = field(default_factory=list)
    """Per-command breakdown records."""

    def add_exec(self, record: SandboxSingleExecUsage) -> None:
        """Fold a single-command record into this accumulator."""
        self.executions.append(record)
        self.exec_count += 1
        self.total_duration_ms += record.duration_ms
        self.cpu_ms += record.cpu_ms
        self.bytes_read += record.bytes_read
        self.bytes_written += record.bytes_written
        if record.memory_peak_mb > self.memory_peak_mb:
            self.memory_peak_mb = record.memory_peak_mb
        if record.cost_usd is not None:
            self.computed_cost_usd += record.cost_usd

    def __add__(self, other: SandboxUsage) -> SandboxUsage:
        """Merge two accumulators into a new one.

        Counters sum elementwise; ``memory_peak_mb`` takes the max;
        ``executions`` are concatenated in left-then-right order;
        ``billed_cost_usd`` sums when either operand has one (else stays
        ``None``). Neither operand is mutated.
        """
        merged = copy(self)
        merged.executions = list(self.executions) + list(other.executions)
        merged.exec_count = self.exec_count + other.exec_count
        merged.total_duration_ms = self.total_duration_ms + other.total_duration_ms
        merged.cpu_ms = self.cpu_ms + other.cpu_ms
        merged.bytes_read = self.bytes_read + other.bytes_read
        merged.bytes_written = self.bytes_written + other.bytes_written
        merged.memory_peak_mb = max(self.memory_peak_mb, other.memory_peak_mb)
        merged.computed_cost_usd = self.computed_cost_usd + other.computed_cost_usd
        if self.billed_cost_usd is None and other.billed_cost_usd is None:
            merged.billed_cost_usd = None
        else:
            left = self.billed_cost_usd if self.billed_cost_usd is not None else 0.0
            right = other.billed_cost_usd if other.billed_cost_usd is not None else 0.0
            merged.billed_cost_usd = left + right
        return merged
