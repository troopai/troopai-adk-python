"""Durable execution engine Protocol and activity configuration types.

:class:`DurableEngine` is a :class:`typing.Protocol` that any durable
execution backend must satisfy.  Concrete implementations live in
sub-packages of this module (``temporal/``, ``restate/``) and are never
imported here to keep the core package dependency-free.

The two frozen dataclasses (:class:`ModelActivityConfig` and
:class:`ToolActivityConfig`) carry cost-conservative defaults so callers
only pay for durability they explicitly configure.

Exports:
    ModelActivityConfig: Retry and timeout policy for LLM-call activities.
    ToolActivityConfig: Retry and timeout policy for tool-call activities.
    DurableEngine: Runtime-checkable Protocol for durable execution engines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.tools.function_tool import FunctionTool


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ModelActivityConfig:
    """Retry and timeout policy for LLM-call activities.

    Applied by the engine to every ``wrap_llm`` invocation.  Defaults
    are cost-conservative: a 5-minute per-attempt ceiling, a 60-second
    heartbeat guard, and a single attempt (no retries) so a wrapped
    LLM call is never re-billed unless the developer opts in.

    Attributes:
        start_to_close_timeout: Maximum wall-clock time for a single
            LLM-call attempt, in seconds.  Default ``300`` (5 minutes).
        heartbeat_timeout: Maximum gap between heartbeat signals before
            the worker is considered lost, in seconds.  Default ``60``.
        maximum_attempts: Total attempts, including the first.  ``1``
            disables retries.  Default ``1`` (no retries); raise it to
            opt into automatic, token-billed retries.
        initial_interval: Seconds before the first retry.  Default
            ``1``.
        backoff_coefficient: Multiplier applied to the interval after
            each failure.  Default ``2.0`` (doubles each retry).
        non_retryable_error_types: Tuple of error type names (strings)
            that must not be retried — typically ``"ClientError"`` for
            4xx-class failures from the LLM provider.
    """

    start_to_close_timeout: int = 300
    """Per-attempt wall-clock ceiling in seconds."""

    heartbeat_timeout: int = 60
    """Heartbeat gap ceiling in seconds."""

    maximum_attempts: int = 1
    """Total attempts including the first."""

    initial_interval: int = 1
    """Seconds before the first retry."""

    backoff_coefficient: float = 2.0
    """Retry interval multiplier."""

    non_retryable_error_types: tuple[str, ...] = ("ClientError",)
    """Error type names that bypass retry logic."""

    def __post_init__(self) -> None:
        if self.start_to_close_timeout <= 0:
            msg = f"start_to_close_timeout must be > 0, got {self.start_to_close_timeout}"
            raise ValueError(msg)
        if self.heartbeat_timeout <= 0:
            msg = f"heartbeat_timeout must be > 0, got {self.heartbeat_timeout}"
            raise ValueError(msg)
        if self.maximum_attempts < 1:
            msg = f"maximum_attempts must be >= 1, got {self.maximum_attempts}"
            raise ValueError(msg)
        if self.backoff_coefficient < 1.0:
            msg = f"backoff_coefficient must be >= 1.0, got {self.backoff_coefficient}"
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class ToolActivityConfig:
    """Retry and timeout policy for tool-call activities.

    Applied by the engine to every ``wrap_tool`` invocation.  The
    per-attempt timeout is tighter than :class:`ModelActivityConfig`
    because tool calls are expected to be short-lived; retries stay off
    by default so a wrapped tool call is never re-run unless the
    developer opts in.

    Attributes:
        start_to_close_timeout: Maximum wall-clock time for a single
            tool-call attempt, in seconds.  Default ``30``.
        maximum_attempts: Total attempts, including the first.  ``1``
            disables retries.  Default ``1`` (no retries); raise it to
            opt into automatic re-runs of the tool call.
    """

    start_to_close_timeout: int = 30
    """Per-attempt wall-clock ceiling in seconds."""

    maximum_attempts: int = 1
    """Total attempts including the first."""

    def __post_init__(self) -> None:
        if self.start_to_close_timeout <= 0:
            msg = f"start_to_close_timeout must be > 0, got {self.start_to_close_timeout}"
            raise ValueError(msg)
        if self.maximum_attempts < 1:
            msg = f"maximum_attempts must be >= 1, got {self.maximum_attempts}"
            raise ValueError(msg)


@runtime_checkable
class DurableEngine(Protocol):
    """Runtime-checkable Protocol for durable execution engines.

    A ``DurableEngine`` wraps :class:`~troopai.adk.llms.llm.LLM` calls and
    :class:`~troopai.adk.tools.function_tool.FunctionTool` calls so they
    survive worker crashes via journaling or checkpointing.  Concrete
    backends (``temporal/``, ``restate/``) implement all three methods.

    Methods are intentionally thin — callers pass a factory callable and
    receive a factory callable back, which keeps the Protocol composable
    with existing runner infrastructure without coupling to any one
    engine's activity API.
    """

    def wrap_llm(
        self,
        llm: LLM,
        *,
        config: ModelActivityConfig,
    ) -> LLM:
        """Return a durable wrapper around ``llm``.

        Args:
            llm: The :class:`~troopai.adk.llms.llm.LLM` instance to wrap.
            config: Retry and timeout policy for the activity.

        Returns:
            A :class:`~troopai.adk.llms.llm.LLM` that delegates to
            ``llm`` but executes inside the engine's durable activity
            boundary.
        """
        ...

    def wrap_tool(
        self,
        tool: FunctionTool,
        *,
        config: ToolActivityConfig,
    ) -> FunctionTool:
        """Return a durable wrapper around ``tool``.

        Args:
            tool: The :class:`~troopai.adk.tools.function_tool.FunctionTool`
                to wrap.
            config: Retry and timeout policy for the activity.

        Returns:
            A :class:`~troopai.adk.tools.function_tool.FunctionTool` that
            delegates to ``tool`` inside the engine's durable activity
            boundary.
        """
        ...

    def in_durable_context(self) -> bool:
        """Return ``True`` when called from within a durable execution context.

        Used by runner infrastructure to gate journal-only operations
        (side-effect recording, deterministic sleep) that must only run
        inside an active workflow execution.

        Returns:
            ``True`` if the current call stack is executing inside the
            engine's durable context; ``False`` otherwise.
        """
        ...
