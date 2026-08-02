"""Cost + capability descriptors for sandbox backend selection.

A backend advertises what it can do (``SandboxBackendCapabilities``)
and what it costs (``SandboxCostDescriptor``); a run states what it
needs (``SandboxRequirements``). A ``SandboxSelector`` matches the two.
Live provider-reported cost arrives as a ``SandboxBillingRecord``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel, field_validator

__all__ = [
    "SandboxBackendCapabilities",
    "SandboxBillingRecord",
    "SandboxCostDescriptor",
    "SandboxRequirements",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SandboxRequirements:
    """What a single run needs from a sandbox backend.

    Cost-conservative defaults: a run asks for nothing, so any
    backend qualifies until the developer states a constraint.

    Attributes:
        network: Run needs outbound network access.
        persistent: Run needs a persistent (non-ephemeral) workspace.
        min_cpu: Minimum CPU count the backend must provide.
        min_memory_mb: Minimum memory (MiB) the backend must provide.
        region: Required region; the backend must list it.
    """

    network: bool = False
    """Run needs outbound network access."""

    persistent: bool = False
    """Run needs a persistent (non-ephemeral) workspace."""

    min_cpu: int | None = None
    """Minimum CPU count the backend must provide."""

    min_memory_mb: int | None = None
    """Minimum memory (MiB) the backend must provide."""

    region: str | None = None
    """Required region; the backend must list it among its regions."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SandboxCostDescriptor:
    """Static rate card a backend advertises for cost-aware selection.

    Attributes:
        usd_per_minute: Dollar cost per wall-clock minute of session
            life. The scalar selection ranks on.
        usd_per_cpu_second: Optional finer-grained CPU rate (for live
            billing reconciliation; not used by per-command estimates).
        usd_per_gb_second: Optional memory rate (same role).
        free: When True the backend costs nothing (self-hosted /
            local); ``rate_key`` and ``cost_for_ms`` return 0.
    """

    usd_per_minute: float = 0.0
    """Dollar cost per wall-clock minute of session life."""

    usd_per_cpu_second: float | None = None
    """Optional CPU-second rate (live-billing reconciliation)."""

    usd_per_gb_second: float | None = None
    """Optional GiB-second memory rate (live-billing reconciliation)."""

    free: bool = False
    """When True the backend costs nothing (self-hosted / local)."""

    def __post_init__(self) -> None:
        if self.usd_per_minute < 0:
            raise ValueError("SandboxCostDescriptor.usd_per_minute must be >= 0")

    def rate_key(self) -> float:
        """Scalar used to rank backends cheapest-first (``free`` => 0)."""
        if self.free:
            return 0.0
        return self.usd_per_minute

    def cost_for_ms(self, duration_ms: int) -> float:
        """Computed dollar cost for a command of ``duration_ms`` wall-clock.

        A negative ``duration_ms`` (a backend clock-skew artifact) clamps
        to 0 so a bad measurement can never credit cost.
        """
        if self.free:
            return 0.0
        return self.usd_per_minute * (max(0, duration_ms) / 60000.0)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SandboxBackendCapabilities:
    """What a backend can do — matched against ``SandboxRequirements``.

    Conservative defaults (no network, ephemeral, unknown limits) so a
    backend that declares nothing only satisfies an empty requirement.

    Attributes:
        network: Backend grants outbound network access.
        persistent: Backend offers a persistent workspace.
        max_cpu: Maximum CPU count available (``None`` => unknown).
        max_memory_mb: Maximum memory (MiB) available (``None`` => unknown).
        regions: Regions the backend can run in.
    """

    network: bool = False
    """Backend grants outbound network access."""

    persistent: bool = False
    """Backend offers a persistent (non-ephemeral) workspace."""

    max_cpu: int | None = None
    """Maximum CPU count available (``None`` => unknown / unconstrained-down)."""

    max_memory_mb: int | None = None
    """Maximum memory (MiB) available (``None`` => unknown)."""

    regions: tuple[str, ...] = ()
    """Regions the backend can run in."""

    def satisfies(self, requirements: SandboxRequirements) -> bool:
        """True iff this backend meets every stated requirement."""
        if requirements.network and not self.network:
            return False
        if requirements.persistent and not self.persistent:
            return False
        if requirements.min_cpu is not None and (self.max_cpu is None or self.max_cpu < requirements.min_cpu):
            return False
        if requirements.min_memory_mb is not None and (
            self.max_memory_mb is None or self.max_memory_mb < requirements.min_memory_mb
        ):
            return False
        return requirements.region is None or requirements.region in self.regions


class SandboxBillingRecord(BaseModel):
    """Provider-reported cost for a sandbox session (live billing).

    Returned by ``BaseSandboxClient.fetch_billing`` when
    ``capture_live_cost`` is enabled. A received/validated type, hence
    a Pydantic ``BaseModel``.

    Attributes:
        cost_usd: Dollar cost the provider reported for the session.
        currency: ISO currency code; the default is ``"USD"``.
        unit: Optional provider billing unit label (e.g. ``"compute-seconds"``).
        raw: Optional untouched provider payload for audit / debugging.
    """

    cost_usd: float
    currency: str = "USD"
    unit: str | None = None
    raw: dict[str, Any] | None = None

    @field_validator("cost_usd")
    @classmethod
    def _validate_cost_usd(cls, value: float) -> float:
        """Reject a negative provider cost so it can never read as a credit."""
        if value < 0:
            raise ValueError("SandboxBillingRecord.cost_usd must be >= 0")
        return value
