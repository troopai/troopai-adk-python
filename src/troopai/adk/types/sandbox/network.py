"""Network policy and port-forward records for sandbox sessions.

``NetworkPolicy`` is the framework-owned, backend-agnostic declaration
of which hosts and ports the sandbox may reach. Each backend
translates it to the wire format it understands:

- ``DockerSandboxClient`` maps it to ``--network`` + iptables rules.
- ``K8sPodSandboxClient`` emits a Kubernetes ``NetworkPolicy`` CR.
- ``RemoteVMSandboxClient`` forwards it to the provider's network API.
- ``LocalSubprocessSandboxClient`` raises ``SandboxNetworkPolicyViolation``
  on ``deny_default=True`` — it cannot enforce a policy inside a
  shared-process subprocess.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

__all__ = [
    "NetworkPolicy",
    "PortForwardRule",
]


_VALID_PROTOCOLS: frozenset[str] = frozenset({"tcp", "udp"})


@dataclasses.dataclass(frozen=True, kw_only=True)
class PortForwardRule:
    """A single port-forward rule a sandbox declares for inbound traffic.

    Attributes:
        local_port: Port the application binds to INSIDE the sandbox.
        remote_port: Port the backend exposes to the host or tunnel.
            ``None`` means "let the backend pick a free port" — common
            with hosted providers that allocate tunnels dynamically.
        protocol: ``"tcp"`` (default) or ``"udp"``.
    """

    local_port: int
    """Port the application binds to INSIDE the sandbox."""

    remote_port: int | None = None
    """Port exposed to the host/tunnel; ``None`` ⇒ backend picks."""

    protocol: Literal["tcp", "udp"] = "tcp"
    """Transport protocol; ``tcp`` covers the common case."""

    def __post_init__(self) -> None:
        if self.local_port < 1 or self.local_port > 65535:
            raise ValueError(f"PortForwardRule.local_port must be in 1..65535, got {self.local_port}")
        if self.remote_port is not None and (self.remote_port < 1 or self.remote_port > 65535):
            raise ValueError(f"PortForwardRule.remote_port must be in 1..65535 or None, got {self.remote_port}")
        if self.protocol not in _VALID_PROTOCOLS:
            raise ValueError(f"PortForwardRule.protocol must be 'tcp' or 'udp', got {self.protocol!r}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class NetworkPolicy:
    """Declarative network access policy for a sandbox session.

    Cost-conservative defaults: ``deny_default=True``,
    ``allow_hosts=()``, ``allow_ports=()``, ``allow_dns=True``. Out of
    the box the sandbox can resolve DNS but cannot reach any
    host/port until the policy explicitly allows them.

    Hostnames in ``allow_hosts`` may be exact (e.g. ``"api.example.com"``)
    or wildcard-prefixed (``"*.example.com"`` matches one level of
    subdomain). IP addresses and CIDR ranges are also accepted; the
    backend interprets them.

    Attributes:
        allow_hosts: Allowed destination hostnames / IPs / CIDRs.
        allow_ports: Allowed destination TCP/UDP ports.
        deny_default: When True (default), every host/port not listed
            in ``allow_hosts``/``allow_ports`` is denied.
        allow_dns: When True (default), DNS (UDP/TCP 53) is allowed
            regardless of other rules. Most agents need name
            resolution.
        port_forwards: Inbound port-forward declarations for services
            the sandbox exposes (e.g. preview servers).
    """

    allow_hosts: tuple[str, ...] = ()
    """Allowed destination hostnames / IPs / CIDRs."""

    allow_ports: tuple[int, ...] = ()
    """Allowed destination TCP/UDP ports."""

    deny_default: bool = True
    """When True, deny everything not explicitly allowed."""

    allow_dns: bool = True
    """When True, DNS resolution is allowed regardless of other rules."""

    port_forwards: tuple[PortForwardRule, ...] = ()
    """Inbound port-forward declarations."""

    def __post_init__(self) -> None:
        for host in self.allow_hosts:
            if len(host) == 0:
                raise ValueError("NetworkPolicy.allow_hosts entries must be non-empty")
        for port in self.allow_ports:
            if port < 1 or port > 65535:
                raise ValueError(f"NetworkPolicy.allow_ports entries must be in 1..65535, got {port}")

    @classmethod
    def open(cls) -> NetworkPolicy:
        """Construct an explicitly-open policy (``deny_default=False``).

        Use SPARINGLY — this disables network isolation entirely.
        Intended for dev iteration only; production deployments
        should always declare allow lists.
        """
        return cls(deny_default=False)

    @classmethod
    def closed(cls) -> NetworkPolicy:
        """Construct a fully-closed policy (no allow lists, no DNS)."""
        return cls(deny_default=True, allow_dns=False)
