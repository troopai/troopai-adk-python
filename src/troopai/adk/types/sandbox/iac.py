"""Infrastructure-as-Code bundle declaration for sandbox sessions.

``IaCBundle`` declares a Terraform or Pulumi working directory the
runtime should ``apply`` before the sandbox session starts and
``destroy`` after it ends. IaC outputs can be seeded into the
sandbox environment via ``output_env_mapping`` so the agent reaches
the provisioned infrastructure without secrets ever appearing in
prompts.

The runner-side implementation that actually invokes ``terraform`` /
``pulumi`` lives in ``troopai.adk.sandbox.runner_integration.iac_runner``.
This module only describes the contract.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

__all__ = ["IaCBundle"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class IaCBundle:
    """Declarative IaC config applied before session, destroyed after.

    Attributes:
        provider: ``"terraform"`` or ``"pulumi"``. The runner picks
            the matching CLI invocation.
        working_directory: Absolute path to the IaC root (Terraform
            module root or Pulumi project root). MUST be a trusted
            host path; the runner does NOT validate IaC content.
        variables: Variables passed to the IaC tool (terraform
            ``-var`` flags / Pulumi config values). Sensitive
            variables SHOULD come from a host secret store, not be
            hard-coded here.
        output_env_mapping: Mapping of IaC output name → env var name
            inside the sandbox. After ``apply``, the runner reads the
            named outputs and seeds them as env vars so the agent's
            tools see them without seeing them in prompts.
        destroy_on_exit: When True (default), the runner runs
            ``destroy`` in the lifecycle ``finally`` block, even if
            the session failed. Set False ONLY for long-lived
            infrastructure the run is meant to leave behind.
        timeout: Wall-clock cap (seconds) for each of ``apply`` and
            ``destroy``. The runner kills the CLI process on timeout.
    """

    provider: Literal["terraform", "pulumi"]
    """``"terraform"`` or ``"pulumi"``."""

    working_directory: str
    """Absolute path to the IaC root."""

    variables: dict[str, str] = dataclasses.field(default_factory=dict)
    """Variables passed to the IaC tool."""

    output_env_mapping: dict[str, str] = dataclasses.field(default_factory=dict)
    """Mapping of IaC output name → env var name inside the sandbox."""

    destroy_on_exit: bool = True
    """Run ``destroy`` in the lifecycle ``finally`` block."""

    timeout: float = 300.0
    """Wall-clock cap (seconds) for each of ``apply`` and ``destroy``."""

    def __post_init__(self) -> None:
        if len(self.working_directory) == 0:
            raise ValueError("IaCBundle.working_directory must be non-empty")
        if not self.working_directory.startswith("/"):
            raise ValueError(f"IaCBundle.working_directory must be an absolute path, got {self.working_directory!r}")
        if self.timeout <= 0:
            raise ValueError(f"IaCBundle.timeout must be positive, got {self.timeout}")
        for output_name, env_var in self.output_env_mapping.items():
            if len(output_name) == 0:
                raise ValueError("IaCBundle.output_env_mapping keys must be non-empty")
            if len(env_var) == 0:
                raise ValueError(f"IaCBundle.output_env_mapping value for {output_name!r} must be non-empty")
