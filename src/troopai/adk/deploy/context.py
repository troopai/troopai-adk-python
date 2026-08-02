"""``DeployContext`` — the inputs that parameterize every deploy artifact.

A frozen, validated description of *what* to deploy (which agent, under
what image/name, on what port, with which extras and secret-env names).
Targets read it to render Dockerfiles, manifests, and CLI invocations;
they never mutate it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# RFC 1123 label — the shape Kubernetes, Cloud Run, and ECS all require for
# a resource/service name. Anchored fullmatch is applied in __post_init__.
_DNS1123_LABEL = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")

_MAX_PORT = 65535


@dataclass(frozen=True, kw_only=True)
class DeployContext:
    """Validated inputs shared by every deploy target.

    Attributes:
        agent_ref: The ``module:var`` reference the container serves via
            ``troopai serve --agent``. Baked into the image as
            ``AGENT_REF``.
        image: Container image name with optional tag/registry
            (e.g. ``"gcr.io/proj/my-agent:latest"``).
        app_name: Service/resource name. Must be an RFC 1123 label
            (lowercase alphanumerics and hyphens) so it is valid as a
            Kubernetes, Cloud Run, and ECS name.
        port: Container listen port. The image binds ``0.0.0.0`` on it
            and honors a platform-injected ``$PORT`` at runtime.
        python_version: Base-image Python minor version (e.g. ``"3.12"``).
        extras: ``troopai-adk-python`` extras installed in the image
            (e.g. ``"serve,a2a"``).
        env_keys: Names (not values) of environment variables the service
            needs — surfaced as Secret references in the manifests.
    """

    agent_ref: str
    image: str
    app_name: str
    port: int = 8080
    python_version: str = "3.12"
    extras: str = "serve,a2a"
    env_keys: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate every field at construction.

        Raises:
            ValueError: If any field is empty, the port is out of range,
                or ``app_name`` is not an RFC 1123 label.
        """
        if len(self.agent_ref) == 0:
            raise ValueError("agent_ref must be a non-empty 'module:var' reference.")
        if len(self.image) == 0:
            raise ValueError("image must be a non-empty image name.")
        if _DNS1123_LABEL.fullmatch(self.app_name) is None:
            raise ValueError(
                f"app_name {self.app_name!r} must be an RFC 1123 label "
                "(lowercase alphanumerics and hyphens, not starting/ending with a hyphen)."
            )
        if self.port < 1 or self.port > _MAX_PORT:
            raise ValueError(f"port must be in 1..{_MAX_PORT}, got {self.port}.")
        if len(self.python_version) == 0:
            raise ValueError("python_version must be non-empty (e.g. '3.12').")
        if len(self.extras) == 0:
            raise ValueError("extras must be non-empty (e.g. 'serve,a2a').")
