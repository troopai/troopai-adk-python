"""``DeployTarget`` — the uniform seam every cloud target implements.

Each target renders its artifacts from a :class:`DeployContext`.
``generate`` is the uniform part ``deploy init`` calls for every target;
build/ship actions are target-specific methods invoked by the matching
CLI subcommand, because their flags differ per cloud.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from troopai.adk.deploy.context import DeployContext


class DeployTarget(Protocol):
    """A deployment target that renders artifacts from a context.

    Attributes:
        key: Stable registry key (e.g. ``"docker"``, ``"k8s"``).
        required_tools: External CLIs the target's ship action needs.
    """

    key: ClassVar[str]
    required_tools: ClassVar[tuple[str, ...]]

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render this target's artifacts as ``{relative_path: content}``.

        Args:
            ctx: The deploy context.

        Returns:
            A map of relative path to file content.
        """
        ...
