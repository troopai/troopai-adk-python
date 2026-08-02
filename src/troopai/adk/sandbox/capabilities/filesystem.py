"""``FilesystemCapability`` — exposes view_image + apply_patch tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, override

from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.sandbox.tools.apply_patch_tool import make_apply_patch_tool
from troopai.adk.sandbox.tools.view_image_tool import make_view_image_tool

if TYPE_CHECKING:
    from troopai.adk.tools.function_tool import FunctionTool

__all__ = ["FilesystemCapability"]


class FilesystemCapability(SandboxCapability):
    """Capability that exposes view_image + apply_patch.

    Attributes:
        type: Discriminator. Always ``"filesystem"``.
    """

    type: Literal["filesystem"] = "filesystem"
    """Discriminator."""

    @override
    def tools(self) -> list[FunctionTool]:
        if self.session is None:
            return []
        return [
            make_view_image_tool(session=self.session, user=self.run_as),
            make_apply_patch_tool(session=self.session, user=self.run_as),
        ]
