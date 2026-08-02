"""``ApplyPatchTool`` — apply a unified-diff patch."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.sandbox.permissions import User

logger = logging.getLogger(__name__)

__all__ = ["ApplyPatchArgs", "make_apply_patch_tool"]


_DEFAULT_DESCRIPTION = (
    "Apply a unified-diff patch to the sandbox workspace. Paths in "
    "the patch are workspace-root-relative. Returns a summary line "
    "describing the outcome (apply succeeded / apply failed + reason)."
)


class ApplyPatchArgs(BaseModel):
    """Arguments for the ``apply_patch`` tool."""

    patch: str = Field(..., description="The unified-diff patch text to apply.")


def make_apply_patch_tool(
    *,
    session: BaseSandboxSession,
    user: User | str | None = None,
    name: str = "apply_patch",
    description: str | None = None,
) -> FunctionTool:
    async def _on_invoke(ctx: ToolContext, raw_args: str) -> dict[str, Any]:
        del ctx
        parsed = ApplyPatchArgs.model_validate_json(raw_args)
        summary = await session.apply_patch(parsed.patch, user=user)
        return {
            "summary": summary,
            "patch_size_bytes": len(parsed.patch.encode("utf-8")),
        }

    return FunctionTool(
        name=name,
        description=description or _DEFAULT_DESCRIPTION,
        schema=ApplyPatchArgs,
        on_invoke=_on_invoke,
    )
