"""``ViewImageTool`` — load a workspace image as base64."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.sandbox.permissions import User

logger = logging.getLogger(__name__)

__all__ = ["ViewImageArgs", "make_view_image_tool"]


_DEFAULT_DESCRIPTION = (
    "Read an image file from the sandbox workspace and return it "
    "base64-encoded with a MIME-type hint. Use this when you need "
    "to inspect screenshots, charts, or other image artifacts "
    "the agent generated."
)


class ViewImageArgs(BaseModel):
    """Arguments for the ``view_image`` tool."""

    path: str = Field(..., description="Workspace-relative path to the image file.")
    mime_type: str | None = Field(
        default=None,
        description="Optional MIME-type override (e.g. 'image/png'); inferred from extension when omitted.",
    )


def _infer_mime(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def make_view_image_tool(
    *,
    session: BaseSandboxSession,
    user: User | str | None = None,
    name: str = "view_image",
    description: str | None = None,
) -> FunctionTool:
    async def _on_invoke(ctx: ToolContext, raw_args: str) -> dict[str, Any]:
        del ctx
        parsed = ViewImageArgs.model_validate_json(raw_args)
        stream = await session.read(parsed.path, user=user)
        try:
            payload = stream.read()
        finally:
            stream.close()
        mime = parsed.mime_type or _infer_mime(parsed.path)
        return {
            "mime_type": mime,
            "size_bytes": len(payload),
            "data_base64": base64.b64encode(payload).decode("ascii"),
        }

    return FunctionTool(
        name=name,
        description=description or _DEFAULT_DESCRIPTION,
        schema=ViewImageArgs,
        on_invoke=_on_invoke,
    )
