"""``LoadSkillTool`` — progressive-disclosure skill loader."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from troopai.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

__all__ = ["LoadSkillArgs", "make_load_skill_tool"]


_DESCRIPTION = (
    "Load a single skill (by name) into the sandbox workspace. Use "
    "this to fetch the body of a skill discovered via the index."
)


class LoadSkillArgs(BaseModel):
    """Arguments for the ``load_skill`` tool."""

    skill_name: str = Field(..., description="Name of the skill to load.")


def make_load_skill_tool(
    *,
    loader: Any,
    name: str = "load_skill",
) -> FunctionTool:
    """Build a FunctionTool that loads a skill on demand.

    ``loader`` is expected to expose an async ``load_skill(name) ->
    dict[str, str]`` method; the SkillsCapability wires the loader
    so the runtime doesn't need to know the underlying source.
    """

    async def _on_invoke(ctx: ToolContext, raw_args: str) -> dict[str, str]:
        del ctx
        parsed = LoadSkillArgs.model_validate_json(raw_args)
        payload: dict[str, str] = await loader.load_skill(parsed.skill_name)
        return payload

    return FunctionTool(
        name=name,
        description=_DESCRIPTION,
        schema=LoadSkillArgs,
        on_invoke=_on_invoke,
    )
