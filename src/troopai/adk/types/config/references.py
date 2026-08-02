"""Reference-bearing config sub-models.

A reference names a Python object or builtin by string rather than inlining
it. Tools are bare dotted ``ref`` strings (resolved to a ``FunctionTool``);
``OutputSchemaRef`` names an output-schema class plus how its JSON schema is
enforced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutputSchemaRef(BaseModel):
    """Reference to an output-schema class with enforcement.

    Attributes:
        ref: Dotted-path reference to the output-schema class.
        enforcement: How the generated JSON schema is processed before being
            sent to the LLM.
    """

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    """Dotted-path reference to the output-schema class."""

    enforcement: Literal["none", "normalized", "strict", "compact"] = "strict"
    """How the generated JSON schema is processed before being sent."""


class AgentFileRef(BaseModel):
    """Pointer to a standalone agent file from a topology's ``agents`` map.

    Lets a topology source a member from its own file instead of inlining it:
    ``{"config_path": "spanish.json"}``. The path is resolved relative to the
    parent topology file's directory (an absolute path is used as-is) and the
    referenced file is loaded as an ``AgentNodeConfig`` — so a file-sourced
    member may itself declare ``handoffs`` by name, wired in the same pass as
    inline members. A ``config_path`` target is a single agent file, never a
    nested topology, so resolution depth is fixed at one.

    When sub-agent files live in different directories and each references a
    tool module, give those modules distinct names: Python caches imported
    modules by name, so two files both named ``tools.py`` in different
    directories would resolve to whichever was imported first.

    Attributes:
        config_path: Path to the agent file, relative to the parent topology
            file (or absolute).
    """

    model_config = ConfigDict(extra="forbid")

    config_path: str = Field(min_length=1)
    """Path to the agent file, relative to the parent topology file."""


class HandoffRef(BaseModel):
    """Reference to a handoff target by local agent name.

    The bare-string form (just the target name) is also accepted wherever a
    handoff is declared; this object form adds an optional description.

    Attributes:
        target: Local name of the target agent (a key in the topology's
            ``agents`` map).
        description: Optional tool description shown to the LLM for this
            handoff.
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)
    """Local name of the target agent."""

    description: str | None = None
    """Optional handoff tool description shown to the LLM."""
