"""Sandbox capabilities — composable extensions for SandboxAgents.

Each capability is a Pydantic ``BaseModel`` extending
``SandboxCapability``. A capability can (1) mutate the manifest
before session start, (2) expose ``FunctionTool``s bound to the
session, (3) inject an instructions fragment into the system prompt,
(4) adjust LLM sampling params, and (5) transform the input context.

Concrete capabilities (``CompactionCapability``, ``ShellCapability``,
``FilesystemCapability``, ``SkillsCapability``, ``MemoryCapability``)
all subclass ``SandboxCapability``.
"""

from __future__ import annotations

from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.sandbox.capabilities.capabilities import Capabilities
from troopai.adk.sandbox.capabilities.compaction import (
    CompactionCapability,
    CompactionModelInfo,
    CompactionPolicy,
    DynamicCompactionPolicy,
    StaticCompactionPolicy,
)

__all__ = [
    "Capabilities",
    "CompactionCapability",
    "CompactionModelInfo",
    "CompactionPolicy",
    "DynamicCompactionPolicy",
    "SandboxCapability",
    "StaticCompactionPolicy",
]
