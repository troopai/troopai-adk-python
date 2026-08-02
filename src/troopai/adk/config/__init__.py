"""Declarative agent configuration for the TroopAI ADK.

Builds an ``Agent`` (or a multi-agent ``AgentTopology``) from a JSON document
validated against a strict schema, instead of constructing it in Python. This
is an operator-facing layer for the data-heavy surface of an agent — name,
description, system prompt, model, tools, and output schema. Code-only
behavior (hooks, middleware, custom guardrail functions, dynamic prompts,
tool bodies) is reached through a dotted-path reference
(``"my_pkg.module:symbol"``) rather than expressed inline.

The Pydantic models are the source of truth; the file format (JSON) is just
a deserializer, and a JSON Schema is generated from the same models for
editor and CI validation.

Security: loading a config imports and later calls the modules it
references, so it executes Python — load only config files you trust.
"""

from __future__ import annotations

from troopai.adk.config.assembler import build_agent
from troopai.adk.config.dump import dump_agent
from troopai.adk.config.hosted_tools import register_hosted_tool
from troopai.adk.config.loader import load_agent
from troopai.adk.config.providers import register_llm_provider
from troopai.adk.config.resolver import resolve_dotted_spec
from troopai.adk.config.topology import AgentTopology, build_topology, load_topology

__all__ = [
    "AgentTopology",
    "build_agent",
    "build_topology",
    "dump_agent",
    "load_agent",
    "load_topology",
    "register_hosted_tool",
    "register_llm_provider",
    "resolve_dotted_spec",
]
