"""Tests for the generated JSON Schema artifact.

The schema is generated from the Pydantic models (the single source of
truth) and committed so it can be published for editor/CI validation. A
drift guard ensures the committed file always matches the models.
"""

from __future__ import annotations

import json

from troopai.adk.config.schema import (
    AGENT_CONFIG_SCHEMA_PATH,
    AGENT_NODE_CONFIG_SCHEMA_PATH,
    TOPOLOGY_CONFIG_SCHEMA_PATH,
    dump_agent_config_schema,
    dump_agent_node_config_schema,
    dump_topology_config_schema,
)


def test_schema_is_strict() -> None:
    # extra="forbid" must surface as additionalProperties: false.
    schema = dump_agent_config_schema()
    assert schema["additionalProperties"] is False


def test_schema_documents_schema_pointer() -> None:
    # The $schema pointer must be a documented property (by alias).
    schema = dump_agent_config_schema()
    assert "$schema" in schema["properties"]


def test_schema_requires_name_and_prompt() -> None:
    schema = dump_agent_config_schema()
    assert "name" in schema["required"]
    assert "system_prompt" in schema["required"]


def test_committed_schema_in_sync() -> None:
    committed = json.loads(AGENT_CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == dump_agent_config_schema(), (
        "Committed JSON Schema is stale. Regenerate it with:\n  python -m troopai.adk.config.schema"
    )


def test_agent_node_schema_documents_handoffs() -> None:
    # The sub-agent-file schema is AgentConfig plus the handoffs field.
    schema = dump_agent_node_config_schema()
    assert schema["additionalProperties"] is False
    assert "handoffs" in schema["properties"]
    assert "name" in schema["required"]


def test_committed_agent_node_schema_in_sync() -> None:
    committed = json.loads(AGENT_NODE_CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == dump_agent_node_config_schema(), (
        "Committed agent-node JSON Schema is stale. Regenerate it with:\n  python -m troopai.adk.config.schema"
    )


def test_topology_schema_is_strict_and_documents_agents() -> None:
    schema = dump_topology_config_schema()
    assert schema["additionalProperties"] is False
    assert "$schema" in schema["properties"]
    assert "agents" in schema["properties"]


def test_committed_topology_schema_in_sync() -> None:
    committed = json.loads(TOPOLOGY_CONFIG_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == dump_topology_config_schema(), (
        "Committed topology JSON Schema is stale. Regenerate it with:\n  python -m troopai.adk.config.schema"
    )


def test_schema_documents_llm_provider_block() -> None:
    schema = dump_agent_config_schema()
    # The widened llm field references provider blocks via $defs.
    assert "AnthropicProviderBlock" in schema["$defs"]
    assert "llm_config" in schema["properties"]


def test_schema_documents_hosted_tool_and_guardrails() -> None:
    schema = dump_agent_config_schema()
    assert "HostedToolRef" in schema["$defs"]
    assert "GuardrailsConfig" in schema["$defs"]
    assert "DynamicPromptRef" in schema["$defs"]
    assert "guardrails" in schema["properties"]
