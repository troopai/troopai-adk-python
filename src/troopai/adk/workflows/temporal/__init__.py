"""Temporal.io durable execution backend for TroopAI ADK.

Wraps agent, graph, swarm, and flow runs as Temporal workflows with
crash-recovery, retry, and deterministic replay.

Install the ``temporal`` optional extra before importing this package::

    pip install "troopai-adk-python[temporal]"

Public surface exported here covers configuration types, the LLM shim,
tool wrapping helpers, the HITL workflow base class, streaming, MCP
routing, and worker wiring.
"""

from __future__ import annotations

from troopai.adk.workflows.engine import ModelActivityConfig, ToolActivityConfig
from troopai.adk.workflows.temporal.llm import TemporalLLM
from troopai.adk.workflows.temporal.mcp import TemporalMCPToolSet
from troopai.adk.workflows.temporal.plugin import TroopAITemporalPlugin
from troopai.adk.workflows.temporal.routing import (
    MappingTaskQueueRouter,
    TenantTaskQueueRouter,
    start_tenant_workflow,
)
from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM
from troopai.adk.workflows.temporal.tools import TemporalToolWrapper, activity_tool, to_durable_tool
from troopai.adk.workflows.temporal.tracing import (
    deterministic_timestamp,
    deterministic_uuid,
    should_emit_span,
)
from troopai.adk.workflows.temporal.workflow import HumanReply, ToolApprovalDecision, TroopAIWorkflow

__all__ = [
    "HumanReply",
    "MappingTaskQueueRouter",
    "ModelActivityConfig",
    "TemporalLLM",
    "TemporalMCPToolSet",
    "TemporalStreamingLLM",
    "TemporalToolWrapper",
    "TenantTaskQueueRouter",
    "ToolActivityConfig",
    "ToolApprovalDecision",
    "TroopAITemporalPlugin",
    "TroopAIWorkflow",
    "activity_tool",
    "deterministic_timestamp",
    "deterministic_uuid",
    "should_emit_span",
    "start_tenant_workflow",
    "to_durable_tool",
]
