"""Restate durable execution backend for TroopAI ADK.

Alternative to the Temporal backend using Restate's journaling-based
durable execution model.  LLM calls and tool calls are routed through
``ctx.run()`` so that results are journaled and replay-safe.

Install the ``restate`` optional extra before importing this package::

    pip install "troopai-adk-python[restate]"
"""

from __future__ import annotations

from troopai.adk.workflows.restate.llm import RestateLLM
from troopai.adk.workflows.restate.service import RestateHumanReply, TroopAIRestateService
from troopai.adk.workflows.restate.tools import restate_tool

__all__ = [
    "RestateHumanReply",
    "RestateLLM",
    "TroopAIRestateService",
    "restate_tool",
]
