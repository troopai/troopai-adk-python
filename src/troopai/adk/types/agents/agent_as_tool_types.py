"""Types for the Agent.as_tool() feature.

Provides the default input schema and callback type aliases used when
wrapping an Agent as a FunctionTool for LLM-orchestrated delegation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.run import RunResult


class AgentToolInput(BaseModel):
    """Default input schema when no custom schema is provided to ``as_tool()``.

    The LLM sees a single ``input`` string field and fills it with the task
    or question it wants to delegate.

    Attributes:
        input: The task description or question to send to the sub-agent.
            The parent LLM fills this field with whatever it wants the
            delegate agent to work on.

    Example:
        When the LLM calls a tool backed by ``as_tool()``, it produces::

            {"input": "Summarize the latest quarterly earnings report."}
    """

    input: str = Field(description="The task or question to send to the agent.")
    """str: The task description or question for the sub-agent.

    This is the only field the LLM needs to fill. It should contain a
    clear, self-contained description of what the sub-agent should do.

    Example:
        >>> schema = AgentToolInput(input="Translate this to French: Hello world")
        >>> schema.input
        'Translate this to French: Hello world'
    """


AgentToolOutputExtractor = Callable[["RunResult | RunResultStreaming"], MaybeAwaitable[str]]
"""Callback to customize what the parent LLM sees from the sub-agent's result.

Takes the full ``RunResult`` (or ``RunResultStreaming`` when ``on_stream``
is used) from the sub-agent's execution and returns a string that becomes
the tool output visible to the parent LLM.

Args:
    result: The ``RunResult`` or ``RunResultStreaming`` from running the
        sub-agent.

Returns:
    A string representation of the result for the parent LLM.

Example::

    def extract_summary(result: RunResult) -> str:
        return f"Agent completed. Output: {result.final_output}"

    tool = agent.as_tool(extractor=extract_summary)
"""

AgentToolInputBuilder = Callable[[Any, "ToolContext"], MaybeAwaitable["UserPrompt"]]
"""Callback to transform parsed tool input into Agent.run() input.

Called after the LLM's JSON arguments are validated against the input schema.
Receives the parsed Pydantic model instance and the ToolContext, and returns
the input that will be passed to ``Agent.run(user_prompt=...)``.

Args:
    parsed: The validated Pydantic model instance from the LLM's arguments.
    ctx: The ``ToolContext`` for this tool invocation.

Returns:
    A string or message list to pass as input to ``Agent.run()``.

Example::

    class ResearchInput(BaseModel):
        topic: str
        depth: str = "brief"

    def build_input(parsed: ResearchInput, ctx: ToolContext) -> str:
        return f"Research {parsed.topic} in {parsed.depth} detail."

    tool = agent.as_tool(
        input_schema=ResearchInput,
        input_builder=build_input,
    )
"""
