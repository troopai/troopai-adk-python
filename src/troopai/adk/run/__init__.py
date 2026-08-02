"""Run module for agent execution.

This module provides the core execution infrastructure for running agent:
- Runner: The main execution engine with sync and async modes
- RunContext: Context that flows through execution
- RunResult: Result of an agent run
- RunResultStreaming: Streaming result with event iterator
- RunConfig: Configuration options
- RunHooks: Lifecycle callbacks
- StreamEvent: Event types for streaming

Execution modes:
- Runner.run(): Synchronous blocking execution
- Runner.arun(): Async non-blocking execution
Both accept stream=True for streaming with real-time events.

Example (sync):
    from troopai.adk.run import Runner

    result = Runner.run(agent, "Hello!")
    logger.info(result.final_output)

Example (async):
    result = await Runner.arun(agent, "Hello!")
    logger.info(result.final_output)

Example (streaming):
    from troopai.adk.run import Runner, RunItemType

    result = Runner.run(agent, "Write a story", stream=True)

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            logger.info(event.data)
        elif event.type == "run_item_stream_event":
            if event.name == RunItemType.TOOL_CALLED:
                logger.info(f"\\nCalling tool: {event.item['name']}")

    logger.info(f"\\nFinal: {result.final_output}")
"""

from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import DEFAULT_MAX_TURNS, DEFAULT_MODEL, DEFAULT_RUN_CONFIG, RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.profile import (
    AgentRunner,
    FlowRunner,
    GraphRunner,
    RunnerProfile,
    SwarmRunner,
    TaskGroupRunner,
    TaskPipelineRunner,
    TaskRunner,
)
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState
from troopai.adk.run.stream import (
    AgentUpdatedStreamEvent,
    CancelMode,
    HookEventKind,
    HookLifecycleEvent,
    QueueCompleteSentinel,
    RawResponseStreamEvent,
    RunItemStreamEvent,
    RunItemType,
    RunResultStreaming,
    StreamEvent,
)

__all__ = [
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "DEFAULT_RUN_CONFIG",
    "AgentRunner",
    "AgentUpdatedStreamEvent",
    "CancelMode",
    "FlowRunner",
    "GraphRunner",
    "HookEventKind",
    "HookLifecycleEvent",
    "QueueCompleteSentinel",
    "RawResponseStreamEvent",
    "RunConfig",
    "RunContext",
    "RunHooks",
    "RunItemStreamEvent",
    "RunItemType",
    "RunResultStreaming",
    "RunState",
    "Runner",
    "RunnerProfile",
    "StreamEvent",
    "SwarmRunner",
    "TaskGroupRunner",
    "TaskPipelineRunner",
    "TaskRunner",
]
