"""TroopAI Agents ADK - A lightweight, provider-agnostic Python framework for building multi-agent workflows.

This ADK provides:
- Agent: Autonomous entities that perform tasks using tools, guardrails, and handoffs
- Runner: Execution engine with sync, async, and streaming modes
- FunctionTool: Wrappers for Python functions with Pydantic schema validation
- Guardrails: Pre/post execution checks at both agent and tool level
- LLMConfig: Provider-agnostic configuration for 100+ LLMs via litellm

Execution modes (via Runner):
- Runner.run(): Synchronous blocking
- Runner.arun(): Async non-blocking
Both accept stream=True for streaming with real-time events.

Example (sync):
    from troopai.adk import Agent, Runner

    agent = Agent(
        name="Assistant",
        system_prompt="You are a helpful assistant.",
    )

    result = Runner.run(agent, "Hello!")
    logger.info(result.final_output)

Example (async):
    result = await Runner.arun(agent, "Hello!")
    logger.info(result.final_output)

Example (streaming):
    from troopai.adk import Agent, Runner, RunItemType

    result = Runner.run(agent, "Write a story", stream=True)

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            logger.info(event.data)
        elif event.type == "run_item_stream_event":
            if event.name == RunItemType.TOOL_CALLED:
                logger.info(f"\\nCalling: {event.item['name']}")

    logger.info(f"\\nFinal: {result.final_output}")

Example (with LLMConfig):
    from troopai.adk import Agent, Runner, LLMConfig

    agent = Agent(
        name="Creative Writer",
        system_prompt="Write creative stories.",
        llm_config=LLMConfig(temperature=0.9, max_output_tokens=2000),
    )
"""

import importlib as _importlib
import logging as _logging
import logging.config as _logging_config
from pathlib import Path as _Path

# 1. Library best practice: NullHandler prevents "No handler found" warnings
#    during the window before config loads (or if config never loads).
_logger = _logging.getLogger("troopai.adk")
_logger.addHandler(_logging.NullHandler())


def setup_logging() -> None:
    """Load logging from the YAML config when present.

    The YAML config at configs/logging/default_logger.yaml (project root)
    routes records to a rotating ``.log`` file only — console display is
    disabled by default so the terminal belongs to the ``VerboseConfig``
    event stream (see :mod:`troopai.adk.verbose`). When the config is
    absent or unloadable, the package stays silent behind the
    ``NullHandler`` — standard library practice; the application owns
    handler installation.

    Users can reconfigure after import:
        logging.getLogger("troopai.adk").setLevel(logging.WARNING)
    """
    # src/troopai/adk/__init__.py → 4 levels up to project root
    _project_root = _Path(__file__).parent.parent.parent.parent
    config_path = _project_root / "configs" / "logging" / "default_logger.yaml"

    if config_path.exists():
        try:
            yaml = _importlib.import_module("yaml")

            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Resolve file handler path relative to project root (not CWD)
            file_cfg = config.get("handlers", {}).get("file", {})
            if "filename" in file_cfg:
                log_dir = _project_root / _Path(file_cfg["filename"]).parent
                log_dir.mkdir(parents=True, exist_ok=True)

                max_bytes = file_cfg.get("maxBytes", 104857600)

                # Reuse the most recent log file under the size limit
                existing_logs = sorted(
                    log_dir.glob("troopai_adk.*.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                reusable = None
                for log_file in existing_logs:
                    if log_file.stat().st_size < max_bytes:
                        reusable = log_file
                        break

                if reusable is not None:
                    file_cfg["filename"] = str(reusable)
                else:
                    from datetime import datetime as _dt

                    timestamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_cfg["filename"] = str(log_dir / f"troopai_adk.{timestamp}.log")

            _logging_config.dictConfig(config)
            return
        except ImportError:
            pass  # pyyaml not installed — NullHandler keeps the package silent
        except Exception as exc:
            _logger.warning("Failed to load logging config %s: %s", config_path, exc)


from troopai.adk.agents import (
    Agent,
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentGuardrailSeverity,
    AgentGuardrailTimeoutInfo,
    AgentInputGuardrail,
    AgentOutputGuardrail,
    AgentTimeoutPolicy,
    Middleware,
    agent_input_guardrail,
    agent_output_guardrail,
)
from troopai.adk.context import CacheStrategy, ContextManagementConfig
from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    AgentOutputGuardrailTripwireTriggered,
    GuardrailTripwireTriggered,
    MaxTurnsExceeded,
    ToolGuardrailTripwireTriggered,
    TroopAIError,
    UserError,
)
from troopai.adk.flows import (
    FLOW_ERROR_TRIGGER,
    Flow,
    FlowCheckpoint,
    FlowConfig,
    FlowDefinitionError,
    FlowExecutable,
    FlowMaxStepsExceeded,
    FlowRunResult,
    FlowRunResultStreaming,
    FlowRunStatus,
    FlowStep,
    FlowStepError,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.llms import LLMConfig
from troopai.adk.memory import (
    Memory,
    MemoryConfig,
    SQLiteMemory,
    TemporaryMemory,
)
from troopai.adk.run import (
    AgentRunner,
    AgentUpdatedStreamEvent,
    CancelMode,
    FlowRunner,
    GraphRunner,
    RawResponseStreamEvent,
    RunConfig,
    RunContext,
    RunHooks,
    RunItemStreamEvent,
    RunItemType,
    Runner,
    RunnerProfile,
    RunResultStreaming,
    StreamEvent,
    SwarmRunner,
    TaskGroupRunner,
    TaskPipelineRunner,
    TaskRunner,
)
from troopai.adk.session import Session, SessionSettings, SQLiteMultiSessions
from troopai.adk.skills import (
    Skill,
    SkillActivation,
    SkillDiscoveryToolset,
    SkillGovernance,
    SkillMetadata,
    SkillSet,
)
from troopai.adk.tasks import (
    ErrorPolicy,
    Task,
    TaskDependency,
    TaskGroup,
    TaskGroupResult,
    TaskInputData,
    TaskInputFilter,
    TaskOutput,
    TaskPipeline,
    TaskPipelineDefinitionError,
    TaskPipelineResult,
    TaskPipelineState,
)
from troopai.adk.tools import DocumentSearchTool, FunctionTool, MemoryTool
from troopai.adk.types.run import RunResult
from troopai.adk.verbose import EventStyle, VerboseConfig

__version__ = "0.1.0"

__all__ = [
    # Flow error-trigger route literal (error_policy="route_to_error_handler")
    "FLOW_ERROR_TRIGGER",
    # Core classes
    "Agent",
    # Guardrails
    "AgentGuardrailFunctionOutput",
    "AgentGuardrailSeverity",
    "AgentGuardrailTimeoutInfo",
    # Per-Agent guardrail config
    "AgentGuardrails",
    "AgentInputGuardrail",
    "AgentInputGuardrailTripwireTriggered",
    "AgentOutputGuardrail",
    "AgentOutputGuardrailTripwireTriggered",
    "AgentRunner",
    "AgentTimeoutPolicy",
    "AgentUpdatedStreamEvent",
    "CacheStrategy",
    "CancelMode",
    "ContextManagementConfig",
    # Document search (RAG built-in tool)
    "DocumentSearchTool",
    # TaskGroup error policy ("collect_all" / "halt_on_first")
    "ErrorPolicy",
    "EventStyle",
    # Flow primitive — decorator-driven multi-step orchestration over typed state
    "Flow",
    "FlowCheckpoint",
    "FlowConfig",
    "FlowDefinitionError",
    "FlowExecutable",
    "FlowMaxStepsExceeded",
    "FlowRunResult",
    "FlowRunResultStreaming",
    "FlowRunStatus",
    "FlowRunner",
    "FlowStep",
    "FlowStepError",
    "FunctionTool",
    "GraphRunner",
    "GuardrailTripwireTriggered",
    "LLMConfig",
    "MaxTurnsExceeded",
    # Memory
    "Memory",
    "MemoryConfig",
    "MemoryTool",
    # Middleware config
    "Middleware",
    "RawResponseStreamEvent",
    "RunConfig",
    "RunContext",
    "RunHooks",
    "RunItemStreamEvent",
    "RunItemType",
    "RunResult",
    # Streaming
    "RunResultStreaming",
    # Runner and execution
    "Runner",
    "RunnerProfile",
    "SQLiteMemory",
    "SQLiteMultiSessions",
    # Session & context management
    "Session",
    "SessionSettings",
    # Skills
    "Skill",
    "SkillActivation",
    "SkillDiscoveryToolset",
    "SkillGovernance",
    "SkillMetadata",
    "SkillSet",
    "StreamEvent",
    "SwarmRunner",
    # Tasks
    "Task",
    "TaskDependency",
    "TaskGroup",
    "TaskGroupResult",
    "TaskGroupRunner",
    "TaskInputData",
    "TaskInputFilter",
    "TaskOutput",
    "TaskPipeline",
    "TaskPipelineDefinitionError",
    "TaskPipelineResult",
    "TaskPipelineRunner",
    "TaskPipelineState",
    "TaskRunner",
    "TemporaryMemory",
    "ToolGuardrailTripwireTriggered",
    # Exceptions
    "TroopAIError",
    "UserError",
    # Verbose output
    "VerboseConfig",
    # Version
    "__version__",
    "agent_input_guardrail",
    "agent_output_guardrail",
    # Flow decorators
    "flow_listen",
    "flow_router",
    "flow_start",
    "setup_logging",
]
