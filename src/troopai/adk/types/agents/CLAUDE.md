# Agent Types

Types for the `Agent.as_tool()` feature (LLM-orchestrated delegation).

## Files

- `agent_as_tool_types.py` - `AgentToolInput`, `AgentToolOutputExtractor`, `AgentToolInputBuilder`

## Types

### `AgentToolInput`

Default Pydantic input schema when no custom schema is provided to `as_tool()`. Single `input: str` field that the LLM fills with the task description. See `docs/agents/as_tool.md` and `examples/agent_patterns/` for usage.

### `AgentToolOutputExtractor`

Callback type: `Callable[[RunResult], MaybeAwaitable[str]]`

Customises what the parent LLM sees from the sub-agent's result.

### `AgentToolInputBuilder`

Callback type: `Callable[[Any, ToolContext], MaybeAwaitable[Union[str, list[Any]]]]`

Transforms parsed tool input into `Agent.run(user_prompt=...)` value. Used with custom `input_schema`.
