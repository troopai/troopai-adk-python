# Most used agentic patterns

This directory contains examples of the most commonly used agentic patterns. Each subdirectory includes a README file that explains the specific pattern, its use cases, and how to implement it using the provided code examples.

## Examples

| Example | Pattern |
|---------|---------|
| `agents_as_tools.py` | Sub-agents as regular tools via `as_tool()`, LLM orchestrates delegation |
| `agents_as_tools_streaming.py` | `on_stream` callback for real-time sub-agent event monitoring |
| `agents_as_tools_conditional.py` | `enabled` callbacks for context-dependent agent tool visibility |
| `deterministic.py` | Sequential pipeline with structured output gating between agents |
| `parallelization.py` | `asyncio.gather()` for concurrent agent execution with judge selection |
| `forcing_tool_use.py` | `LLMConfig.tool_choice="required"` + `ToolUseBehavior` modes |
| `llm_as_a_judge.py` | Iterative generate-evaluate-refine loop with structured feedback |
| `human_in_the_loop.py` | `requires_approval` on tools, `RunState.approve()/reject()` for resumption |
| `human_in_the_loop_custom_rejection.py` | Rejection messages that guide the LLM toward alternatives |
| `human_in_the_loop_stream.py` | Streaming combined with approval checkpoints |
| `nested_human_in_the_loop.py` | Approvals from sub-agents propagate transparently via `as_tool()` |
