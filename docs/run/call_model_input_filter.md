# `call_model_input_filter` — Pre-LLM Input Rewrite Hook

A Layer 1, context-aware hook that runs immediately before every LLM
call. The filter can rewrite, truncate, or inject messages based on the
current agent and the run context.

## Type signature

```python
from troopai.adk.run.config import (
    CallModelData,
    CallModelInputFilter,
    ModelInputData,
    RunConfig,
)

# ModelInputData wraps the Layer 1 messages list
@dataclass
class ModelInputData:
    input: list[LLMInputContentItem]

# CallModelData is the payload handed to the filter
@dataclass
class CallModelData(Generic[TContext]):
    model_data: ModelInputData
    agent: Agent[TContext]
    context: Optional[TContext]

# The filter is a sync or async callable
CallModelInputFilter: TypeAlias = Callable[
    [CallModelData[Any]],
    Union[ModelInputData, Awaitable[ModelInputData]],
]
```

The filter is set on `RunConfig.call_model_input_filter` and runs on
every turn — once per LLM call, across handoffs.

## When it runs

TroopAI has three pre-LLM hooks; they run in this order, each stage's
output flowing into the next:

| Hook                                   | Layer                              | Access to agent/context | Sync/async    | Runs        |
|----------------------------------------|------------------------------------|-------------------------|---------------|-------------|
| `RunConfig.history_processors`         | Layer 3 (`list[RunItem]`)          | No                      | Sync          | First       |
| `ContextManager.prepare_messages()`    | Layer 1 (`list[LLMInputContentItem]`) | Via `RunConfig`      | Async         | Second      |
| `RunConfig.call_model_input_filter`    | Layer 1 (`list[LLMInputContentItem]`) | Yes (agent + context) | Sync or async | **Third**   |

Immediately after the filter returns, the Runner fires
`hooks.on_llm_start(...)` and then calls `llm.acomplete(messages=...)`.
`on_llm_start` observers therefore see exactly what the model will see.

## Worked example — inject a per-request system note

```python
from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import (
    CallModelData,
    ModelInputData,
    RunConfig,
)
from troopai.adk.run.runner import Runner

def add_user_context(payload: CallModelData) -> ModelInputData:
    user_id = (payload.context or {}).get("user_id", "anonymous")
    new_input = list(payload.model_data.input)
    new_input.append({
        "role": "user",
        "content": f"[context] acting as user {user_id}",
    })
    return ModelInputData(input=new_input)

agent = Agent(name="assistant", system_prompt="You are helpful.")
config = RunConfig(call_model_input_filter=add_user_context)

result = await Runner.arun(
    agent,
    "What should I buy?",
    context={"user_id": "u-123"},
    run_config=config,
)
```

Every turn, immediately before `llm.acomplete(...)` is called, the
filter appends a `[context]` note derived from the run context. The
agent loop's own messages, context-management output, and history
processors all complete before the filter runs.

## Rules and invariants

- The filter receives a **shallow copy** of the messages list, so
  in-place mutation of `payload.model_data.input` cannot corrupt the
  pre-filter state. Prefer returning a new `ModelInputData`.
- The return value **must** be a `ModelInputData`. Returning anything
  else (including a bare list) raises `TypeError`.
- Exceptions from the filter propagate to the caller after a
  `logger.error` call with `exc_info=True` — the loop does not swallow
  them.
- Setting `call_model_input_filter=None` (the default) is a true
  pass-through: no copy is allocated and the messages list reference
  flows straight through to the LLM.
- The filter runs on **every** turn, including turns after a handoff.
- If you only need pure Layer 3 (`RunItem`) transforms without access
  to the agent or context, prefer `history_processors` — it avoids the
  Layer 3 ↔ Layer 1 round-trip that the processors themselves perform.

See `examples/agent_patterns/call_model_input_filter.py` for a
runnable example.
