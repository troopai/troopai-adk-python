# System Prompt Usage

## SystemPrompt Construction

```python
from troopai.adk.prompts import SystemPrompt, SystemPromptTone

prompt = SystemPrompt(
    role="You are a senior Python code reviewer specializing in security.",
    context="You work at a fintech company. Code must comply with PCI-DSS.",
    guidelines=["Flag security vulnerabilities immediately", "Always suggest type hints"],
    tone=SystemPromptTone.TECHNICAL,
    constraints=["Never execute code", "Ask for full file if snippet is incomplete"],
    output_format="Use Markdown with headers for each finding.",
)

# Renders to a single string with ## headers for each section
print(prompt.generate())
```

## DynamicSystemPrompt

Type alias for callables that receive a `DynamicSystemPromptData` bundle, returning a system prompt dynamically:

```python
DynamicSystemPrompt = Callable[[DynamicSystemPromptData], MaybeAwaitable[Union[str, SystemPrompt]]]
```

The callable receives a single `DynamicSystemPromptData` with:
- `data.context: RunContext` -- the execution context (carries user-provided context and usage metrics)
- `data.agent: Agent` -- the agent instance being run

Supports sync and async callables:

```python
from troopai.adk.prompts.system_prompt import DynamicSystemPromptData
from troopai.adk.prompts import SystemPrompt

# Sync -- adapt prompt based on user context
def get_prompt(data: DynamicSystemPromptData) -> SystemPrompt:
    return SystemPrompt(
        role=f"You are {data.agent.name}.",
        context=data.context.context.get("tenant_guidelines", ""),
    )

# Async -- fetch context at runtime
async def get_prompt(data: DynamicSystemPromptData) -> SystemPrompt:
    guidelines = await fetch_guidelines(data.context.context["tenant_id"])
    return SystemPrompt(role=f"You are {data.agent.name}.", knowledge=guidelines)
```

## Agent Integration

The `Agent.system_prompt` field accepts `str`, `SystemPrompt`, or `DynamicSystemPrompt`:

```python
from troopai.adk.agents import Agent
from troopai.adk.prompts import SystemPrompt

# Plain string
agent = Agent(name="Bot", system_prompt="You are helpful.")

# Structured prompt
agent = Agent(name="Bot", system_prompt=SystemPrompt(role="You are helpful."))

# Dynamic callable (receives DynamicSystemPromptData at resolution time)
def my_prompt(data):
    return SystemPrompt(role="You are helpful.", context=data.context.context.get("extra", ""))

agent = Agent(name="Bot", system_prompt=my_prompt)
```

The `Runner` resolves the prompt via `_resolve_system_prompt(agent, ctx_wrapper)` before building LLM messages.
The callable is invoked with `DynamicSystemPromptData(context=ctx_wrapper, agent=agent)`.
