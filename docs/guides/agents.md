(guides/agents)=

# 🤖 Agents

An `Agent` is **configuration** — name, instructions, tools, handoffs,
guardrails. The `Runner` executes; the `Agent` does not. This separation
means a single `Agent` instance can be reused across many concurrent runs
without coordination, and all execution policy (budgets, retries, streaming,
checkpoints) lives in one place on the `Runner`. See
[Architecture: The Runner](../architecture/runner.md) for the
full rationale.

## Anatomy of an `Agent`

`Agent` is a `@dataclass` defined in `src/troopai/adk/agents/agent.py`. All
fields have defaults except `name` and `system_prompt`.

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique identifier. Must be non-empty. |
| `description` | `str \| None` | `None` | One-line description used as the tool description when this agent is exposed via `as_tool()` or listed as a handoff target. |
| `system_prompt` | `str \| SystemPrompt \| DynamicSystemPrompt` | — | Instructions defining the agent's behaviour. Must be non-empty. |
| `tools` | `list[Tool \| Toolset]` | `[]` | All tools and toolsets the agent can use each turn. |
| `handoffs` | `HandoffRoute \| list[Agent \| Handoff] \| None` | `None` | Delegation targets — LLM-orchestrated list or code-orchestrated route. |
| `guardrails` | `AgentGuardrails` | `AgentGuardrails()` | Input and output guardrail config. |
| `llm` | `str \| LLM \| None` | `None` | Per-agent model override. `None` uses the `RunConfig` default. |
| `llm_config` | `LLMConfig \| None` | `None` | Temperature, `max_output_tokens`, `num_retries`, etc. |
| `middleware` | `Middleware` | `Middleware()` | Per-layer plumbing middleware (tools / agents / LLMs). |
| `tool_use_behavior` | `ToolUseBehavior` | `"run_llm_again"` | What happens after tool execution. |
| `output_schema` | `type \| AgentOutputSchemaBase \| None` | `None` | Structured-output schema; validated against each LLM response. |
| `skills` | `list[Skill]` | `[]` | Skills that contribute instructions and tools. |
| `skill_activation` | `SkillActivation` | `LAZY` | When skill instructions enter the system prompt. |
| `hooks` | `AgentHooks \| None` | `None` | Per-agent lifecycle callbacks. |
| `verbose` | `VerboseConfig \| None` | `None` | Per-agent verbose-output override. |

## Minimal working example

```python
import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.run import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


support_agent = Agent(
    name="Customer Support",
    system_prompt=(
        "You are a friendly customer support specialist. "
        "Answer questions clearly and concisely."
    ),
)


async def main() -> None:
    result = await Runner.arun(support_agent, "How do I reset my password?")
    logger.info("Final output: %s", result.final_output)


asyncio.run(main())
```

:::{note}
`Runner.arun()` is the only entry point. `Agent` has no `run()` or `arun()`
method — invoking execution via the agent class would break concurrent reuse
and scatter execution policy across instances.
:::

## Instructions: static vs dynamic

### Static instructions

Pass a plain `str` for straightforward system prompts:

```python
agent = Agent(
    name="Analyst",
    system_prompt="Analyse the supplied data and return a structured summary.",
)
```

For longer prompts with discrete sections, use `SystemPrompt`:

```python
from troopai.adk.prompts import SystemPrompt, SystemPromptTone

agent = Agent(
    name="Code Reviewer",
    system_prompt=SystemPrompt(
        role="You are a senior Python code reviewer specialising in security.",
        context="You work at a fintech company. Code must comply with PCI-DSS.",
        guidelines=[
            "Flag security vulnerabilities immediately.",
            "Always suggest type hints.",
        ],
        tone=SystemPromptTone.TECHNICAL,
        constraints=["Never execute code.", "Ask for the full file if the snippet is incomplete."],
        output_format="Use Markdown with a header for each finding.",
    ),
)
```

### Dynamic instructions

`DynamicSystemPrompt` is a callable that receives `DynamicSystemPromptData`
(containing `data.context: RunContext` and `data.agent: Agent`) and returns
a `str` or `SystemPrompt`. This lets instructions depend on runtime state —
per-tenant config, session data, feature flags.

```python
import logging
from troopai.adk.agents import Agent
from troopai.adk.prompts import DynamicSystemPrompt, DynamicSystemPromptData, SystemPrompt

logger = logging.getLogger(__name__)


def build_prompt(data: DynamicSystemPromptData) -> SystemPrompt:
    tenant_id = data.context.context.get("tenant_id", "default")
    guidelines = fetch_tenant_guidelines(tenant_id)  # your lookup
    logger.debug("Building prompt for tenant %s", tenant_id)
    return SystemPrompt(
        role="You are a compliance assistant.",
        knowledge=guidelines,
    )


compliance_agent = Agent(
    name="Compliance",
    system_prompt=build_prompt,
)
```

Async callables are also accepted:

```python
async def build_prompt_async(data: DynamicSystemPromptData) -> str:
    policy = await fetch_policy_async(data.context.context["org"])
    return f"Follow this policy:\n{policy}"
```

:::{tip}
Dynamic instructions are resolved once per agent turn, just before the LLM
call. The callable is invoked with the current `RunContext`, so it sees any
context values you passed to `Runner.arun(..., context=...)`.
:::

## Tools

The `tools` field accepts any mix of `Tool` and `Toolset` instances:

- **`FunctionTool`** — wraps a Python callable decorated with `@function_tool`.
- **`Toolset` variants** — `FunctionToolset`, `PrefixedToolset`, `FilteredToolset`,
  `MCPToolset` — live collections that materialise per turn. Useful for MCP
  servers and dynamic tool lists without name collisions.

```python
from troopai.adk.tools import function_tool

@function_tool(name="lookup", description="Look up a support article by keyword.")
def lookup(keyword: str) -> str:
    return search_knowledge_base(keyword)  # your implementation


agent = Agent(
    name="Support",
    system_prompt="Help users by looking up articles.",
    tools=[lookup],
)
```

For full coverage of tool types, hosted tools (web search, code execution),
toolsets, tool governance (`requires_approval`, `rate_limit`,
`schema_enforcement`), and tool-level guardrails, see
[Tools guide](tools.md).

## Handoffs

`handoffs` specifies how this agent can permanently transfer control to
another agent:

- **LLM-orchestrated** — pass a `list[Agent | Handoff]`. The framework adds
  `transfer_to_<name>` tools to the LLM's tool list. The LLM decides when
  and to whom to hand off.
- **Code-orchestrated** — pass a `HandoffRoute`. The agent outputs a
  structured `Intent` type; application code routes deterministically with
  zero routing tokens.

```python
from troopai.adk.agents import Agent
from troopai.adk.handoffs import Handoff

# LLM-orchestrated
triage = Agent(
    name="Triage",
    system_prompt="Route customer requests to the right specialist.",
    handoffs=[
        Handoff(target=refunds_agent, description="Handle refund requests."),
        Handoff(target=billing_agent, description="Handle billing questions."),
    ],
)
```

For history transfer cost control (`HandoffConfig`), code-orchestrated
routing (`HandoffRoute`), typed handoff input, and input filters, see
[Handoffs guide](handoffs.md).

## Guardrails

`AgentGuardrails` holds two phase-typed lists:

- `guardrails.input` — validates input before (or in parallel with) the
  agent. A fired tripwire raises `AgentInputGuardrailTripwireTriggered`.
- `guardrails.output` — validates the agent's output. A fired tripwire raises
  `AgentOutputGuardrailTripwireTriggered`, or triggers re-prompting when
  `remediation` is set.

```python
from troopai.adk.agents import Agent, AgentGuardrails

agent = Agent(
    name="Support",
    system_prompt="Help customers.",
    guardrails=AgentGuardrails(
        input=[pii_guardrail],
        output=[content_policy_guardrail],
    ),
)
```

For decorator syntax (`@agent_input_guardrail`, `@agent_output_guardrail`),
severity levels, timeout policies, remediation loops, and the difference
between agent-level and run-level guardrails, see
[Guardrails guide](guardrails.md).

## `Agent.as_tool()` — agent-as-tool composition

`as_tool()` wraps a sub-agent as a `FunctionTool` that the parent LLM calls
like any other tool. Control stays with the parent throughout — unlike
handoffs, where control transfers permanently.

The sub-agent starts fresh: it sees only its own system prompt and the task
string provided in the tool input. The parent sees only the final output
string, not the sub-agent's internal turns.

```python
import logging
from troopai.adk.agents import Agent
from troopai.adk.run import Runner

logger = logging.getLogger(__name__)


researcher = Agent(
    name="Researcher",
    system_prompt="Summarise key findings on the given topic.",
)

writer = Agent(
    name="Writer",
    system_prompt="Produce polished prose from the supplied content.",
)

supervisor = Agent(
    name="Supervisor",
    system_prompt=(
        "Coordinate a research-and-writing pipeline.\n"
        "1. Use the researcher tool to gather information.\n"
        "2. Use the writer tool to produce a polished article."
    ),
    tools=[
        researcher.as_tool(tool_description="Research a topic and return key findings."),
        writer.as_tool(tool_description="Write polished prose from provided content."),
    ],
)
```

Key governance parameters on `as_tool()`:

| Parameter | Description |
|---|---|
| `tool_name` | Name the parent LLM sees. Defaults to `snake_case(agent.name)`. |
| `tool_description` | Description for the parent LLM. Defaults to `"Delegate a task to the {name} agent."` |
| `max_turns` | Maximum agent-loop turns for the sub-agent (default `10`). |
| `timeout` | Seconds before aborting via `asyncio.wait_for`. On timeout, an error string is returned to the parent — execution does not raise. |
| `budget` | `LLMUsageLimits` for the sub-agent run, merged into its `RunConfig`. |
| `max_result_tokens` | Truncates the result before the parent sees it. |
| `extractor` | Callback receiving the full `RunResult`; its return value becomes the tool output string. |
| `input_schema` | Custom Pydantic model for structured tool input (default: single `input: str` field). |
| `on_stream` | Callback receiving each `StreamEvent` from the sub-agent in real time. |

:::{tip}
Use `agent.get_agent_graph()` to introspect the full agent topology —
it returns a nested dict of delegate agents and handoff targets, handling
cycles. `agent.get_delegate_tools()` returns just the `FunctionTool` wrappers.
:::

## Non-mutating agent copies with `dataclasses.replace`

`Agent` is a standard `@dataclass`, so non-mutating copies with field
overrides use `dataclasses.replace()`. This is the correct pattern when you
need to vary a field per run without affecting other callers sharing the same
agent definition:

```python
import dataclasses
from troopai.adk.agents import Agent
from troopai.adk.llms import LLMConfig

base_agent = Agent(
    name="Analyst",
    system_prompt="Analyse the supplied data.",
    llm_config=LLMConfig(temperature=0.3),
)

# Creative variant — higher temperature, same everything else
creative = dataclasses.replace(base_agent, llm_config=LLMConfig(temperature=0.9))

# Per-tenant variant — different instructions, same tools
tenant_agent = dataclasses.replace(
    base_agent,
    system_prompt=f"Analyse data under {tenant_policy} constraints.",
)
```

Because `Agent` carries no mutable runtime state, the original `base_agent`
is unaffected by either replace call.

## Structured output with `output_schema`

When `output_schema` is set, the Runner instructs the LLM to return JSON
matching the schema and validates the response automatically.

```python
import asyncio
import logging
from pydantic import BaseModel, Field
from troopai.adk.agents import Agent
from troopai.adk.run import Runner

logger = logging.getLogger(__name__)


class SupportResponse(BaseModel):
    category: str = Field(description="Request category: billing, refund, or technical.")
    priority: int = Field(ge=1, le=5, description="Priority 1 (low) to 5 (critical).")
    response: str = Field(description="Agent response text.")


classifier = Agent(
    name="Support Classifier",
    system_prompt="Classify and respond to customer support requests.",
    output_schema=SupportResponse,
)


async def main() -> None:
    result = await Runner.arun(classifier, "My invoice is wrong.")
    parsed = result.final_output_as(SupportResponse)
    logger.info("Category: %s, Priority: %d", parsed.category, parsed.priority)


asyncio.run(main())
```

Pass a plain type (Pydantic `BaseModel`, `TypedDict`, `@dataclass`) directly
and the framework wraps it automatically. For advanced enforcement control,
pass an explicit `AgentOutputSchema` instance:

```python
from troopai.adk.schemas import AgentOutputSchema, SchemaEnforcement

agent = Agent(
    name="Support Classifier",
    system_prompt="...",
    output_schema=AgentOutputSchema(
        SupportResponse,
        schema_enforcement=SchemaEnforcement.NORMALIZED,
    ),
)
```

## Common patterns

### Triage with LLM-orchestrated handoffs

A front-line triage agent routes to specialists. Each specialist is a
separate `Agent`; the triage agent holds both as handoff targets:

```python
refunds_agent = Agent(
    name="Refunds",
    system_prompt="Process refund requests. Ask for order ID if not provided.",
)

billing_agent = Agent(
    name="Billing",
    system_prompt="Handle billing questions: invoices, charges, payment methods.",
)

triage_agent = Agent(
    name="Triage",
    system_prompt="Route customer requests to the right specialist.",
    handoffs=[refunds_agent, billing_agent],
)
```

For token-efficient routing (zero routing tokens), use `HandoffRoute` with
structured intent types. See [Handoffs guide](handoffs.md) for the full
pattern.

### Agent + tools combination

Combine `tools` and `handoffs` on the same agent. The LLM uses tools for
direct actions and handoff targets for specialist delegation:

```python
agent = Agent(
    name="Operations",
    system_prompt=(
        "Help users with account operations. "
        "Use tools for lookups; escalate complex billing issues."
    ),
    tools=[lookup_account, reset_password],
    handoffs=[billing_agent],
)
```

### Reusing an agent across runs

Because `Agent` carries no execution state, the same instance can be passed
to `Runner.arun()` concurrently:

```python
import asyncio
from troopai.adk.run import Runner

# Run the same agent with three different inputs in parallel
results = await asyncio.gather(
    Runner.arun(support_agent, "How do I reset my password?"),
    Runner.arun(support_agent, "Where is my invoice?"),
    Runner.arun(support_agent, "Cancel my subscription."),
)
```

Each call gets its own isolated message list and `RunContext`. No locking
or copying required.

## See also

- [Architecture: The Runner](../architecture/runner.md) — the execution loop,
  budgets, streaming, `RunConfig`, and the rationale for separating config from
  execution.
- [Tools guide](tools.md) — function tools, hosted tools, toolsets, and
  tool governance.
- [Handoffs guide](handoffs.md) — LLM-orchestrated and code-orchestrated
  routing, `HandoffConfig`, and context transfer cost control.
- [Guardrails guide](guardrails.md) — input/output guardrails, severity
  levels, timeout policies, and remediation loops.
