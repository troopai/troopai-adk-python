# Multi-Agent Patterns

The TroopAI Agents ADK provides three primitives for building multi-agent systems. Each primitive makes a different trade-off between control retention, context sharing, token cost, and orchestration flexibility.

| Pattern | Who Decides | Control After | History Shared |
|---------|-------------|---------------|----------------|
| Agent-as-Tool (`as_tool()`) | Parent LLM | Returned to parent | None — input string only |
| Handoff (LLM-orchestrated) | LLM | Transferred permanently | Filtered via `HandoffConfig` |
| Handoff (code-orchestrated) | Application code | Transferred permanently | Filtered via `HandoffConfig` |
| Parallel execution | Application code | Collected by aggregator | None — each run is isolated |

---

## Pattern 1: Agent-as-Tool

`agent.as_tool()` wraps a sub-agent as a standard `FunctionTool`. From the parent LLM's perspective, the sub-agent is indistinguishable from any other tool call. Control stays with the parent throughout.

### Basic usage

```python
from troopai.adk.agents import Agent
from troopai.adk.run import Runner

researcher = Agent(
    name="Researcher",
    system_prompt="You are a research specialist. Given a topic, summarize key findings.",
)

writer = Agent(
    name="Writer",
    system_prompt="You are a writing specialist. Produce polished prose from provided content.",
)

supervisor = Agent(
    name="Supervisor",
    system_prompt=(
        "Coordinate a research-and-writing pipeline.\n"
        "1. Use the researcher tool to gather information.\n"
        "2. Use the writer tool to produce a polished article.\n"
        "Synthesize the results into a final answer."
    ),
    tools=[
        researcher.as_tool(tool_description="Research a topic and return key findings."),
        writer.as_tool(tool_description="Write polished prose from provided content."),
    ],
)

result = await Runner.arun(supervisor, "Write a short article about open-source software.")
```

### Governance parameters

All parameters are keyword-only on `agent.as_tool()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tool_name` | `str` | Name the parent LLM sees. Defaults to `snake_case(agent.name)`. |
| `tool_description` | `str` | Description for the LLM. Defaults to `"Delegate a task to the {name} agent."` |
| `input_schema` | `type[BaseModel]` | Pydantic model for structured input. Default: single `input: str` field. |
| `input_builder` | `Callable` | Transform the parsed Pydantic instance into the string passed to the sub-agent. |
| `extractor` | `Callable` | Post-process `RunResult` before the parent LLM sees it. Defaults to `str(result.final_output)`. |
| `on_stream` | `Callable` | Callback receiving each `StreamEvent` from the sub-agent in real time. |
| `max_turns` | `int` | Maximum agent loop turns for the sub-agent. Default: `10`. |
| `timeout` | `float` | Seconds before aborting via `asyncio.wait_for()`. On timeout, an error string is returned — execution does not raise. |
| `budget` | `LLMUsageLimits` | Token and request limits for the sub-agent run. Merged into the sub-agent's `RunConfig`. |
| `requires_approval` | `bool \| Callable` | HITL gate — sub-agent run does not start until approved. |
| `run_config` | `RunConfig` | Explicit `RunConfig` for the sub-agent. Inherits from the parent's `ToolContext.run_config` when `None`. |

### Structured input with `input_schema` and `input_builder`

When you need the LLM to provide structured parameters instead of a plain string, supply a custom Pydantic schema:

```python
from pydantic import BaseModel, Field
from troopai.adk.tools.tool_context import ToolContext

class TranslationInput(BaseModel):
    text: str = Field(description="Text to translate.")
    target_language: str = Field(description="Target language, e.g. 'French'.")
    formality: str = Field(default="neutral", description="Tone: formal, informal, or neutral.")

def build_input(parsed: TranslationInput, ctx: ToolContext) -> str:
    return (
        f"Translate the following to {parsed.target_language} "
        f"using a {parsed.formality} tone:\n\n{parsed.text}"
    )

translator_tool = translator_agent.as_tool(
    tool_name="translate",
    input_schema=TranslationInput,
    input_builder=build_input,
)
```

### Custom output with `extractor`

By default, the parent LLM receives `str(result.final_output)`. Use `extractor` to control exactly what is returned:

```python
from troopai.adk.types.run import RunResult

def extract_summary(result: RunResult) -> str:
    output = str(result.final_output)
    if len(output) > 2_000:
        output = output[:2_000] + "\n... [truncated]"
    return f"[From {result.last_agent.name}]\n{output}"

analyst_tool = analyst_agent.as_tool(
    tool_name="analyze",
    extractor=extract_summary,
)
```

### Introspection

`agent.get_delegate_tools()` returns all `FunctionTool` instances on an agent that wrap sub-agents. `agent.get_agent_graph()` recursively builds the full topology, including delegate tools and handoff targets, as a nested dict. Both methods handle cycles.

```python
graph = supervisor.get_agent_graph()
# {
#     "name": "Supervisor",
#     "delegates": [
#         {"name": "Researcher", "delegates": [], "handoffs": []},
#         {"name": "Writer",     "delegates": [], "handoffs": []},
#     ],
#     "handoffs": [],
# }
```

---

## Context Isolation Explained

This is the most commonly misunderstood aspect of `as_tool()`. Many developers assume that intermediate results from a sub-agent — its LLM turns, tool calls, and reasoning — accumulate in the parent's context. They do not.

When the parent LLM calls a sub-agent tool, the following sequence occurs:

**Step 1.** The parent LLM produces a tool call in its output:

```json
{"name": "researcher", "arguments": "{\"input\": \"summarize open-source licensing\"}"}
```

**Step 2.** The Runner's tool executor calls `_on_invoke_tool`, which calls `Runner.arun(researcher_agent, user_prompt="summarize open-source licensing")`. This is a completely fresh run with its own message list. The sub-agent's message list starts from zero — it contains only the system prompt and the single user message derived from the tool input.

**Step 3.** The sub-agent executes its own agent loop. It may make multiple LLM calls, invoke its own tools, and produce extensive intermediate output. All of this lives in the sub-agent's isolated message list.

**Step 4.** The sub-agent reaches a final output. By default, `str(result.final_output)` is returned as the tool result string. If an `extractor` is provided, it processes `RunResult` and returns a custom string instead.

**Step 5.** The parent receives exactly two messages added to its context: the tool call message it generated, and the tool result string. The entire sub-agent execution is opaque to the parent.

```
Parent context before delegation:
  [system] [user] [assistant: calls researcher tool]

Sub-agent context (isolated, never seen by parent):
  [system] [user: "summarize open-source licensing"]
  [assistant: calls search_tool]
  [tool result: ...]
  [assistant: calls search_tool again]
  [tool result: ...]
  [assistant: final summary]

Parent context after delegation:
  [system] [user] [assistant: calls researcher tool] [tool result: "...final summary..."]
```

The sub-agent may consume thousands of tokens internally. The parent pays only for the tool call plus the result string — typically a few hundred tokens — regardless of how many turns the sub-agent took internally.

This isolation is not configurable. It is a structural property of how `as_tool()` works: the sub-agent runs in a separate `Runner.arun()` call with its own message list, and only the final output crosses the boundary.

---

## Pattern 2: Handoffs

Handoffs permanently transfer control from one agent to another. Unlike `as_tool()`, the originating agent does not regain control after the handoff. The target agent picks up the conversation and runs to completion.

The ADK provides two handoff strategies with different trade-offs for token cost and routing flexibility.

### LLM-orchestrated handoffs

The source agent's available tool list includes `transfer_to_<target_name>` function tools. The LLM decides when and to whom to hand off by calling the appropriate tool.

```python
from troopai.adk.agents import Agent
from troopai.adk.handoffs import Handoff

refunds_agent = Agent(
    name="Refunds Specialist",
    system_prompt="Handle refund requests. Ask for order ID if not provided.",
)

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt="Handle billing questions: invoices, charges, payment methods.",
)

triage_agent = Agent(
    name="Triage",
    system_prompt="Route customer requests to the appropriate specialist.",
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer when the user wants a refund or return.",
        ),
        Handoff(
            target=billing_agent,
            description="Transfer for billing questions, invoices, or charges.",
        ),
    ],
)
```

A bare `Agent` in the `handoffs` list is also valid and is auto-wrapped in a `Handoff` with default description.

**Typed handoff input.** When the LLM should provide structured context at handoff time, set `input_type` to a Pydantic model. The schema is auto-generated, and the validated instance is available in `HandoffInputData.intent` and in `on_handoff` callbacks.

```python
from pydantic import BaseModel

class EscalationInput(BaseModel):
    reason: str
    priority: int

Handoff(
    target=escalation_agent,
    input_type=EscalationInput,
    description="Escalate with reason and priority.",
)
```

### Code-orchestrated handoffs (HandoffRoute)

The source agent outputs a structured Intent type rather than calling a transfer tool. Application code routes the intent to a target agent deterministically. Zero routing tokens are spent — the LLM is not involved in the routing decision.

```python
from pydantic import Field
from typing import Literal, Optional, Union

from troopai.adk.agents import Agent
from troopai.adk.handoffs import HandoffRoute
from troopai.adk.types.intents import Intent, Respond

class RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: Optional[str] = Field(None, description="Order ID if mentioned.")

class BillingIntent(Intent):
    kind: Literal["billing"] = "billing"

class CancelIntent(Intent):
    kind: Literal["cancel"] = "cancel"

# Union type becomes the agent's output_schema
TriageOutput = Union[Respond, RefundIntent, BillingIntent, CancelIntent]

triage_agent = Agent(
    name="Triage",
    system_prompt="Classify the user's request. Respond directly only for simple questions.",
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("support")
        .when(RefundIntent).to(refunds_agent)
        .when(BillingIntent, CancelIntent).to(billing_agent)
        .otherwise(general_agent)
    ),
)
```

When the triage LLM outputs a `RefundIntent`, the Runner calls `HandoffRoute.resolve(intent)` in Python, which returns the target agent without any additional LLM call. When the LLM outputs `Respond`, `resolve()` returns `None` — no handoff occurs, and the agent's `message` field is returned as the final output.

The `.when()` method accepts multiple intent types that map to the same target. More specific intent types must be registered before their parent classes to prevent shadowing errors.

### HandoffConfig: controlling context transfer cost

By default, the full conversation history is forwarded to the target agent. `HandoffConfig` controls how much history transfers and at what cost.

```python
from troopai.adk.handoffs import Handoff, HandoffConfig

# Transfer only the last 10 messages, capped at 5,000 tokens
Handoff(
    target=specialist_agent,
    config=HandoffConfig(
        strategy="last_n",
        window=10,
        budget=5_000,
    ),
)

# Transfer only the classified intent — no prior conversation history
Handoff(
    target=specialist_agent,
    config=HandoffConfig(strategy="intent_only"),
)
```

| `strategy` | What transfers |
|------------|----------------|
| `"full"` (default) | Entire conversation history. |
| `"last_n"` | Last `window` messages only. |
| `"intent_only"` | Only the detected intent — no conversation messages. |
| `"summary"` | LLM-summarized version of the history. |

`budget` (token count) is applied after strategy selection. Messages exceeding the budget are compacted by `ContextCompactor`. `collapse=True` wraps all forwarded messages into a single system message instead of replaying them individually.

### Handoff input filters

Input filters transform `HandoffInputData` before it reaches the target agent. Built-in filters live in `troopai.adk.handoffs.handoff_filters`.

```python
from troopai.adk.handoffs import Handoff
from troopai.adk.handoffs.handoff_filters import remove_tool_calls, keep_last_n, compose

Handoff(
    target=specialist_agent,
    input_filter=compose(remove_tool_calls, keep_last_n(5)),
)
```

Available built-in filters: `forward_intent`, `remove_tool_calls`, `remove_system_messages`, `keep_last_n(n)`, `intent_only`, `compose(*filters)`.

`HandoffInputData` separates messages into temporal slices: `context` (messages before the current agent's turn) and `output` (messages generated during the current agent's turn). Filters set `forwarded` on the result without mutating `context` or `output`, preserving the full audit trail.

---

## Pattern 3: Parallel Execution

For fan-out workloads — generating multiple candidates, gathering perspectives from specialist agents, or running independent analyses — launch multiple `Runner.arun()` calls concurrently with `asyncio.gather()`.

```python
import asyncio
from pydantic import BaseModel, Field
from troopai.adk.agents import Agent
from troopai.adk.run import Runner

class JudgeVerdict(BaseModel):
    best_index: int = Field(ge=0, lt=3, description="Index of the best result (0-2).")
    reasoning: str = Field(description="Why this result was chosen.")

judge = Agent(
    name="Judge",
    system_prompt="Evaluate the three outputs and select the best one.",
    output_schema=JudgeVerdict,
)

async def run_parallel(prompt: str) -> str:
    # All three agents run concurrently — each in an isolated context
    results = await asyncio.gather(
        Runner.arun(agent_a, prompt),
        Runner.arun(agent_b, prompt),
        Runner.arun(agent_c, prompt),
    )

    outputs = [r.final_output for r in results]

    comparison = "\n\n".join(
        f"Output {i}:\n{outputs[i]}" for i in range(len(outputs))
    )
    judge_result = await Runner.arun(judge, comparison)
    verdict = judge_result.final_output_as(JudgeVerdict)
    return str(outputs[verdict.best_index])
```

Each parallel agent run is completely isolated: separate message lists, separate `RunContext` objects, separate token usage tracking. The outputs are plain strings (or structured objects) that the aggregator agent receives as part of its input prompt.

`asyncio.gather()` preserves result order, so `results[0]` always corresponds to the first agent argument.

---

## Cost Comparison

Understanding token cost per pattern helps you make the right architectural choice.

### Agent-as-Tool

The parent LLM pays for its own context plus, per delegation:
- Tokens to generate the tool call (small — JSON arguments for the task description)
- Tokens consumed by the result string returned from the sub-agent

The sub-agent's internal costs are separate. If you run many delegations with the same parent, the parent's context grows by approximately `(tool_call_tokens + result_tokens)` per delegation. Use `max_result_tokens` on `as_tool()` to cap result size.

Example: a supervisor that calls three sub-agents in sequence. The parent's context accumulates the results of all three tool calls. Sub-agent A's 50-turn internal loop is invisible to the parent.

### Handoffs

The target agent inherits conversation history from the source agent. Without `HandoffConfig`, it inherits the full history — which can be expensive when the source agent ran for many turns.

Token cost grows with: the source agent's conversation length, the number of handoff hops, and whether intermediate agents used many tool calls (which inflate the history).

Use `HandoffConfig` to bound this cost:
- `strategy="last_n"` with a small `window` keeps recent context only
- `budget=N` compacts anything over N tokens before transfer
- `strategy="intent_only"` eliminates conversation history entirely — best for triage flows where the specialist only needs the classified intent

For a triage flow where the source agent runs a single turn to classify the request, the conversation history passed to the target is small regardless of `HandoffConfig`. The cost concern becomes significant in multi-turn source agents.

Code-orchestrated handoffs add a routing cost advantage: the triage agent's output schema is a union of Intent types rather than a free-form response, and no `transfer_to_*` tools are added to the tool list. This saves the tokens that would otherwise be spent on tool definitions.

### Parallel execution

Each parallel agent pays for its own independent run with no shared context. The aggregator agent pays for its input prompt, which grows linearly with the number of results being compared.

Total cost: `sum(cost_per_agent_i) + aggregator_input_cost`.

For best-of-N selection, the aggregator's input is proportional to the combined length of all N results. Using `extractor` or `max_result_tokens` on sub-agents (when running them via `as_tool()` in the aggregator) keeps aggregator input bounded.

---

## When to Use Which

### Use `as_tool()` when:

- The parent agent needs to **retain control** after the sub-agent completes.
- You have a **supervisor orchestrating multiple specialists** in a pipeline (research → write → review).
- You need **parallel delegation** from a single parent LLM — the parent can call multiple delegate tools in one turn or across several turns.
- Context isolation is valuable: sub-agent intermediate steps should not pollute the parent's context.
- You want governance: per-sub-agent `timeout`, `budget`, `requires_approval`, and `max_result_tokens`.

### Use LLM-orchestrated handoffs when:

- You want the **LLM to decide routing** based on intent signals in the conversation.
- You need **multi-turn conversations** where the specialist agent continues the dialogue with the user — handoffs let the specialist see the prior conversation and respond directly.
- The routing logic is fuzzy or depends on nuanced conversation context that is hard to encode as explicit rules.
- You have relatively few routing targets (fewer `transfer_to_*` tools means lower token overhead per LLM call).

### Use code-orchestrated handoffs (HandoffRoute) when:

- Routing is **deterministic and rule-based** — intent type determines destination without ambiguity.
- You want **zero routing tokens**: no `transfer_to_*` tools in the tool list, no tool call from the LLM to trigger the handoff.
- You are building **triage flows** where a classifier agent maps requests to specialists and then exits.
- You need **subclass-based routing** (e.g., `PaymentFailureIntent` is a subclass of `BillingIntent`) — `HandoffRoute` uses `isinstance` matching and enforces registration order to prevent shadowing.

### Use parallel execution when:

- Tasks are **independent** — no agent needs results from another to do its work.
- You want **multiple perspectives or candidates** evaluated by a judge (best-of-N, voting, ensemble).
- You are running **fan-out/fan-in**: distribute one prompt to N agents, collect results, aggregate.
- Latency matters more than total token cost — parallel agents complete in wall-clock time equal to the slowest agent.

### Combining patterns

The patterns compose naturally:

- A parent agent uses `as_tool()` to delegate to a specialist, and that specialist internally uses a `HandoffRoute` to route sub-tasks to its own set of agents.
- A parallel execution runs N variants of an `as_tool()` supervisor, and the results are evaluated by a judge.
- A code-orchestrated handoff transfers to a target agent that uses `as_tool()` to orchestrate its own sub-agents.

The key constraint is that handoffs are one-way permanent transfers. If the originating agent needs to see the outcome, use `as_tool()` instead.

---

## Reference

- Source: `src/troopai/adk/agents/agent.py` — `Agent.as_tool()`, `Agent.get_delegate_tools()`, `Agent.get_agent_graph()`
- Source: `src/troopai/adk/handoffs/handoff.py` — `Handoff`, `handoff()`
- Source: `src/troopai/adk/handoffs/handoff_route.py` — `HandoffRoute`, `handoff_route()`
- Source: `src/troopai/adk/handoffs/handoff_config.py` — `HandoffConfig`
- Source: `src/troopai/adk/handoffs/handoff_filters.py` — built-in filters
- Source: `src/troopai/adk/handoffs/handoff_input_data.py` — `HandoffInputData`
- Source: `src/troopai/adk/types/agents/agent_as_tool_types.py` — `AgentToolInput`, `AgentToolOutputExtractor`, `AgentToolInputBuilder`
- Source: `src/troopai/adk/types/intents/__init__.py` — `Intent`, `Respond`
- Examples: `./examples/agent_patterns/agents_as_tools.py`
- Examples: `./examples/agent_patterns/parallelization.py`
- Examples: `./examples/handoffs/llm_orchestrated.py`
- Examples: `./examples/handoffs/code_orchestrated.py`
- Examples: `./examples/handoffs/cost_optimized.py`
