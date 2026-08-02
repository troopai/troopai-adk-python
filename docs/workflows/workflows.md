# Durable Workflows

The `workflows` package makes any TroopAI `Agent`, `Swarm`, `Graph`, or `Flow`
run durably inside **Temporal** or **Restate**.  The bridge intercepts LLM calls
and tool calls at their boundaries; everything else — the runner, agent loop,
tools — stays unchanged.

Install the matching optional extra:

```bash
pip install "troopai-adk-python[temporal]"   # Temporal backend
pip install "troopai-adk-python[restate]"    # Restate backend
```

---

## Quick start: Temporal

```python
from temporalio.client import Client
from temporalio.worker import Worker

from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM
from troopai.adk.run import Runner
from troopai.adk.workflows.engine import ModelActivityConfig
from troopai.adk.workflows.temporal import (
    TroopAITemporalPlugin,
    TroopAIWorkflow,
    TemporalLLM,
)


# 1. Build the agent with a TemporalLLM shim
base_llm = LiteLLM(model="gpt-4o")
temporal_llm = TemporalLLM(
    wrapped=base_llm,
    activity_config=ModelActivityConfig(),
)
agent = Agent(name="assistant", llm=temporal_llm, instructions="You are helpful.")


# 2. Define the workflow
from temporalio import workflow


@workflow.defn
class AssistantWorkflow(TroopAIWorkflow):
    @workflow.run
    async def run(self, prompt: str) -> str:
        runner = Runner()
        result = await runner.arun(agent, prompt)
        return result.output


# 3. Wire the worker
plugin = TroopAITemporalPlugin()
plugin.register_model("gpt-4o", base_llm)

client = await Client.connect("localhost:7233")
async with Worker(
    client,
    task_queue="agent-queue",
    workflows=[AssistantWorkflow],
    activities=[],          # activity list from plugin.build_worker_kwargs()
    **plugin.build_worker_kwargs(),
):
    result = await client.execute_workflow(
        AssistantWorkflow.run,
        "Tell me a joke.",
        id="run-1",
        task_queue="agent-queue",
    )
    print(result)
```

`TemporalLLM.install(agent)` is a convenience method that installs
`TemporalLLM` on every agent in the handoff graph at once:

```python
TemporalLLM.install(agent, activity_config=ModelActivityConfig())
```

---

## Quick start: Restate

```python
import restate
from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM
from troopai.adk.run import Runner
from troopai.adk.workflows.engine import ModelActivityConfig
from troopai.adk.workflows.restate import TroopAIRestateService, RestateLLM


base_llm = LiteLLM(model="gpt-4o")
agent = Agent(
    name="assistant",
    llm=RestateLLM(wrapped=base_llm, activity_config=ModelActivityConfig()),
    instructions="You are helpful.",
)


@restate.service
class AgentService(TroopAIRestateService):
    @restate.handler
    async def run(self, ctx: restate.Context, prompt: str) -> str:
        runner = Runner()
        result = await runner.arun(agent, prompt)
        return result.output
```

`RestateLLM` detects whether it is inside a Restate handler via
`restate.current_context()`.  Outside a handler the wrapped LLM is called
directly — no overhead in tests or CLI invocations.

---

## Tool wrapping

### `activity_tool()` — make a tool durable

Promote any `@activity.defn`-decorated async function into a
`FunctionTool` whose invocation is routed through `execute_activity` inside
a workflow:

```python
from datetime import timedelta
from temporalio import activity
from troopai.adk.workflows.temporal import activity_tool


@activity.defn
async def fetch_weather(city: str) -> str:
    """Fetch current weather for *city*."""
    # ... real HTTP call
    return f"Sunny in {city}"


weather_tool = activity_tool(
    fetch_weather,
    start_to_close_timeout=timedelta(seconds=15),
    maximum_attempts=3,
)

agent = Agent(name="weather", llm=temporal_llm, tools=[weather_tool])
```

Outside a workflow (tests, CLI) the tool calls `fetch_weather` directly.

### `TemporalToolWrapper` — selective per-tool config

`TemporalToolWrapper` lets you override timeout and retry for individual tools
or opt specific tools out of activity wrapping entirely:

```python
from troopai.adk.workflows.temporal import TemporalToolWrapper
from troopai.adk.workflows.engine import ToolActivityConfig

wrapper = TemporalToolWrapper(
    tool_configs={
        "fast_lookup": False,                         # keep in-workflow
        "expensive_api": ToolActivityConfig(
            start_to_close_timeout=120,
            maximum_attempts=3,
        ),
    },
)

for tool in agent.tools:
    if wrapper.should_wrap(tool.name):
        cfg = wrapper.get_config(tool.name)
        # rebuild tool with cfg ...
```

### `restate_tool()` — Restate equivalent

```python
from troopai.adk.workflows.restate import restate_tool

async def fetch_weather(city: str) -> str:
    return f"Sunny in {city}"

durable_weather = restate_tool(fetch_weather, name="fetch_weather")
```

---

## Human-in-the-loop (HITL)

### Temporal signals, queries, and updates

`TroopAIWorkflow` pre-wires three HITL primitives:

| Primitive | Method | Use |
|---|---|---|
| Signal | `send_human_reply(HumanReply)` | Human posts a reply to an interrupted node |
| Query | `get_state()` | Read the current workflow state snapshot |
| Update | `approve_tool_call(ToolApprovalDecision)` | Approve or reject a deferred tool call |

**Interrupt to resume cycle** (Temporal):

```python
from temporalio import workflow
from troopai.adk.workflows.temporal import (
    HumanReply,
    TroopAIWorkflow,
)


@workflow.defn
class ReviewWorkflow(TroopAIWorkflow):
    @workflow.run
    async def run(self, prompt: str) -> str:
        self.update_state({"status": "awaiting_human"})

        # Block until a human reply arrives
        await workflow.wait_condition(lambda: len(self._pending_replies) > 0)
        replies = self.consume_replies()

        self.update_state({"status": "running", "human_reply": replies[0].value})
        runner = Runner()
        result = await runner.arun(agent, replies[0].value)
        return result.output
```

Send a reply from the client side:

```python
handle = client.get_workflow_handle("run-1")
await handle.signal(
    ReviewWorkflow.send_human_reply,
    HumanReply(node_id="root", value="proceed"),
)
```

Approve or reject a deferred tool call:

```python
from troopai.adk.workflows.temporal import ToolApprovalDecision

await handle.execute_update(
    ReviewWorkflow.approve_tool_call,
    ToolApprovalDecision(call_id="call-abc", approved=True),
)
```

### Restate HITL via durable promises

`TroopAIRestateService.wait_for_human_reply` blocks durably until an external
actor resolves the named promise:

```python
@restate.service
class ReviewService(TroopAIRestateService):
    @restate.handler
    async def run(self, ctx: restate.Context, prompt: str) -> str:
        reply = await self.wait_for_human_reply(ctx, promise_name="approval")
        runner = Runner()
        result = await runner.arun(agent, reply.value)
        return result.output
```

---

## Streaming with `TemporalStreamingLLM`

`TemporalStreamingLLM` extends `TemporalLLM` with `acomplete_streamed`.
Outside a workflow it delegates directly to the wrapped LLM's native streaming
path.  Inside a workflow the activity executes non-streaming and surfaces the
complete response as a single `"done"` event:

```python
from troopai.adk.workflows.temporal import TemporalStreamingLLM

llm = TemporalStreamingLLM(
    wrapped=LiteLLM(model="gpt-4o"),
    activity_config=ModelActivityConfig(),
)

async for event in await llm.acomplete_streamed(messages="Hello!"):
    if event.type == "done":
        print(event.response)
```

---

## MCP tools over Temporal activities

`TemporalMCPToolSet` routes MCP list-tools and call-tool operations through
named Temporal activities, making MCP I/O durable and tracked in the event
history:

```python
from troopai.adk.workflows.temporal import TemporalMCPToolSet

toolset = TemporalMCPToolSet(
    name="search-server",
    connection_params={"url": "http://localhost:3000"},
    start_to_close_timeout=30,
)

# Inside a workflow:
tools = await toolset.list_tools_in_workflow()
result = await toolset.call_tool_in_workflow("web_search", {"query": "Temporal Python"})
```

The activity names are `"{name}-mcp-list-tools"` and `"{name}-mcp-call-tool"`;
register matching activity functions on the worker side.

---

## Checkpointers: when to use vs when Temporal replaces them

| Scenario | Recommendation |
|---|---|
| Agent-as-workflow (Temporal) | Temporal event history IS the durable state — no `GraphCheckpointer` needed |
| Graph inside Temporal workflow | Temporal handles crash recovery; `GraphCheckpointer` composable for mid-superstep snapshots |
| Graph without Temporal | `SQLiteCheckpointer` or `InMemoryCheckpointer` from `graphs/checkpointers/` |
| Restate | `ctx.run()` journals each step result; no separate checkpointer needed |

If you attach a `GraphCheckpointer` inside a Temporal workflow, make all
writes idempotent (upsert rather than insert) because Temporal replay may
trigger the `on_graph_end` hook more than once.

---

## Replay-safe tracing helpers

```python
from troopai.adk.workflows.temporal import (
    deterministic_timestamp,
    deterministic_uuid,
    should_emit_span,
)

# Inside a workflow these use Temporal's deterministic clock / PRNG.
# Outside a workflow they fall back to time.time() / uuid.uuid4().
ts = deterministic_timestamp()
uid = deterministic_uuid()

if should_emit_span():
    # Emit OpenTelemetry span — suppressed automatically during replay
    ...
```
