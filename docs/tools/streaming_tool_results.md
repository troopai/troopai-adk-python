# Streaming Tool Results

A *streaming function tool* yields incremental progress events to
consumers of `Runner.arun(stream=True)` while the LLM still sees
exactly one tool-result message. Use this for long-running tools
where progress matters to a UI or operator but the LLM only needs
the final summary.

## When to use

- A tool that polls an API, streams a download, or runs a multi-step
  job whose progress matters to the consumer.
- A tool whose final result is small but intermediate state is
  expensive to recompute on a refresh.

## When NOT to use

- The LLM benefits from seeing chunks (use a normal tool that
  returns the whole stream as a string instead).
- The result is a single value that arrives quickly (added drain
  overhead is wasted).

## Author API

A streaming tool is an async generator that yields
`ToolStreamEvent` instances:

```python
from collections.abc import AsyncIterator

from troopai.adk.tools import function_tool
from troopai.adk.types.tools import ToolStreamEvent


@function_tool(
    name="search_documents",
    description="Search the corpus for documents matching a query.",
    streaming=True,
)
async def search_documents(query: str) -> AsyncIterator[ToolStreamEvent]:
    yield ToolStreamEvent(type="part_start", index=0)
    yield ToolStreamEvent(type="part_delta", delta=f"Searching '{query}'…")
    yield ToolStreamEvent(type="part_delta", delta=" scanning index…")
    yield ToolStreamEvent(type="part_end", index=0)
    yield ToolStreamEvent(
        type="done",
        response="Found 3 documents: doc_a, doc_b, doc_c.",
    )
```

The `"done"` event's `response` value is what the LLM sees as the
tool result. Anything yielded before the `"done"` event is a
*partial output* — surfaced to the run's stream consumer only.

## Consumer API

When the run is started with `stream=True`, partial events arrive as
`RunItemStreamEvent` with name `RunItemType.TOOL_PARTIAL_OUTPUT`:

```python
from troopai.adk.run.runner import Runner
from troopai.adk.run.stream import RunItemStreamEvent, RunItemType

result = Runner.run(agent, "Find docs about streaming.", stream=True)

async for event in result.stream_events():
    if isinstance(event, RunItemStreamEvent):
        if event.name == RunItemType.TOOL_PARTIAL_OUTPUT:
            inner = event.item["event"]  # the original ToolStreamEvent
            if inner.delta is not None:
                print(inner.delta, end="", flush=True)
        elif event.name == RunItemType.TOOL_OUTPUT:
            # One TOOL_OUTPUT per tool call, carrying the final value.
            print("\n[final]:", event.item["output"])
```

## Event shape

```python
@dataclass
class ToolStreamEvent:
    type: Literal["part_start", "part_delta", "part_end", "done"]
    index: int | None = None
    delta: str | None = None
    response: Any = None
```

The discriminator vocabulary mirrors `LLMStreamEvent` so streaming
tools and streaming LLM responses compose cleanly in the same
consumer loop.

## Approval gates

`requires_approval=True` and `streaming=True` coexist. The HITL
gate runs first; the iterator only starts after approval is
granted (and on resumption). Approval is a gate; streaming is the
body.

```python
@function_tool(
    name="deploy",
    description="Deploy with progress.",
    streaming=True,
    requires_approval=True,
)
async def deploy(env: str) -> AsyncIterator[ToolStreamEvent]:
    yield ToolStreamEvent(type="part_delta", delta=f"deploying to {env}…")
    yield ToolStreamEvent(type="done", response="deployed")
```

## Mutually incoherent flags

`streaming=True` cannot combine with these — each combination raises
`ValueError` at construction:

| Conflicting flag | Why |
|---|---|
| `cache=True` | Cache stores a single value, not a stream |
| `cache_function=…` | Only consulted with `cache=True` |
| `response_format="content_and_artifact"` | Artifact channel needs the full payload |
| `return_direct=True` | Return-direct semantics don't apply to a streaming intermediary |

## Running under the non-streaming path

If a streaming tool runs under `Runner.arun()` (without `stream=True`),
the executor drains the iterator silently and emits a
`logger.warning`. The final value still flows back to the LLM —
only the partial events are discarded. This is a deliberate
fail-safe: streaming-tool authors don't need a separate
non-streaming code path.

## Middleware preservation

`ToolMiddleware` registered via `Agent.middleware.tools` or
`WrapperToolset.middleware` observes the *final accumulated value*,
not individual chunks. The drain happens inside the innermost
middleware terminal so logging / metrics / tracing middlewares see
exactly one result per call regardless of the tool's streaming
mode.

## See also

- [`tools.md`](tools.md) — base FunctionTool surface
- [`middleware.md`](middleware.md) — `ToolMiddleware` Protocol
- [`../run/streaming_cancel.md`](../run/streaming_cancel.md) — cancel
  semantics that apply to streaming-tool batches too
- `examples/tools/streaming_tool_results.py` — runnable end-to-end
