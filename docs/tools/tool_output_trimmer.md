# `trim_tool_output` — Wrap a Tool in an Output-Size Budget

Returns a new `FunctionTool` whose `on_invoke` stringifies and truncates the
original tool's output to fit a character and/or token budget. Composes on
an existing tool instance — handy when wiring third-party tools whose raw
output is too chatty for the model.

## When to use this vs `FunctionTool.max_result_tokens`

| Concern | Use |
|---------|-----|
| You own the tool's decorator call | `max_result_tokens=N` on the decorator |
| You want to wrap a third-party tool instance | `trim_tool_output(tool, ...)` |
| You need a character cap (no tokenizer) | `trim_tool_output(tool, max_chars=N)` |
| You want both char + token caps composed | `trim_tool_output(tool, max_chars=N, max_tokens=M, model=...)` |

`trim_tool_output` is purely additive — it composes on top of any existing
tool and leaves the original instance untouched.

## Signature

```python
def trim_tool_output(
    tool: FunctionTool,
    *,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    marker: str = "... [truncated]",
) -> FunctionTool
```

- `max_chars` — hard character cap. Exceeding text is cut to
  `max_chars - len(marker)` and the marker is appended.
- `max_tokens` — hard token cap (requires `model`). Uses
  `TokenCounter.count_text()` and shrinks iteratively until the budget is
  met. Bounded at 5 passes.
- `model` — litellm model id (e.g. `"gpt-4o"`) used for token counting.
  Required when `max_tokens` is set.
- `marker` — truncation suffix. Defaults to `"... [truncated]"`.

At least one of `max_chars` or `max_tokens` must be provided.

## Basic usage

```python
from troopai.adk.tools import function_tool, trim_tool_output


@function_tool
def fetch_docs(query: str) -> str:
    return retrieve_huge_document(query)  # might be 50k chars


trimmed_fetch = trim_tool_output(
    fetch_docs,
    max_tokens=2000,
    model="gpt-4o",
)

agent = Agent(name="rag", tools=[trimmed_fetch])
```

## `content_and_artifact` format

When `tool.response_format == "content_and_artifact"`, the wrapper trims
**only the content part** of the returned `(content, artifact)` tuple —
the artifact is passed through unchanged.

```python
@function_tool(name="rag", response_format="content_and_artifact")
def rag_search(query: str) -> tuple[str, list[Document]]:
    docs = retrieve(query)
    return f"Found {len(docs)} results\n\n" + "\n".join(d.text for d in docs), docs


trimmed = trim_tool_output(rag_search, max_chars=1200)
# The LLM sees a 1200-char summary, while the application still receives
# the full Document list via FunctionToolCallResult.artifact.
```

## Non-string results

If the source tool returns a non-string value, the wrapper calls `str()`
on it before applying the budget:

```python
@function_tool
def fetch_rows() -> list[dict]: ...

trimmed = trim_tool_output(fetch_rows, max_chars=500)
# The LLM sees str(rows)[:500] + "... [truncated]"
```

## Original tool is untouched

`trim_tool_output` uses `dataclasses.replace()` to clone the tool with a
wrapped `on_invoke`. The original instance is never mutated, and the clone
gets a fresh cache:

```python
trimmed = trim_tool_output(my_tool, max_chars=200)
assert trimmed is not my_tool
assert trimmed.on_invoke is not my_tool.on_invoke
# my_tool.on_invoke still returns the full, un-trimmed output
```

## See also

- `src/troopai/adk/tools/tool_output_trimmer.py` — implementation
- `tests/unit/tools/test_tool_output_trimmer.py` — behavior tests
- `src/troopai/adk/tools/function_tool.py` — `FunctionTool.max_result_tokens`
  (the alternative, declared at decoration time)
