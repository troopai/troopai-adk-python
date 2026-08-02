# `RunResult` Helpers

Three small conveniences on `RunResult` that keep long-running processes and
provider-native response chaining ergonomic.

## `last_response_id: Optional[str]`

The `response_id` of the most recent assistant message in this run. Walks
`new_items` in reverse and returns the first `MessageOutputItem.id` it
finds.

**Use cases:**

- Provider-native response chaining (OpenAI Responses API, prompt-caching
  audit trails)
- Correlating a run with provider-side logs
- Resuming a conversation via provider state APIs without re-sending history

**Returns `None`** when:

- The run produced no message output (e.g. every response was pure tool
  calls, or the run was interrupted before any text landed)
- `release_agents(release_new_items=True)` has been called

```python
result = await Runner.arun(agent, "Hello")
logger.info("Last response id: %s", result.last_response_id)
```

## `release_agents(*, release_new_items: bool = True)`

Drop strong references to the agent graph and (optionally) the run items.
Long-lived processes that retain many completed `RunResult` instances can
pin significant memory — system prompts, tool closures, handoff targets,
compiled schemas — all reachable via `result.last_agent`.

```python
result = await Runner.arun(agent, "Quick task")
logger.info(result.final_output)

# Release heavyweight refs; keep cheap metadata for audit trail.
result.release_agents()
cache[user_id] = result  # safe to retain forever now
```

After `release_agents()`:

- `result.last_agent is None`
- `result.new_items == []` (unless `release_new_items=False`)
- `result.final_output`, `result.user_prompt`, `result.context`, and both
  guardrail result tuples are preserved

Pass `release_new_items=False` to keep the conversation history intact while
still dropping the agent reference:

```python
result.release_agents(release_new_items=False)
assert result.last_agent is None
assert len(result.new_items) > 0  # history still there
assert result.last_response_id is not None  # still accessible
```

## `to_input_list(mode=...)`

Convert the run's accumulated items into a message list suitable for feeding
into the next `Runner.arun()` call.

```python
next_input = result.to_input_list()
next_result = await Runner.arun(agent, next_input)
```

### Modes

- `mode="preserve_all"` (default) — every item's `to_param()` is emitted in
  order. Reasoning blocks, tool calls, and tool outputs are all preserved.
  This is the shape required for multi-turn tool-use flows against providers
  that demand the full prior trace (Anthropic, OpenAI Responses API).
- `mode="normalized"` — **forward-compatible reservation**. Identical to
  `preserve_all` today, but reserved for a future mode that strips reasoning
  blocks and collapses provider-specific metadata for models that don't
  accept them. Callers that opt in today will automatically pick up the
  stripped variant once it ships — no breaking change.

```python
# Opt into the future stripped shape now; no behavior change today.
cheap_input = result.to_input_list(mode="normalized")
```

## Composition

The three helpers compose cleanly:

```python
result = await Runner.arun(agent, "Task")
remembered_id = result.last_response_id  # capture for provider chaining
result.release_agents()                  # drop agent graph
await save_audit(result)                 # final_output + usage still there
```

## See also

- `src/troopai/adk/types/run/run_result.py` — dataclass definition
- `tests/unit/run/test_run_result_helpers.py` — full behavior tests
