# Streaming Cancellation

Graceful stop for ``Runner.run(..., stream=True)``. Two modes, one
method: ``RunResultStreaming.cancel(mode=...)``.

## TL;DR

```python
result = Runner.run(agent, "Write a long story", stream=True)

async for event in result.stream_events():
    if user_hit_escape():
        result.cancel(mode="immediate")
        break
```

After ``cancel()`` returns, the ``async for`` loop exits on its next
receive — no polling delay, no half-baked tool launches.

## Two modes

| Mode          | Producer task               | Event queue              | Semantic contract                                              |
|---------------|-----------------------------|--------------------------|----------------------------------------------------------------|
| `immediate`   | ``task.cancel()`` **sync**  | Drained + sentinel       | Stop as fast as possible. In-flight tools finish, nothing new starts. |
| `after_turn`  | Left running                | Untouched                | Finish the current LLM response and its tool batch, then stop. |

### `mode="immediate"`

Synchronously:

1. Flips ``_cancel_mode = CancelMode.IMMEDIATE``.
2. Drains every pending event from the queue so the consumer sees
   nothing stale.
3. Cancels the background producer task via ``task.cancel()``. A
   ``CancelledError`` is scheduled at its next ``await``.
4. Enqueues a ``_QueueCompleteSentinel`` so the blocked consumer
   ``queue.get()`` wakes immediately.

The previous implementation used a 100ms polling loop inside
``stream_events()`` so it could re-check the flag. That's gone — the
sentinel wakes the consumer on the same scheduling round as the
``cancel()`` call.

In-flight tool calls are **not** interrupted mid-execution, but the
between-tool check in ``execute_tool_calls_streamed`` prevents the
batch from starting the next tool:

```python
for tool_call in tool_calls:
    if result.cancel_mode == CancelMode.IMMEDIATE:
        break  # don't start any more tools
    ...
```

### `mode="after_turn"`

A cooperative stop. The flag is set; the producer task is left alone.
The streaming agent loop checks the flag at the top of each turn:

```python
while result.current_turn < result.max_turns:
    ...
    if result.cancel_mode in (CancelMode.IMMEDIATE, CancelMode.AFTER_TURN):
        break
```

Meaning: the current LLM response finishes streaming, its tool batch
runs to completion, the results are applied, then the loop exits
before starting the next turn. Use this when you want partial results
without amputating in-flight work.

## Cancellation is safe to call from any async context

``cancel()`` is a plain synchronous method, but it works correctly
whether called:

- From inside the ``async for`` iterator (the usual case).
- From an external task observing the stream.
- Before ``stream_events()`` has even been awaited (deferred producer
  path). In this case no task exists yet; the sentinel is simply
  enqueued and the first ``stream_events()`` call exits immediately.

## What the consumer sees

After ``cancel()``:

- ``result.is_complete`` → ``True`` once the iterator exits.
- ``result.cancel_mode`` → ``CancelMode.IMMEDIATE`` or
  ``CancelMode.AFTER_TURN`` depending on the call.
- ``result.final_output`` → whatever ``NextStepFinalOutput`` was
  resolved before cancel fired; ``None`` if cancellation interrupted
  turn 1 before any output was produced.
- ``result.new_items`` → every RunItem that actually reached the
  consumer before cancel landed.
- ``result.usage`` → token counts for LLM calls that actually
  completed. Cancelled mid-stream calls contribute what they billed.

## Tests

See ``tests/unit/run/test_streaming_cancel.py`` for the full
contract:

- ``cancel(mode="immediate")`` sets the mode, drains the queue, enqueues
  the sentinel.
- The producer task is cancelled synchronously.
- ``cancel()`` with no task is a no-op.
- ``stream_events()`` exits within a tight 1s budget under the
  polling-free implementation.
- ``mode="after_turn"`` sets the flag but leaves the task and the
  queue alone.
- ``execute_tool_calls_streamed`` stops before launching the next
  tool once IMMEDIATE fires, both pre-batch and mid-batch.
