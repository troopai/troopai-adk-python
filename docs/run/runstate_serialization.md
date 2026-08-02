# `RunState` Serialization and Approval Audit

Persistence-friendly serialization for interrupted agent runs, plus
structured audit metadata for human-in-the-loop approvals.

## TL;DR

```python
from troopai.adk.run.state import RunState
from troopai.adk.run.runner import Runner

# First run — may halt for HITL approval
result = await Runner.arun(agent, "Delete user u-1")
if result.state is not None:
    await db.save_pending_approval(result.state.to_json())

# Later — load, approve with audit trail, resume
state = RunState.from_json(await db.get_pending_approval(id))
deferred = state.deferred_tool_requests.approvals[0]
state.approve(
    deferred,
    approver_id="alice@example.com",
    reason="authorized per change CR-1234",
)
result = await Runner.arun(agent, state)
```

## `to_json()` / `from_json()`

Thin JSON wrappers over ``to_dict()`` / ``from_dict()``.

- ``to_json()`` is exactly ``json.dumps(self.to_dict())`` — a bare
  dict, no envelope and no stamp of any kind.
- ``from_json()`` is exactly ``from_dict(json.loads(data))``.
  ``from_dict`` reads every field by name with a safe default
  (``dict.get(key, default)``), so it is **tolerant**: a payload
  produced by an earlier build that lacks a later-added field loads
  cleanly (the field takes its default), and any key ``from_dict``
  does not recognise is ignored.

| Method        | Returns            | Behaviour |
|---------------|--------------------|-----------|
| ``to_dict``   | ``dict[str, Any]`` | Bare dict |
| ``to_json``   | ``str``            | ``json.dumps`` of the bare dict |
| ``from_dict`` | ``RunState``       | Per-field ``dict.get`` with defaults; extras ignored |
| ``from_json`` | ``RunState``       | ``from_dict`` of the parsed JSON |

### Evolving the persisted shape

Add a new field with a safe default and serialize it in
``to_dict``; older payloads simply omit the key and ``from_dict``
supplies the default. Removing or renaming a field is a separate
change — rename the field and ``from_dict`` stops reading the old
name, dropping it from any stale payload. No version integer, no
mismatch check: a stale payload degrades to defaults rather than
failing loudly, which is the intended behaviour for resumable
state.

## Structured approval metadata

``approve()`` and ``reject()`` accept optional keyword-only
``approver_id`` and ``reason``:

```python
state.approve(deferred_call, approver_id="alice@example.com", reason="ok")
state.reject(
    deferred_call,
    message="Not authorized for this resource",   # shown to the LLM
    approver_id="bob@example.com",                 # audit only
    reason="policy violation: tier-0 write",       # audit only
)
```

Metadata lands on ``RunState.approval_metadata`` — a dict keyed by
``tool_call_id``. ``approved_tools`` / ``rejected_tools`` are the
resumption-driver lists; the audit metadata is a separate dict, so
resumption logic that never opens it is unaffected.

### `message` vs `reason` — don't conflate

The distinction is deliberate:

| Field       | Audience       | Purpose                                     |
|-------------|----------------|---------------------------------------------|
| ``message`` | LLM            | Tell the model *why* so it picks a better alternative. Goes into the next turn's tool-result message. |
| ``reason``  | Humans / audit | Internal rationale for the compliance log. **Not** shown to the model. |

Putting a compliance explanation in ``message`` leaks policy details
into the model prompt. Putting a model hint in ``reason`` means the
LLM never sees it. Choose deliberately.

### `ApprovalMetadata`

```python
@dataclass
class ApprovalMetadata:
    approver_id: str | None = None
    reason: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
```

Records automatically when ``approve()`` / ``reject()`` are called with
any of the keyword-only fields set. If neither ``approver_id`` nor
``reason`` is supplied, the metadata dict is left untouched — no
empty entries, no audit noise.

## What does NOT cross the wire

- ``Agent`` definitions — you rebuild those in code; resumption
  assumes the same agent topology is available.
- ``LLM`` instances — same rationale. ``from_json()`` gives you a
  state to feed into a Runner; you still configure the Runner
  yourself.
- Live task handles, event queues, async primitives — serialization
  only covers the cold state (conversation history, deferred calls,
  approvals, metadata). Hot in-flight work is intentionally out of
  scope.

## Tests

See ``tests/unit/run/test_runstate_serialization.py`` for the full
contract: JSON round-trip, no version key emitted, tolerant load of
older payloads with unrecognised keys, audit-metadata separation,
and the ``to_dict`` / ``from_dict`` round-trip.
