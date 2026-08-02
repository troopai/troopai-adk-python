(guides/guardrails)=

# 🛡️ Guardrails

Guardrails are the input and output safety gates on either side of the
agent loop. They intercept every run at Stage 2 (before the LLM sees
the prompt) and again at Stage 4 (after the loop produces its final
reply), giving you deterministic, cost-free checkpoints for content
policy, length limits, and any custom logic your application needs.

```{admonition} Core contract
:class: important

A guardrail is a **sync or async pure function**. It takes a data object
(prompt or output), inspects it, and returns a verdict. It never calls an
LLM, never has side-effects on run state, and never fails silently.
```

## Where guardrails sit in the pipeline

Every call to `Runner.arun(agent, prompt)` follows the same shape:

```
Input → [Stage 2: Input guardrails] → Agent loop → [Stage 4: Output guardrails] → Result
```

**Stage 2 — Input guardrails** run before the loop opens. A tripwire
stops the run immediately; the LLM never receives the prompt and you pay
zero tokens.

**Stage 4 — Output guardrails** run once the loop has produced its final
reply, before `RunResult` is returned to the caller. Output guardrails
support an optional `remediation` string: when set, the runner re-prompts
the agent with that feedback instead of raising on the first failure, giving
the model one chance to self-correct.

Guardrail results from both stages are stored in
`RunResult.guardrail_results` (an `AgentGuardrailResults` object with
`.input` and `.output` tuples) and are always present, whether or not
any tripwire fired.

## Verdict shape

Agent-level guardrail functions have the signature:

```python
async def my_guardrail(
    data: AgentInputGuardrailData,           # or AgentOutputGuardrailData
) -> AgentGuardrailFunctionOutput:
    ...
```

`AgentGuardrailFunctionOutput` carries these fields:

| Field | Type | Meaning |
|---|---|---|
| `tripwire_triggered` | `bool` | `True` halts execution immediately |
| `severity` | `AgentGuardrailSeverity \| None` | When set, overrides `tripwire_triggered` for the halt decision |
| `output_info` | `Any` | Optional metadata (entities found, pattern matched, …) |
| `transformed_output` | `Any` | Complete replacement text for the agent output (output guardrails, `str` only). When set, the runner substitutes it for `RunResult.final_output` and rewrites the trailing history message. Must also set `tripwire_triggered=True`. |
| `changed_spans` | `list[GuardrailSpan] \| None` | Character ranges the guardrail flagged — for observability and audit only; never used to apply the transform. |

Every verdict maps onto a shared `GuardrailAction` (`PASS` / `RAISE` /
`TRANSFORM`) via `resolved_action()`. The runner dispatches uniformly on
this vocabulary across agent, tool, and flow guardrail levels.

### Severity levels

When `severity` is not `None` it overrides `tripwire_triggered`:

| Severity | Halts? | Use case |
|---|---|---|
| `INFO` | No | Low-confidence detection; telemetry only |
| `WARNING` | No | Auditable signal that should not block |
| `ERROR` | Yes | Same effect as `tripwire_triggered=True` |

When `severity` is `None` (the default), `tripwire_triggered` alone
controls whether execution halts.

A tripped input guardrail raises `AgentInputGuardrailTripwireTriggered`;
a tripped output guardrail raises `AgentOutputGuardrailTripwireTriggered`.
Both are importable from `troopai.adk.exceptions`.

## Built-in guardrail hub

For the most common safety concerns the ADK ships ready-to-use guardrails
under `troopai.adk.guardrails`:

| Factory | Level | Default `on_fail` | Description |
|---|---|---|---|
| `pii_guardrail()` | output | `RAISE` | Detects email, URL, and phone patterns; `TRANSFORM` available for automatic span-redaction. |
| `injection_scan_guardrail()` | input | `RAISE` | Detects high-confidence prompt-injection markers. |
| `fence_untrusted_text(text)` | prompt helper | n/a | Wraps untrusted text in a nonce-fenced delimiter for safe prompt construction. |
| `wrong_language_guardrail(target_language=…)` | output | `RAISE` | Trips when the output is in the wrong language. Needs `[guardrails-lingua]` extra. |

```python
from troopai.adk.guardrails import injection_scan_guardrail, pii_guardrail
from troopai.adk.types.guardrails.action import GuardrailAction

agent = Agent(
    name="Support",
    system_prompt="Help customers.",
    guardrails=AgentGuardrails(
        input=[injection_scan_guardrail()],
        output=[pii_guardrail(on_fail=GuardrailAction.TRANSFORM)],
    ),
)
```

See {ref}`guardrails/guardrail_hub` for the full reference including
`PatternScanner`, `mask_pii_spans`, `detect_wrong_language`, and the guardrail
audit side-car (`RunResult.guardrail_audit`).

## Writing a guardrail

All guardrails are user-authored Python functions. The ADK provides hooks to
attach them and a built-in hub for common checks. For application-specific
logic, write a function as described below.

### Using the decorator (recommended)

The `@agent_input_guardrail` and `@agent_output_guardrail` decorators are
the idiomatic approach. They accept both sync and async functions.

```python
from troopai.adk.agents.agent_guardrails import (
    agent_input_guardrail,
    AgentInputGuardrailData,
    AgentGuardrailFunctionOutput,
)

# Async (most common)
@agent_input_guardrail
async def no_sql_injection(
    data: AgentInputGuardrailData,
) -> AgentGuardrailFunctionOutput:
    text = data.user_prompt if isinstance(data.user_prompt, str) else str(data.user_prompt)
    triggered = any(kw in text.upper() for kw in ("DROP TABLE", "OR 1=1", "UNION SELECT"))
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=triggered,
        output_info={"reason": "SQL injection pattern detected"} if triggered else None,
    )

# Sync (also supported — the runner awaits if needed)
@agent_input_guardrail
def block_empty_input(
    data: AgentInputGuardrailData,
) -> AgentGuardrailFunctionOutput:
    text = data.user_prompt if isinstance(data.user_prompt, str) else str(data.user_prompt)
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=len(text.strip()) == 0,
        output_info={"reason": "Empty input rejected"},
    )
```

Use keyword arguments to set `name`, `run_in_parallel`, `timeout`, and
`timeout_policy`:

```python
@agent_input_guardrail(
    name="sql_injection_guard",
    run_in_parallel=False,    # Block before the LLM call
    timeout=2.0,
    timeout_policy=AgentTimeoutPolicy.FAIL,
)
async def guarded_sql_check(
    data: AgentInputGuardrailData,
) -> AgentGuardrailFunctionOutput:
    ...
```

Output guardrails follow the same pattern with `@agent_output_guardrail`,
which additionally supports `remediation` and `max_retries`:

```python
from troopai.adk.agents.agent_guardrails import (
    agent_output_guardrail,
    AgentOutputGuardrailData,
    AgentGuardrailFunctionOutput,
)

@agent_output_guardrail(
    remediation="Your reply contained a code block with credentials. "
                "Please regenerate without including sensitive tokens.",
    max_retries=2,
)
async def no_credentials_in_output(
    data: AgentOutputGuardrailData,
) -> AgentGuardrailFunctionOutput:
    text = str(data.output)
    triggered = any(
        marker in text.lower() for marker in ("api_key=", "password=", "secret=")
    )
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=triggered,
        output_info={"reason": "Credential pattern in output"},
    )
```

### Using the dataclass directly

When you need to share a guardrail instance across agents or configure it
dynamically, construct `AgentInputGuardrail` or `AgentOutputGuardrail`
directly:

```python
from troopai.adk.agents.agent_guardrails import (
    AgentInputGuardrail,
    AgentInputGuardrailData,
    AgentGuardrailFunctionOutput,
)

async def _check_length(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    text = data.user_prompt if isinstance(data.user_prompt, str) else str(data.user_prompt)
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=len(text) > 8000,
        output_info={"length": len(text)},
    )

length_guard = AgentInputGuardrail(
    guardrail_function=_check_length,
    name="max_prompt_length",
    run_in_parallel=False,
)
```

### Registering guardrails on an agent

Pass guardrails as an `AgentGuardrails` config on the agent:

```python
from troopai.adk import Agent
from troopai.adk.agents.agent_guardrails import AgentGuardrails

agent = Agent(
    name="support-bot",
    system_prompt="You are a customer support assistant.",
    guardrails=AgentGuardrails(
        input=[
            length_guard,
            no_sql_injection,            # decorator-based guardrail
        ],
        output=[
            no_credentials_in_output,    # custom output guardrail
        ],
    ),
)
```

### Run-scope guardrails

`RunConfig.guardrails` applies a second layer of guardrails that run
before the agent's own guardrails, across every agent in the run. This
is useful for organisation-wide policies applied uniformly regardless of
which agent handles the request:

```python
from troopai.adk.run import RunConfig
from troopai.adk.agents.agent_guardrails import AgentGuardrails

result = await Runner.arun(
    agent,
    prompt,
    run_config=RunConfig(
        guardrails=AgentGuardrails(
            input=[organisation_content_guard],
        )
    ),
)
```

### Referencing guardrails from declarative config

In a JSON or YAML agent config, attach guardrail functions via a dotted
`ref` pointing to an `AgentInputGuardrail` or `AgentOutputGuardrail`
instance in an importable module:

```json
"guardrails": {
  "input": [
    {"ref": "my_pkg.guards.no_sql_injection"},
    {"ref": "my_pkg.guards.length_guard"}
  ],
  "output": [
    {"ref": "my_pkg.guards.no_credentials_in_output"}
  ]
}
```

The referenced symbols must be `AgentInputGuardrail` (or
`AgentOutputGuardrail`) instances at module scope — decorator-created
or dataclass-constructed. See
[Declarative config](../config/config.md) for the full schema.

## Tool-level vs agent-level guardrails

Agent-level guardrails (described above) operate on the whole-run
boundary. Tool-level guardrails operate on each individual tool
invocation and have a richer three-state verdict.

| | Agent guardrails | Tool guardrails |
|---|---|---|
| **Scope** | Whole run (input prompt / final output) | Single tool call (arguments / return value) |
| **Classes** | `AgentInputGuardrail`, `AgentOutputGuardrail` | `ToolInputGuardrail`, `ToolOutputGuardrail` |
| **Decorators** | `@agent_input_guardrail`, `@agent_output_guardrail` | `@tool_input_guardrail`, `@tool_output_guardrail` |
| **Verdict type** | `AgentGuardrailFunctionOutput` | `ToolGuardrailFunctionOutput` |
| **Halt signal** | `tripwire_triggered=True` or `severity=ERROR` | `.raise_exception()` |
| **Soft reject** | `severity=WARNING` (record, no halt) | `.reject_content(message)` — LLM sees the message, loop continues |
| **Pass** | `tripwire_triggered=False` | `.allow()` |
| **When to use** | Prompt-level policy (content in user input, compliance on final reply) | Call-level policy (block a specific argument pattern, redact a field from tool output) |

Tool guardrail example:

```python
from troopai.adk.tools.tool_guardrails import (
    tool_input_guardrail,
    ToolInputGuardrailData,
    ToolGuardrailFunctionOutput,
)

@tool_input_guardrail
async def reject_internal_paths(
    data: ToolInputGuardrailData,
) -> ToolGuardrailFunctionOutput:
    args = data.context.tool_args or {}
    path = args.get("path", "")
    if isinstance(path, str) and path.startswith("/etc"):
        return ToolGuardrailFunctionOutput.reject_content(
            message="Access to /etc is not permitted.",
            output_info={"path": path},
        )
    return ToolGuardrailFunctionOutput.allow()
```

Register tool guardrails directly on a `FunctionTool` via
`input_guardrails` / `output_guardrails`, or pass them inside a
`ToolGuardrails` config to `FunctionTool`.

## Guardrails are pure — no LLM calls

```{admonition} Guardrails must not call an LLM
:class: warning

Guardrail functions **must not call an LLM**. The contract is a sync or
async pure function: deterministic, bounded-latency, zero extra tokens.

If you need a nuanced semantic check — context-dependent detection,
entailment scoring, factuality grading — that belongs in the **evals**
subsystem (graders), not in a guardrail. Evals run on test data;
guardrails run on production traffic.
```

## Common patterns

### Composing guardrails in sequence

The `AgentGuardrails.input` and `AgentGuardrails.output` lists run in
order. A tripwire on any one of them halts the run; subsequent
guardrails in the list do not execute. Build your most selective (fastest)
checks first:

```python
AgentGuardrails(
    input=[
        block_empty_input,          # fastest, cheapest
        no_sql_injection,           # pattern matching
        custom_topic_guard,         # application-specific
    ],
)
```

### Per-tenant guardrail policies

Thread tenant identity through `RunContext.context` and gate inside the
guardrail function:

```python
@agent_input_guardrail
async def tenant_policy(
    data: AgentInputGuardrailData,
) -> AgentGuardrailFunctionOutput:
    ctx = data.context.context
    if isinstance(ctx, dict):
        tier = ctx.get("tier", "free")
        text = data.user_prompt if isinstance(data.user_prompt, str) else str(data.user_prompt)
        if tier == "free" and len(text) > 2000:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                output_info={"reason": "Prompt length exceeds free-tier limit"},
            )
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)
```

### Testing a guardrail in isolation

Because guardrails are pure functions, you can call them directly without
spinning up a full agent run:

```python
import asyncio
from unittest.mock import MagicMock
from troopai.adk.agents.agent_guardrails import AgentInputGuardrailData

async def test_no_sql_injection_fires():
    agent_mock = MagicMock()
    context_mock = MagicMock()

    data = AgentInputGuardrailData(
        context=context_mock,
        agent=agent_mock,
        user_prompt="SELECT * FROM users; DROP TABLE users;--",
    )
    result = await no_sql_injection.run(data)
    assert result.tripwire_triggered is True

asyncio.run(test_no_sql_injection_fires())
```

`AgentInputGuardrail.run()` and `AgentOutputGuardrail.run()` are async
and accept the matching `*Data` object directly, making unit tests
straightforward without the Runner.

### Timeout handling

Both input and output guardrails accept `timeout` (seconds) and
`timeout_policy`:

```python
from troopai.adk.agents.agent_guardrails import AgentTimeoutPolicy

@agent_input_guardrail(
    timeout=1.0,
    timeout_policy=AgentTimeoutPolicy.PASS,   # continue silently on timeout
    on_timeout=log_timeout_metric,
)
async def slow_external_check(
    data: AgentInputGuardrailData,
) -> AgentGuardrailFunctionOutput:
    ...
```

`FAIL` (the default) treats a timeout as a tripwire trigger — safe by
default. `PASS` lets the run continue silently, useful when an external
check is advisory and not availability-critical.

## See also

- {ref}`guardrails/guardrail_hub` — built-in hub reference: `pii_guardrail`,
  `injection_scan_guardrail`, `fence_untrusted_text`, `wrong_language_guardrail`,
  `PatternScanner`, the `GuardrailAction` vocabulary, and the guardrail audit
  side-car (`RunResult.guardrail_audit`).
- {ref}`guardrails/agent_guardrails` — full reference for agent-level
  guardrails: TRANSFORM action, severity, timeout, remediation, hooks,
  and exception types.
- {ref}`concepts/index` — the "Guardrails vs Middleware vs Hooks vs Sandbox"
  section explains how to choose the right extension point.
- LLM-as-judge belongs in the evals package, not in guardrails.
- [Tools guide](../guides/tools.md) — tool-level guardrails
  (`ToolInputGuardrail`, `ToolOutputGuardrail`) in full detail.
- [Architecture overview](../architecture/overview.md) — Stages 2 and 4
  in context.
- [Declarative config](../config/config.md) — the `guardrails` field and
  dotted `ref` syntax.
