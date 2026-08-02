(guardrails/agent_guardrails)=

# Agent-Level Guardrails

> **Scope:** This document covers **agent-level** guardrails — validation
> that runs at the agent boundary (input/output). Tool-level guardrails
> (validation around individual tool execution) are a separate system
> documented in `docs/tools/`.

Agent-level guardrails validate input before and output after agent execution. They provide safety checks for jailbreak prevention, PII detection, content policy enforcement, hallucination detection, and compliance.

## Overview

Guardrails are validation functions that attach to an `Agent` and run automatically during execution. Two types exist, corresponding to where they run in the execution flow:

- **Input guardrails** run on the user's prompt, either blocking before the agent starts or racing alongside it in parallel. They catch problems at the gate — jailbreak attempts, off-topic requests, prompt injection, and PII in inbound messages.
- **Output guardrails** run after the agent has produced its final response, before it is returned to the caller. They check the agent's answer for PII leakage, hallucinations, policy violations, and compliance requirements.

The execution flow is:

```
user prompt
    |
    v
[blocking input guardrails] -- trips --> AgentInputGuardrailTripwireTriggered
    |
    +---> agent loop starts
    |         |
    |     [parallel input guardrails racing alongside]
    |         |
    v         v
agent produces final output
    |
    v
[output guardrails, all in parallel]
    |                   |
  pass              trips: transform? --> substitute output (once per guardrail)
    |                   |
    |          no transform: remediation? --> re-run agent (up to max_retries)
    |                   |
    v               exhausted --> AgentOutputGuardrailTripwireTriggered
RunResult (guardrail_audit populated)
```

Config-level guardrails (set on `RunConfig`) merge with agent-level guardrails and always run first.

## Guardrail Action Vocabulary

Every guardrail verdict maps onto a shared three-value vocabulary that the
runner uses to dispatch uniformly across agent, tool, and flow levels:

```python
from troopai.adk.types.guardrails.action import GuardrailAction, GuardrailSpan
```

| Action | Meaning |
|---|---|
| `PASS` | Accept the checked artifact unchanged and continue. |
| `RAISE` | Halt the run — the tripwire fired. |
| `TRANSFORM` | Substitute the checked artifact with the replacement the guardrail supplies (output guardrails only). |

Each verdict type exposes a `resolved_action()` method. For
`AgentGuardrailFunctionOutput`:

- `transformed_output` is set → `TRANSFORM`
- Otherwise: `severity=ERROR` or `tripwire_triggered=True` (with no severity)
  → `RAISE`; anything else → `PASS`

`GuardrailSpan` is the companion observability type — a frozen `(start, end,
reason)` dataclass marking a character range the guardrail flagged. Spans are
for observability only; the runner never splices them. A transforming guardrail
supplies the complete replacement string, and spans ride along in the audit
record.

```python
@dataclass(frozen=True, kw_only=True)
class GuardrailSpan:
    start: int   # inclusive start index into the checked text
    end: int     # exclusive end index
    reason: str  # e.g. the matched pattern label
```

For how tool and flow guardrails map their verdicts onto the same vocabulary,
see {ref}`guardrails/guardrail_hub`.

## Quick Start

Define an agent with one input guardrail and one output guardrail, run it, and handle the exception if a violation is detected.

```python
import asyncio
from troopai.adk.agents import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentInputGuardrailData,
    AgentOutputGuardrailData,
    agent_input_guardrail,
    agent_output_guardrail,
)
from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    AgentOutputGuardrailTripwireTriggered,
)
from troopai.adk.run import Runner


@agent_input_guardrail
async def block_jailbreak(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    prompt = str(data.user_prompt).lower()
    if "ignore your instructions" in prompt or "forget your system prompt" in prompt:
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": "Jailbreak attempt detected"},
        )
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)


@agent_output_guardrail
async def block_pii_in_output(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    import re
    output_text = str(data.output)
    # Simple SSN pattern — replace with a real PII detector in production
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", output_text):
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": "SSN detected in output"},
        )
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)


agent = Agent(
    name="Support Agent",
    system_prompt="You are a helpful customer support agent.",
    guardrails=AgentGuardrails(
        input=[block_jailbreak],
        output=[block_pii_in_output],
    ),
)


async def main():
    try:
        result = await Runner.arun(agent, "How do I reset my password?")
        print(result.final_output)
    except AgentInputGuardrailTripwireTriggered as exc:
        print(f"Input rejected by '{exc.guardrail_result.guardrail.get_name()}'")
        print(f"Details: {exc.guardrail_result.guardrail_output.output_info}")
    except AgentOutputGuardrailTripwireTriggered as exc:
        print(f"Output rejected by '{exc.guardrail_result.guardrail.get_name()}'")


asyncio.run(main())
```

## Input Guardrails

Input guardrails receive the user's prompt and return a verdict. They run before or alongside the agent loop.

### Basic Usage

The `@agent_input_guardrail` decorator wraps a function and produces an `AgentInputGuardrail` instance. The function receives `AgentInputGuardrailData` and must return `AgentGuardrailFunctionOutput`.

**With decorator (no arguments):**

```python
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentInputGuardrailData,
    agent_input_guardrail,
)


@agent_input_guardrail
async def check_topic(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    prompt = str(data.user_prompt).lower()
    off_topic_keywords = ["cryptocurrency", "nft", "investment advice"]
    for keyword in off_topic_keywords:
        if keyword in prompt:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                output_info={"matched_keyword": keyword},
            )
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)
```

**With decorator arguments:**

```python
@agent_input_guardrail(name="topic_filter", run_in_parallel=False)
async def check_topic_blocking(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # Blocks before the agent starts — use when you need to save tokens
    ...
```

**Direct construction (without decorator):**

```python
from troopai.adk.agents.agent_guardrails import AgentInputGuardrail


async def check_topic_fn(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...


topic_guardrail = AgentInputGuardrail(
    guardrail_function=check_topic_fn,
    name="topic_filter",
    run_in_parallel=False,
)
```

Both approaches produce identical `AgentInputGuardrail` instances. Direct construction is useful when guardrails are defined programmatically or assembled at runtime.

Sync functions are also supported — the runner detects awaitable return values automatically:

```python
@agent_input_guardrail
def sync_keyword_check(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    if "bad_word" in str(data.user_prompt):
        return AgentGuardrailFunctionOutput(tripwire_triggered=True)
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)
```

### Blocking vs Parallel

The `run_in_parallel` attribute on `AgentInputGuardrail` controls the execution mode.

**Parallel mode (default, `run_in_parallel=True`):**

The guardrail runs concurrently with the agent loop using `asyncio.gather`. The agent starts immediately — if the guardrail later trips, the agent run is cancelled. This minimizes latency for the happy path (no violation) but does not save the tokens spent on the LLM call if the guardrail trips.

```python
@agent_input_guardrail(run_in_parallel=True)  # default
async def fast_check(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # Runs alongside the agent — no added latency on the happy path
    result = await call_fast_classifier(data.user_prompt)
    return AgentGuardrailFunctionOutput(tripwire_triggered=result.is_violation)
```

**Blocking mode (`run_in_parallel=False`):**

The guardrail runs sequentially and must complete before the agent starts. If it trips, the agent never runs — no LLM tokens are consumed. Use this for cheap checks (regex, keyword matching) that save significant tokens when they trigger.

```python
@agent_input_guardrail(run_in_parallel=False)
async def keyword_filter(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # Blocks the agent start — trips here saves the full LLM call
    banned = ["hack", "exploit", "bypass"]
    if any(w in str(data.user_prompt).lower() for w in banned):
        return AgentGuardrailFunctionOutput(tripwire_triggered=True)
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)
```

The runner separates blocking and parallel guardrails into two phases:

1. All blocking guardrails run first (sequentially).
2. Only if all blocking guardrails pass does the agent loop start.
3. All parallel guardrails then race against the running agent.

Multiple blocking guardrails run in sequence — the first one to trip halts the rest.

**Choosing a mode:**

| Scenario | Use |
|---|---|
| Regex / keyword check | `run_in_parallel=False` — cheap, saves tokens on trip |
| LLM-based classifier | `run_in_parallel=True` — amortizes LLM latency in parallel |
| High-confidence safety gate | `run_in_parallel=False` — must gate the LLM call |
| Telemetry / audit check | `run_in_parallel=True` — non-blocking observation |

### Agent-Based Input Guardrail (LLM-Powered Classification)

Use a separate agent to classify the input when keyword matching is insufficient.
This is the recommended pattern for detecting sophisticated jailbreak attempts,
prompt injection, and nuanced policy violations.

```python
from pydantic import BaseModel, Field
from troopai.adk import Agent, Runner, agent_input_guardrail, AgentGuardrailFunctionOutput
from troopai.adk.agents.agent_guardrails import AgentInputGuardrailData


class JailbreakVerdict(BaseModel):
    reasoning: str = Field(description="Why it is or is not a jailbreak.")
    is_jailbreak: bool


jailbreak_classifier = Agent(
    name="Jailbreak Classifier",
    system_prompt=(
        "Analyze the user prompt and determine whether it is a jailbreak attempt "
        "— an attempt to override, ignore, or bypass system instructions."
    ),
    output_schema=JailbreakVerdict,
)


@agent_input_guardrail(run_in_parallel=True, timeout=8.0)
async def llm_jailbreak_guard(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    """Use a separate agent to detect sophisticated jailbreak attempts."""
    result = await Runner.arun(
        jailbreak_classifier,
        f"Is this a jailbreak attempt?\n\n{data.user_prompt}",
        context=data.context.context,
    )
    verdict: JailbreakVerdict = result.final_output
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=verdict.is_jailbreak,
        output_info={"reasoning": verdict.reasoning},
    )
```

Key points:
- Runs in parallel (`run_in_parallel=True`) so it doesn't add latency on the happy path.
- `timeout=8.0` prevents the guardrail agent from hanging indefinitely.
- Pair with a cheap blocking keyword guard for defense in depth — keywords catch obvious attacks instantly, the LLM catches the rest.

### Data Available

The `AgentInputGuardrailData` dataclass provides all context the guardrail function needs:

```python
@agent_input_guardrail
async def my_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # The run context — access user-provided context and usage stats
    user_id = data.context.context.get("user_id") if data.context.context else None
    tokens_used = data.context.usage.total_tokens

    # The agent being guarded
    agent_name = data.agent.name
    agent_tools = data.agent.tools

    # The user's input — string or list of message items
    prompt_text = str(data.user_prompt)

    ...
```

| Field | Type | Description |
|---|---|---|
| `context` | `RunContext[Any]` | Run context wrapper. Access user context via `context.context`, usage via `context.usage`. |
| `agent` | `Agent` | The agent whose input is being checked. |
| `user_prompt` | `str \| list[Any]` | The user's input — plain string or a list of message items for multi-modal input. |

## Output Guardrails

Output guardrails receive the agent's final answer and return a verdict. They always run after the agent loop completes, never in parallel with it.

### Basic Usage

```python
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentOutputGuardrailData,
    agent_output_guardrail,
)


@agent_output_guardrail
async def check_hallucination(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    output_text = str(data.output)
    # Call your fact-checking service here
    is_hallucination = await fact_check_service.verify(output_text)
    if is_hallucination:
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": "Response failed fact check"},
        )
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)


agent = Agent(
    name="Research Agent",
    system_prompt="Answer questions with verified facts.",
    guardrails=AgentGuardrails(output=[check_hallucination]),
)
```

All output guardrails on an agent run in parallel via `asyncio.gather`. The order of results in `RunResult.guardrail_results.output` matches the order guardrails were declared on the agent.

### Agent-Based Output Guardrail (LLM-Powered Validation)

The most powerful pattern: use a separate agent inside the guardrail function
to reason about the primary agent's output. This handles nuanced checks that
string matching cannot — "Does this contain medical advice?", "Is this a
hallucination?", "Does this violate our content policy?"

```python
from pydantic import BaseModel, Field
from troopai.adk import Agent, Runner, agent_output_guardrail, AgentGuardrailFunctionOutput
from troopai.adk.agents.agent_guardrails import AgentOutputGuardrailData


class MedicalAdviceVerdict(BaseModel):
    reasoning: str = Field(description="Why it does or does not contain medical advice.")
    contains_medical_advice: bool


# Lightweight classifier agent with structured output
guardrail_agent = Agent(
    name="Medical Advice Detector",
    system_prompt=(
        "Analyze the given text and determine whether it contains medical advice. "
        "Medical advice includes diagnoses, treatment recommendations, medication "
        "suggestions, or health guidance that should come from a licensed professional."
    ),
    output_schema=MedicalAdviceVerdict,
)


@agent_output_guardrail(timeout=10.0)
async def no_medical_advice(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    """Use a separate agent to detect medical advice in the output."""
    result = await Runner.arun(
        guardrail_agent,
        f"Analyze this text for medical advice:\n\n{data.output}",
        context=data.context.context,
    )
    verdict: MedicalAdviceVerdict = result.final_output
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=verdict.contains_medical_advice,
        output_info={"reasoning": verdict.reasoning},
    )


agent = Agent(
    name="Support Agent",
    system_prompt="Help customers. Never give medical advice.",
    guardrails=AgentGuardrails(output=[no_medical_advice]),
)
```

Key points:
- The guardrail agent uses `output_schema` for a typed verdict (not free-text).
- `data.context.context` propagates the parent's user context to the guardrail agent.
- Set `timeout` on agent-based guardrails — they make LLM calls that can hang.
- The guardrail agent's token cost is separate from the primary agent's cost.

### Data Available

The `AgentOutputGuardrailData` dataclass provides the agent's output alongside the run context:

```python
@agent_output_guardrail
async def my_output_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    # The run context
    user_id = data.context.context.get("user_id") if data.context.context else None

    # The agent that produced the output
    agent_name = data.agent.name

    # The agent's output — str for plain text, Pydantic model for structured output
    output_text = str(data.output)

    ...
```

| Field | Type | Description |
|---|---|---|
| `context` | `RunContext[Any]` | Run context wrapper. Access user context via `context.context`, usage via `context.usage`. |
| `agent` | `Agent` | The agent that produced the output being checked. |
| `output` | `Any` | The agent's final output — `str` for plain text, a Pydantic model instance for structured output. |

### TRANSFORM: Span-Redaction Without Failing the Run

An output guardrail can rewrite the agent's output without halting the run by
setting `transformed_output` on its verdict. The runner then calls
`apply_output_transform`, which substitutes the replacement both in
`RunResult.final_output` and in the trailing assistant message in
`new_items` — so the persisted session events and any memory extraction see the
masked text rather than the raw output.

```python
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentOutputGuardrailData,
    agent_output_guardrail,
)
from troopai.adk.types.guardrails.action import GuardrailSpan
import re

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


@agent_output_guardrail
async def mask_emails(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    output = data.output
    if not isinstance(output, str):
        return AgentGuardrailFunctionOutput(tripwire_triggered=False)

    matches = list(EMAIL_PATTERN.finditer(output))
    if not matches:
        return AgentGuardrailFunctionOutput(tripwire_triggered=False)

    # Build the complete masked text (never splice spans — supply the full string)
    masked = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", output)
    spans = [
        GuardrailSpan(start=m.start(), end=m.end(), reason="email")
        for m in matches
    ]
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=True,   # required as halt fallback
        transformed_output=masked,
        changed_spans=spans,       # observability only — not used to apply the mask
        output_info={"masked_count": len(matches)},
    )
```

Key properties of the TRANSFORM action:

- **Opt-in.** Without a `transformed_output` value the runner behaves exactly
  as before — the `TRANSFORM` path is only active when the field is non-`None`.
- **Bounded.** Each guardrail may transform at most once per run. The runner
  tracks which guardrails have transformed and will not apply a second
  substitution from the same guardrail.
- **Text outputs only.** `apply_output_transform` rewrites a trailing
  `MessageOutputItem`; there is no structured-output equivalent.
- **Halt fallback required.** A transforming verdict must also set
  `tripwire_triggered=True` (and leave `severity` unset) so the run still
  halts when the runner cannot apply the substitution — for example, when the
  output is structured rather than text.
- **`changed_spans` is observability.** The runner never splices spans; the
  guardrail is responsible for computing the entire replacement string itself.

:::{admonition} Streaming caveat
:class: warning

When using streaming delivery, tokens may already have been emitted to a
consumer before the output guardrail runs. The substitution lands on the
persisted `RunResult` and session history, not on already-streamed tokens. Pair
a `TRANSFORM`-mode guardrail with non-streaming delivery when hard guarantees
are required.
:::

The built-in `pii_guardrail(on_fail=GuardrailAction.TRANSFORM)` applies this
pattern using `PatternScanner`. See {ref}`guardrails/guardrail_hub` for the
ready-made implementation.

## Severity Levels

By default, a guardrail that returns `tripwire_triggered=True` halts execution immediately. The `AgentGuardrailSeverity` enum adds a second axis of control: whether a detected violation should halt execution or be recorded for audit/monitoring without blocking.

```python
from troopai.adk.agents.agent_guardrails import AgentGuardrailFunctionOutput, AgentGuardrailSeverity
```

Three severity levels are available:

| Level | Logger call | Halts execution | Use case |
|---|---|---|---|
| `INFO` | `logger.debug(...)` | No | Low-confidence detections, telemetry, audit trails |
| `WARNING` | `logger.warning(...)` | No | Medium-confidence detections, reviewable but not blocking |
| `ERROR` | `logger.info(...)` | Yes | High-confidence violations that must halt |

### When Severity is None (Default)

When `severity` is not set on `AgentGuardrailFunctionOutput`, the `tripwire_triggered` field alone controls whether execution halts.

```python
# Classic behavior — tripwire_triggered controls everything
return AgentGuardrailFunctionOutput(tripwire_triggered=True)   # halts
return AgentGuardrailFunctionOutput(tripwire_triggered=False)  # passes
```

### When Severity is Set

When `severity` is set, it overrides `tripwire_triggered` for the halt decision. The value of `tripwire_triggered` no longer matters for execution control — `severity` is authoritative.

```python
# INFO — logged at DEBUG level, included in results, never halts
@agent_input_guardrail
async def audit_log_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    contains_pii = await pii_detector.scan(str(data.user_prompt))
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=contains_pii,     # recorded, but does not halt
        severity=AgentGuardrailSeverity.INFO,
        output_info={"pii_detected": contains_pii},
    )


# WARNING — logged at WARNING level, included in results, never halts
@agent_input_guardrail
async def medium_confidence_check(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    score = await risk_classifier.score(str(data.user_prompt))
    is_risky = score > 0.6
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=is_risky,
        severity=AgentGuardrailSeverity.WARNING if is_risky else None,
        output_info={"risk_score": score},
    )


# ERROR — halts execution (same effect as tripwire_triggered=True with no severity)
@agent_input_guardrail
async def high_confidence_block(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    score = await risk_classifier.score(str(data.user_prompt))
    is_violation = score > 0.95
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=is_violation,
        severity=AgentGuardrailSeverity.ERROR if is_violation else None,
        output_info={"risk_score": score},
    )
```

### Adaptive Severity

A common pattern is to return different severities based on confidence:

```python
@agent_input_guardrail
async def adaptive_jailbreak_check(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    score = await jailbreak_classifier.score(str(data.user_prompt))

    if score > 0.95:
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True,
            severity=AgentGuardrailSeverity.ERROR,
            output_info={"score": score, "action": "blocked"},
        )
    if score > 0.60:
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True,
            severity=AgentGuardrailSeverity.WARNING,
            output_info={"score": score, "action": "flagged"},
        )
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"score": score},
    )
```

Non-halting results (INFO and WARNING) are always included in `RunResult.guardrail_results.input` and `RunResult.guardrail_results.output`, making them available for downstream audit logging even though they did not trip the wire.

### Logging Behavior

The executor logs guardrail verdicts using the standard Python `logging` module under the logger name `troopai.adk.run.guardrails_executor`:

- **WARNING severity**: `logger.warning("... warning for agent '...'", ...)`
- **INFO severity**: `logger.debug("... info for agent '...'", ...)`
- **ERROR severity** (or `tripwire_triggered=True` with no severity): `logger.info("... tripwire triggered for agent '...'", ...)`
- **Timeouts**: always `logger.warning(...)` regardless of policy

## Timeout

Guardrails that call external services or LLMs can hang. The `timeout` attribute caps execution time — when exceeded, the `timeout_policy` determines what happens next.

### AgentTimeoutPolicy.FAIL (default)

Treat the timeout as a tripwire trigger. The guardrail is considered to have detected a violation. Execution halts with an exception (or proceeds to remediation if configured).

Use `FAIL` for safety-critical guardrails where a slow response means something is wrong and the request should not proceed.

```python
from troopai.adk.agents.agent_guardrails import AgentTimeoutPolicy, agent_input_guardrail


@agent_input_guardrail(timeout=3.0, timeout_policy=AgentTimeoutPolicy.FAIL)
async def safety_classifier(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # If this takes more than 3 seconds, the request is blocked
    result = await external_safety_api.check(str(data.user_prompt))
    return AgentGuardrailFunctionOutput(tripwire_triggered=result.is_unsafe)
```

### AgentTimeoutPolicy.PASS

Treat the timeout as a pass — execution continues silently. The guardrail result includes `output_info` describing the timeout, but `tripwire_triggered` is `False`.

Use `PASS` for non-blocking enrichment or monitoring guardrails where a slow response should not block the user.

```python
@agent_input_guardrail(timeout=2.0, timeout_policy=AgentTimeoutPolicy.PASS)
async def enrichment_check(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # If this takes more than 2 seconds, continue without enrichment
    metadata = await slow_enrichment_service.lookup(str(data.user_prompt))
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info=metadata,
    )
```

### on_timeout Callback

Use `on_timeout` to run side effects when a timeout fires — metrics, alerting, or audit logging. The callback receives a `AgentGuardrailTimeoutInfo` dataclass and runs after the timeout policy is applied. It cannot change the outcome.

```python
from troopai.adk.agents.agent_guardrails import AgentGuardrailTimeoutInfo, AgentInputGuardrail


async def alert_on_timeout(info: AgentGuardrailTimeoutInfo) -> None:
    # info.guardrail_name — name of the guardrail that timed out
    # info.agent_name    — name of the agent whose guardrail timed out
    # info.timeout       — the configured timeout in seconds
    # info.policy        — the AgentTimeoutPolicy that was applied
    await metrics.increment(
        "guardrail.timeout",
        tags={
            "guardrail": info.guardrail_name,
            "agent": info.agent_name,
            "policy": info.policy.value,
        },
    )
    if info.policy == AgentTimeoutPolicy.FAIL:
        await alerting.fire(f"Safety guardrail '{info.guardrail_name}' timed out")


guardrail = AgentInputGuardrail(
    guardrail_function=my_safety_check,
    timeout=5.0,
    timeout_policy=AgentTimeoutPolicy.PASS,
    on_timeout=alert_on_timeout,
)
```

The `AgentGuardrailTimeoutInfo` fields:

| Field | Type | Description |
|---|---|---|
| `guardrail_name` | `str` | Name of the guardrail that timed out. |
| `agent_name` | `str` | Name of the agent whose guardrail timed out. |
| `timeout` | `float` | The timeout duration in seconds that was exceeded. |
| `policy` | `AgentTimeoutPolicy` | The policy that was applied (`FAIL` or `PASS`). |

### Best Practices

- Set `timeout` on any guardrail that calls an external service or LLM.
- Use `FAIL` for security-critical guardrails (jailbreak, content policy) and `PASS` for audit/enrichment guardrails.
- Always provide `on_timeout` in production to track timeout rates in your metrics system.
- Set timeouts shorter than your user-facing SLA — a guardrail that times out should not cause the entire request to exceed its deadline.

## Remediation

Output guardrails support a self-correction loop. When a guardrail trips and the `remediation` attribute is set, the runner injects the remediation message as feedback and re-runs the agent instead of raising immediately. This gives the agent a chance to correct itself.

### How It Works

1. All output guardrails run in parallel on the agent's output.
2. If a guardrail trips and has `remediation` set (and the runner provides an `on_remediate` callback — which it always does), the runner calls the callback with the remediation message.
3. The callback re-runs the agent with the remediation feedback injected as a user message.
4. The new output is passed through all output guardrails again.
5. If the guardrail trips again and the attempt count is within `max_retries`, steps 3–4 repeat.
6. After `max_retries` failed attempts, `AgentOutputGuardrailTripwireTriggered` is raised.

Remediation attempts are tracked per guardrail name. If two different guardrails both have remediation and both trip on the same response, each has its own independent retry budget.

### Code Example

```python
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentOutputGuardrailData,
    agent_output_guardrail,
)


@agent_output_guardrail(
    remediation="Your response contained personal information (names, emails, phone numbers, or SSNs). Please regenerate your answer without including any personal data.",
    max_retries=2,
)
async def pii_output_check(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    output_text = str(data.output)
    has_pii = await pii_detector.scan(output_text)
    return AgentGuardrailFunctionOutput(
        tripwire_triggered=has_pii,
        output_info={"pii_found": has_pii},
    )


agent = Agent(
    name="Assistant",
    system_prompt="Help users with questions. Never include personal data in responses.",
    guardrails=AgentGuardrails(output=[pii_output_check]),
)
```

If the first response contains PII, the runner re-prompts the agent with the remediation message. If the second and third responses also contain PII (exhausting `max_retries=2`), `AgentOutputGuardrailTripwireTriggered` is raised.

### When Remediation Is Not Invoked

Remediation requires both conditions to be true:

1. `AgentOutputGuardrail.remediation` is set (non-None).
2. The runner provides an `on_remediate` callback (always the case when running via `Runner.arun` or `Runner.run`).

If `remediation` is `None` (the default), or if `max_retries` is `0`, the guardrail trips immediately without any retry. Setting `max_retries=0` effectively disables remediation even when `remediation` is set.

```python
# No remediation — trips immediately
@agent_output_guardrail
async def strict_check(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...

# Remediation disabled by max_retries=0
@agent_output_guardrail(remediation="Fix this.", max_retries=0)
async def disabled_remediation(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...
```

## Config Guardrails

Global guardrails can be applied to every agent in a run via `RunConfig.input_guardrails` and `RunConfig.output_guardrails`. This is the primary mechanism for organization-wide safety policies.

Config guardrails merge with agent-level guardrails and always run first. An agent's own guardrails run after config guardrails in both the blocking and parallel phases.

```python
from troopai.adk.run import RunConfig, Runner
from troopai.adk.agents.agent_guardrails import agent_input_guardrail, agent_output_guardrail


@agent_input_guardrail(run_in_parallel=False)
async def global_jailbreak_filter(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    # Applied to every agent in every run using this config
    ...


@agent_output_guardrail
async def global_content_filter(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    # Applied to every agent's output in runs using this config
    ...


config = RunConfig(
    guardrails=AgentGuardrails(
        input=[global_jailbreak_filter],
        output=[global_content_filter],
    ),
)

result = await Runner.arun(agent, "Hello!", run_config=config)
```

In a multi-agent workflow with handoffs, config guardrails apply to every agent that runs — the triage agent, the specialist agents, all of them. This makes `RunConfig` guardrails the right place for enterprise-wide policies.

The `run_in_parallel` flag on config guardrails is respected. Setting `run_in_parallel=False` on a config guardrail ensures it gates the LLM call for every agent in the run.

## Result Types

Every guardrail that runs produces a result object. These are collected on `RunResult` and on exceptions.

### AgentInputGuardrailResult

Returned for each input guardrail that completed, whether it tripped or passed.

| Field | Type | Description |
|---|---|---|
| `guardrail` | `AgentInputGuardrail[Any]` | The guardrail instance that ran. |
| `agent` | `Agent` | The agent whose input was checked. |
| `guardrail_output` | `AgentGuardrailFunctionOutput` | The verdict — `tripwire_triggered`, `severity`, `output_info`. |

### AgentOutputGuardrailResult

Returned for each output guardrail that completed.

| Field | Type | Description |
|---|---|---|
| `guardrail` | `AgentOutputGuardrail[Any]` | The guardrail instance that ran. |
| `agent` | `Agent` | The agent that produced the checked output. |
| `agent_output` | `Any` | The specific output that was validated by this guardrail. |
| `guardrail_output` | `AgentGuardrailFunctionOutput` | The verdict — `tripwire_triggered`, `severity`, `output_info`. |

### Accessing Results on RunResult

```python
result = await Runner.arun(agent, "Hello!")

# Input guardrail results — blocking + parallel combined, in execution order
for r in result.guardrail_results.input:
    print(f"Guardrail: {r.guardrail.get_name()}")
    print(f"Tripped:   {r.guardrail_output.tripwire_triggered}")
    print(f"Severity:  {r.guardrail_output.severity}")
    print(f"Info:      {r.guardrail_output.output_info}")

# Output guardrail results
for r in result.guardrail_results.output:
    print(f"Guardrail:   {r.guardrail.get_name()}")
    print(f"Agent:       {r.agent.name}")
    print(f"Output was:  {r.agent_output}")
    print(f"Tripped:     {r.guardrail_output.tripwire_triggered}")
```

Both `guardrail_results.input` and `guardrail_results.output` are tuples — immutable once the run completes. They include results for all severities, including non-halting `INFO` and `WARNING` verdicts.

### Results on Exceptions

When a guardrail trips and raises an exception, the exception carries both the triggering result and all results collected before it fired.

```python
try:
    result = await Runner.arun(agent, "suspicious input")
except AgentInputGuardrailTripwireTriggered as exc:
    # The specific result that triggered the exception
    triggering = exc.guardrail_result
    print(f"Triggered by: {triggering.guardrail.get_name()}")
    print(f"Output info:  {triggering.guardrail_output.output_info}")

    # All results collected before (and including) the triggering guardrail
    # Useful when multiple blocking guardrails ran before the one that tripped
    for r in exc.all_results:
        print(f"  {r.guardrail.get_name()}: {r.guardrail_output.tripwire_triggered}")

except AgentOutputGuardrailTripwireTriggered as exc:
    triggering = exc.guardrail_result
    print(f"Triggered by: {triggering.guardrail.get_name()}")
    print(f"Agent output was: {triggering.agent_output}")
    for r in exc.all_results:
        print(f"  {r.guardrail.get_name()}: {r.guardrail_output.tripwire_triggered}")
```

## Guardrail Audit Side-Car

Every guardrail that runs — across agent input, agent output, tool input, tool
output, and flow pre/post steps — is automatically recorded in
`RunResult.guardrail_audit`. The records are immutable and privacy-preserving:
they store SHA-256 hashes of the checked artifact and any replacement, never
the raw text, so the audit log cannot become a secondary sink for the very PII
a guardrail is meant to catch.

```python
result = await Runner.arun(agent, prompt)

for record in result.guardrail_audit:
    print(record.level, record.guardrail_name, record.action, record.triggered)
    if record.transformed_hash is not None:
        # A transform was applied — the output was rewritten
        print(f"  original hash: {record.output_hash}")
        print(f"  replacement hash: {record.transformed_hash}")
```

`GuardrailAuditRecord` fields:

| Field | Type | Description |
|---|---|---|
| `level` | `GuardrailAuditLevel` | Which surface produced the record: `"agent_input"`, `"agent_output"`, `"tool_input"`, `"tool_output"`, `"flow_pre"`, `"flow_post"`. |
| `guardrail_name` | `str` | The guardrail's name. |
| `agent_name` | `str \| None` | The agent the guardrail ran for, or `None`. |
| `action` | `GuardrailAction` | The action actually taken: `PASS`, `RAISE`, or `TRANSFORM`. |
| `severity` | `AgentGuardrailSeverity \| None` | Agent-level severity when set; `None` at tool/flow levels. |
| `triggered` | `bool` | `True` when `action` is anything other than `PASS`. |
| `output_hash` | `str \| None` | SHA-256 hex of the checked artifact, or `None` when nothing was hashed. |
| `transformed_hash` | `str \| None` | SHA-256 hex of the replacement; set only for a transform, else `None`. A differing pair marks a substitution. |
| `changed_spans` | `tuple[GuardrailSpan, ...]` | Observability ranges the guardrail reported; empty when none. |
| `timestamp` | `datetime` | UTC time the record was created. |

The records are populated by `run/governance.py:emit_guardrail_audit`, which is
called inside the guardrail executor for every guardrail that completes —
passing or tripping. The collection is attached to `RunResult.guardrail_audit`
as an immutable tuple once the run finishes.

## Hooks

The `RunHooks` class provides four guardrail lifecycle callbacks for observability, latency tracking, and audit logging.

```python
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.agents.agent_guardrails import AgentInputGuardrailResult, AgentOutputGuardrailResult
from troopai.adk.run.context import RunContext
import time


class GuardrailObserver(RunHooks):
    def __init__(self):
        self._start_times: dict[str, float] = {}

    async def on_input_guardrail_start(
        self,
        context: RunContext,
        agent,
        guardrail_name: str,
    ) -> None:
        """Called before each input guardrail runs."""
        self._start_times[guardrail_name] = time.monotonic()
        print(f"[guardrail] Input '{guardrail_name}' starting for agent '{agent.name}'")

    async def on_input_guardrail_end(
        self,
        context: RunContext,
        agent,
        result: AgentInputGuardrailResult,
    ) -> None:
        """Called after each input guardrail completes (pass or trip)."""
        name = result.guardrail.get_name()
        elapsed = time.monotonic() - self._start_times.get(name, 0)
        tripped = result.guardrail_output.tripwire_triggered
        print(f"[guardrail] Input '{name}' done in {elapsed:.2f}s (tripped={tripped})")

    async def on_output_guardrail_start(
        self,
        context: RunContext,
        agent,
        guardrail_name: str,
    ) -> None:
        """Called before each output guardrail runs."""
        self._start_times[guardrail_name] = time.monotonic()
        print(f"[guardrail] Output '{guardrail_name}' starting for agent '{agent.name}'")

    async def on_output_guardrail_end(
        self,
        context: RunContext,
        agent,
        result: AgentOutputGuardrailResult,
    ) -> None:
        """Called after each output guardrail completes (pass or trip)."""
        name = result.guardrail.get_name()
        elapsed = time.monotonic() - self._start_times.get(name, 0)
        tripped = result.guardrail_output.tripwire_triggered
        print(f"[guardrail] Output '{name}' done in {elapsed:.2f}s (tripped={tripped})")


result = await Runner.arun(agent, "Hello!", hooks=GuardrailObserver())
```

Hook signatures:

| Hook | Signature | Fires |
|---|---|---|
| `on_input_guardrail_start` | `(context, agent, guardrail_name: str)` | Before each input guardrail runs |
| `on_input_guardrail_end` | `(context, agent, result: AgentInputGuardrailResult)` | After each input guardrail completes |
| `on_output_guardrail_start` | `(context, agent, guardrail_name: str)` | Before each output guardrail runs |
| `on_output_guardrail_end` | `(context, agent, result: AgentOutputGuardrailResult)` | After each output guardrail completes |

Hooks fire for every guardrail — both passing and tripping ones. For parallel input guardrails, `on_input_guardrail_start` fires when each individual coroutine begins executing, not when the `gather` call is made. Hook ordering within a parallel batch is non-deterministic.

## Exception Handling

### AgentInputGuardrailTripwireTriggered

Raised when any input guardrail halts execution. Inherits from `GuardrailTripwireTriggered` and `TroopAIError`.

```python
from troopai.adk.exceptions import AgentInputGuardrailTripwireTriggered

try:
    result = await Runner.arun(agent, user_input)
except AgentInputGuardrailTripwireTriggered as exc:
    # exc.guardrail_result — AgentInputGuardrailResult for the guardrail that tripped
    name = exc.guardrail_result.guardrail.get_name()
    info = exc.guardrail_result.guardrail_output.output_info

    # exc.all_results — all AgentInputGuardrailResult objects collected before the trip
    # Includes the triggering result as the last entry
    audit_results = exc.all_results or []

    # exc.message — human-readable string: "Input guardrail '<name>' tripwire triggered"
    print(exc.message)

    return {"error": "request_rejected", "guardrail": name, "details": info}
```

### AgentOutputGuardrailTripwireTriggered

Raised when any output guardrail halts execution (and remediation is exhausted, if configured). Inherits from `GuardrailTripwireTriggered` and `TroopAIError`.

```python
from troopai.adk.exceptions import AgentOutputGuardrailTripwireTriggered

try:
    result = await Runner.arun(agent, user_input)
except AgentOutputGuardrailTripwireTriggered as exc:
    # exc.guardrail_result — AgentOutputGuardrailResult for the guardrail that tripped
    name = exc.guardrail_result.guardrail.get_name()
    agent_output = exc.guardrail_result.agent_output
    info = exc.guardrail_result.guardrail_output.output_info

    # exc.all_results — all AgentOutputGuardrailResult objects from the final pass
    # Includes results from parallel guardrails that passed, not just the one that tripped
    passing = [r for r in (exc.all_results or []) if not r.guardrail_output.tripwire_triggered]

    print(f"Output rejected by '{name}' after {len(passing)} passing guardrails")
    return {"error": "output_rejected", "guardrail": name}
```

### Exception Hierarchy

```
TroopAIError
└── GuardrailTripwireTriggered
    ├── AgentInputGuardrailTripwireTriggered
    │       .guardrail_result   AgentInputGuardrailResult
    │       .all_results        list[AgentInputGuardrailResult] | None
    │       .message            str
    └── AgentOutputGuardrailTripwireTriggered
            .guardrail_result   AgentOutputGuardrailResult
            .all_results        list[AgentOutputGuardrailResult] | None
            .message            str
```

## Decorators Reference

### @agent_input_guardrail

Creates an `AgentInputGuardrail` from a function. Supports two call styles.

**Without arguments** (function passed directly):

```python
@agent_input_guardrail
async def my_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...
```

**With keyword arguments** (returns a decorator):

```python
@agent_input_guardrail(
    name="custom_name",          # Override guardrail name (default: function.__name__)
    run_in_parallel=False,       # Block before agent starts (default: True)
    timeout=5.0,                 # Seconds before timeout fires (default: None)
    timeout_policy=AgentTimeoutPolicy.PASS,  # FAIL or PASS on timeout (default: FAIL)
    on_timeout=my_callback,      # Async callback on timeout (default: None)
)
async def my_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `Optional[str]` | `None` | Guardrail name. Falls back to `function.__name__`. |
| `run_in_parallel` | `bool` | `True` | Whether to run alongside the agent (`True`) or block before it (`False`). |
| `timeout` | `Optional[float]` | `None` | Execution timeout in seconds. No timeout if `None`. |
| `timeout_policy` | `AgentTimeoutPolicy` | `AgentTimeoutPolicy.FAIL` | Behavior on timeout. `FAIL` halts, `PASS` continues silently. |
| `on_timeout` | `Optional[Callable]` | `None` | Async callback `(AgentGuardrailTimeoutInfo) -> None` invoked on timeout. |

### @agent_output_guardrail

Creates an `AgentOutputGuardrail` from a function.

**Without arguments:**

```python
@agent_output_guardrail
async def my_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...
```

**With keyword arguments:**

```python
@agent_output_guardrail(
    name="custom_name",
    remediation="Please regenerate without including personal data.",
    max_retries=2,
    timeout=10.0,
    timeout_policy=AgentTimeoutPolicy.FAIL,
    on_timeout=my_callback,
)
async def my_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `Optional[str]` | `None` | Guardrail name. Falls back to `function.__name__`. |
| `remediation` | `Optional[str]` | `None` | Feedback message injected as user turn when guardrail trips. Triggers self-correction retry loop. |
| `max_retries` | `int` | `1` | Maximum remediation attempts before raising. Only meaningful when `remediation` is set. |
| `timeout` | `Optional[float]` | `None` | Execution timeout in seconds. No timeout if `None`. |
| `timeout_policy` | `AgentTimeoutPolicy` | `AgentTimeoutPolicy.FAIL` | Behavior on timeout. `FAIL` halts, `PASS` continues silently. |
| `on_timeout` | `Optional[Callable]` | `None` | Async callback `(AgentGuardrailTimeoutInfo) -> None` invoked on timeout. |

## Attribute Reference

### AgentInputGuardrail

```python
from troopai.adk.agents.agent_guardrails import AgentInputGuardrail
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `guardrail_function` | `Callable[[AgentInputGuardrailData], AgentGuardrailFunctionOutput]` | required | The validation function. Sync and async both supported. |
| `name` | `Optional[str]` | `None` | Display name. Falls back to `guardrail_function.__name__` via `get_name()`. |
| `run_in_parallel` | `bool` | `True` | `True` — races alongside the agent loop. `False` — blocks before agent starts. |
| `timeout` | `Optional[float]` | `None` | Timeout in seconds. Wraps execution in `asyncio.wait_for()`. |
| `timeout_policy` | `AgentTimeoutPolicy` | `AgentTimeoutPolicy.FAIL` | Behavior when `timeout` fires. |
| `on_timeout` | `Optional[Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]]]` | `None` | Async side-effect callback on timeout. Runs after policy is applied. |

### AgentOutputGuardrail

```python
from troopai.adk.agents.agent_guardrails import AgentOutputGuardrail
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `guardrail_function` | `Callable[[AgentOutputGuardrailData], AgentGuardrailFunctionOutput]` | required | The validation function. Sync and async both supported. |
| `name` | `Optional[str]` | `None` | Display name. Falls back to `guardrail_function.__name__` via `get_name()`. |
| `remediation` | `Optional[str]` | `None` | Feedback message for agent self-correction. When set, triggers retry loop on trip. |
| `max_retries` | `int` | `1` | Maximum self-correction attempts before raising. Only meaningful when `remediation` is set. |
| `timeout` | `Optional[float]` | `None` | Timeout in seconds. Wraps execution in `asyncio.wait_for()`. |
| `timeout_policy` | `AgentTimeoutPolicy` | `AgentTimeoutPolicy.FAIL` | Behavior when `timeout` fires. |
| `on_timeout` | `Optional[Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]]]` | `None` | Async side-effect callback on timeout. Runs after policy is applied. |

### AgentGuardrailFunctionOutput

```python
from troopai.adk.agents.agent_guardrails import AgentGuardrailFunctionOutput
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `output_info` | `Any` | `None` | Arbitrary metadata about the check — detection details, scores, matched rules. Included in result objects and exception payloads. |
| `tripwire_triggered` | `bool` | `False` | Whether a violation was detected. Authoritative when `severity` is `None`. Ignored for halt decisions when `severity` is set. Must be `True` when `transformed_output` is set (halt fallback). |
| `severity` | `Optional[AgentGuardrailSeverity]` | `None` | When set, overrides `tripwire_triggered` for the halt decision. `INFO` and `WARNING` never halt; `ERROR` always halts. Must not be a non-halting value when `transformed_output` is set. |
| `transformed_output` | `Any` | `None` | Complete replacement for the checked output (output guardrails, text outputs only). When non-`None`, the runner substitutes it for `RunResult.final_output` and rewrites the trailing assistant message. The guardrail must also set `tripwire_triggered=True`. |
| `changed_spans` | `Optional[list[GuardrailSpan]]` | `None` | Character ranges the guardrail flagged, for audit and tracing only. Never read by the runner to construct or apply a transform. |

## Best Practices

**Use blocking mode for cheap checks.** Keyword matching, regex patterns, and simple rule evaluations are fast. Setting `run_in_parallel=False` on these guardrails means a violation caught before the agent starts costs zero LLM tokens — exactly the right tradeoff for high-precision, low-cost checks.

**Use parallel mode for expensive checks.** LLM-based classifiers, external API calls, and embedding similarity checks take time. Running them in parallel with the agent amortizes their latency on the happy path. The cost is that tokens are spent on the agent even if the guardrail would have tripped — acceptable for low-violation-rate scenarios.

**Use severity for monitoring without blocking.** Not every detected pattern warrants halting execution. A 60%-confidence PII detection may be worth logging without blocking. Return `AgentGuardrailSeverity.WARNING` or `AgentGuardrailSeverity.INFO` on uncertain verdicts and reserve `ERROR` (or plain `tripwire_triggered=True`) for high-confidence violations. This lets you tune thresholds over time by analyzing the logged results.

**Always set timeouts on LLM-based guardrails.** A guardrail that calls an external classification API or makes its own LLM call can hang indefinitely. A hung guardrail blocks the user's request. Set `timeout` to a value shorter than your user-facing SLA and choose `timeout_policy` based on the guardrail's safety role: `FAIL` for security gates, `PASS` for enrichment and monitoring.

**Use TRANSFORM for deterministic output redaction.** When the set of patterns
to mask is known and deterministic, prefer `transformed_output` over
`remediation`. A transform substitutes the replacement immediately — no
extra LLM turn, no additional token cost — and is guaranteed to mask exactly
what the guardrail's regex found. Remediation is better suited for content
violations that require the model's judgment to correct. Always pair a
transform with non-streaming delivery when hard PII guarantees are required.

**Use remediation for output PII scrubbing.** When the pattern is not known in
advance or the correction requires judgment (e.g. "remove names without
breaking sentence flow"), configure `remediation` with a clear instruction and
`max_retries=2`. The agent gets a chance to self-correct, the user never sees
the error, and only persistent violations raise an exception. Keep remediation
messages specific — "Remove names, email addresses, and phone numbers" works
better than "Remove PII".

**Use config guardrails for organization-wide policies.** Security gates that must apply to every agent in every workflow belong in `RunConfig.input_guardrails` and `RunConfig.output_guardrails`. This prevents individual agent developers from accidentally bypassing org-wide safety requirements and centralizes policy updates.

**Prefer `output_info` for structured metadata.** Pass a dict or dataclass as `output_info` rather than embedding diagnostic data in a message string. This makes results machine-readable for downstream audit pipelines and makes exception handling easier.

**Layer your guardrails.** A robust guardrail stack typically combines:
1. A fast blocking input guardrail for obvious violations (regex, keyword list).
2. A parallel input guardrail using an LLM classifier for nuanced detection.
3. One or more output guardrails with remediation for PII and compliance.
4. A config-level guardrail for organization-wide baseline enforcement.

```python
agent = Agent(
    name="Production Agent",
    system_prompt="...",
    guardrails=AgentGuardrails(
        input=[
            keyword_blocker,        # run_in_parallel=False — fast, saves tokens
            llm_safety_classifier,  # run_in_parallel=True  — slow, amortized
        ],
        output=[
            pii_output_check,       # remediation set — self-corrects
            compliance_check,       # remediation=None — hard block
        ],
    ),
)

config = RunConfig(
    guardrails=AgentGuardrails(
        input=[org_jailbreak_policy],   # Applies to every agent
        output=[org_content_policy],    # Applies to every agent
    ),
)
```
