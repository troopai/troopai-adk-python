# Agent Harness vs Agent Governance

## The Core Distinction

The difference is not about scale or automation. It is about mindset.

**Agent harness** is reactive. It assumes agents will misbehave and the developer's job is to catch it. Every guardrail, every HITL gate, every usage limit is a response to a potential failure. The question is always: *what can go wrong, and how do I stop it?*

**Agent governance** is proactive. It defines how things should work and makes the correct behavior the default — or the only option. The question is: *here is exactly how we do things here.*

The difference is subtle but changes everything:

| | Harness (Reactive) | Governance (Proactive) |
|---|---|---|
| **Mindset** | "What you can't do" | "Here is how we do things here" |
| **Mechanism** | Catch violations after they happen | Make violations impossible by design |
| **Scaling** | Linear — more agents = more things to watch | Sublinear — define policy once, all agents comply |
| **Developer burden** | Exhausting — constant monitoring | Sustainable — system enforces correctness |
| **Analogy** | Security guard checking every person | Building codes that make unsafe construction impossible |

---

## The Kubernetes Lesson

Kubernetes scales insanely well because of one principle: **declarative desired state with automatic reconciliation**.

You never say "start 3 pods." You say "I want 3 replicas of this service." The system continuously reconciles actual state to desired state. You sleep at night because the SYSTEM governs, not you.

| K8s Concept | What it does | Agent Governance Equivalent |
|-------------|-------------|----------------------------|
| **Deployment spec** | Declares desired state | **Agent policy** — "this agent responds in under 200 words, cites sources, uses professional tone" |
| **Controller** | Watches actual state, reconciles to desired | **Runner** — observes agent behavior, enforces policy |
| **ResourceQuota** | Makes overspending impossible | **Budget** — sub-agent can use max 10K tokens |
| **LimitRange** | Default resource bounds for all pods | **Fleet defaults** — every sub-agent gets 30s timeout |
| **NetworkPolicy** | Defines which pods can communicate | **Delegation policy** — which agents can delegate to which |
| **Admission Controller** | Shapes requests before they enter the system | **Policy injection** — not catching bad output, but shaping correct input |
| **Labels + Selectors** | Grouping and targeting | **Agent metadata** — description, tags, routing |

The key insight: K8s `ResourceQuota` doesn't catch overspending — it makes overspending impossible. K8s `NetworkPolicy` doesn't monitor unauthorized traffic — it prevents it from being routed. This is governance by design, not governance by detection.

---

## How the ADK Provides Both

The TroopAI Agents ADK supports both harness and governance. Both are needed — governance is the primary mechanism, harness is the safety net. Like Kubernetes has both admission controllers (proactive) and liveness probes (reactive).

### Governance Features (Proactive)

These define the correct behavior upfront. Agents follow them by design.

| Primitive | Where | What it prescribes |
|-----------|-------|-------------------|
| `SystemPrompt` / `DynamicSystemPrompt` | `Agent.system_prompt` | Agent behavior, tone, constraints, operating procedures. The primary governance mechanism. |
| `Agent.description` | `Agent.description` | What this agent does — flows to `as_tool()` and handoffs. Tells the supervisor LLM exactly when to delegate. |
| `Agent.output_schema` | `Agent.output_schema` | Defines what correct output looks like. Not a validation — a structural requirement. The LLM produces this shape or nothing. |
| `ToolUseBehavior` | `Agent.tool_use_behavior` | Defines what happens after tool execution. `"stop_on_first_tool"` means the tool result IS the output — no LLM rewrite. |
| `HandoffConfig.strategy` | `Handoff.config` | Defines exactly what context flows to the next agent. `"intent_only"` means the target gets only the intent, not the full conversation. |
| `as_tool(budget=...)` | `Agent.as_tool()` | Resource allocation per delegation. Not a limit that catches overspending — a budget that makes overspending impossible. |
| `as_tool(timeout=...)` | `Agent.as_tool()` | Time allocation per delegation. The sub-agent run is bounded by `asyncio.wait_for()`. |

### Harness Features (Reactive)

These catch problems that slip past governance. The safety net, not the primary mechanism.

| Primitive | Where | What it catches |
|-----------|-------|----------------|
| `AgentInputGuardrail` | `Agent.input_guardrails` | Bad input that the system prompt alone can't prevent. PII leaks, jailbreaks, off-topic requests. |
| `AgentOutputGuardrail` | `Agent.output_guardrails` | Bad output that the output schema can't enforce. Content policy violations, hallucinated data. |
| `requires_approval` | `FunctionTool` | Dangerous tool calls that need human judgment. A reactive gate — the agent already decided to act. |
| `LLMUsageLimits` | `RunConfig.usage_limits` | Token/request overspend. A hard cap when budget-by-design isn't sufficient. |
| `max_turns` / `max_total_turns` | `Runner` / `RunConfig` | Infinite loops. A circuit breaker when the agent can't converge. |
| `FunctionTool.max_retries` | `FunctionTool` | Broken tools. Removes a failing tool from the LLM's view after N failures. |
| `RunHooks` | `Runner.arun(hooks=...)` | Observability — what agents are doing, when, at what cost. Needed when governance isn't enough and you need to debug. |

### Why Both Matter

Governance without harness is naive — even well-designed systems have edge cases. System prompts can be circumvented. Output schemas can be satisfied with garbage. Budgets can be consumed on useless work.

Harness without governance is exhausting — you're constantly reacting to problems instead of preventing them. Every new agent means more guardrails to write, more HITL gates to configure, more dashboards to watch.

**The right architecture**: Governance handles the 95% case by making correct behavior the default. Harness handles the 5% where governance alone isn't sufficient.

---

## Feature Classification

Every ADK primitive classified by its role:

| ADK Primitive | Governance (Proactive) | Harness (Reactive) | Notes |
|--------------|:---------------------:|:-----------------:|-------|
| `SystemPrompt` | Yes | | Primary governance. Defines behavior, tone, SOPs. |
| `DynamicSystemPrompt` | Yes | | Context-aware governance. Runtime policy adaptation. |
| `Agent.description` | Yes | | Routing signal for supervisor LLMs. |
| `Agent.output_schema` | Yes | | Structural output requirement. |
| `ToolUseBehavior` | Yes | | Post-tool execution policy. |
| `HandoffConfig` | Yes | | Context transfer policy. |
| `as_tool(budget=...)` | Yes | | Resource allocation per delegation. |
| `as_tool(timeout=...)` | Yes | | Time allocation per delegation. |
| `FunctionTool.max_result_tokens` | Yes | | Output size policy. |
| `AgentInputGuardrail` | | Yes | Catches bad input. |
| `AgentOutputGuardrail` | | Yes | Catches bad output. |
| `requires_approval` (HITL) | | Yes | Human gate on dangerous actions. |
| `LLMUsageLimits` | | Yes | Hard cap on token spend. |
| `max_turns` / `max_total_turns` | | Yes | Loop circuit breaker. |
| `FunctionTool.max_retries` | | Yes | Broken tool circuit breaker. |
| `RunHooks` | | Yes | Observability for debugging. |
| `Context compaction` | | Yes | Reactive context overflow management. |
| `can_use_tool` | Yes | Yes | Both: defines access policy (governance) and enforces it (harness). |

---

## The Governance Roadmap

The ADK provides governance building blocks today. These are the primitives a declarative governance layer can be built on:

### Agent Policy (future)

A structured declaration of how an agent should operate. Not a system prompt (LLM-interpreted text) but a machine-enforceable specification that the Runner applies proactively.

Inspiration: K8s Deployment spec + PodSecurityPolicy. You don't watch every pod — you declare what "secure" means, and the admission controller enforces it.

### Delegation Contracts (future)

When agent A delegates to agent B, a contract defines the input format, expected output format, quality criteria, and resource allocation. Currently this is string in / string out with optional schemas. Contracts would make delegation expectations explicit and measurable.

Inspiration: K8s Service + Ingress. Services declare their interface; ingress defines access rules. Consumers and producers agree on a contract.

### Fleet Defaults (future)

Default governance applied to all agents in a run without per-agent configuration. Every sub-agent gets a timeout, every delegation gets a budget, every output gets a quality standard — unless explicitly overridden.

Inspiration: K8s LimitRange. Every pod in a namespace gets default resource limits without the developer specifying them on each pod.

---

## Summary

| Layer | Question | Scales | When to use |
|-------|----------|--------|-------------|
| **Governance** | "How should this agent behave?" | Sublinearly | Always. The primary mechanism. |
| **Harness** | "What if governance isn't enough?" | Linearly | When you need a safety net for edge cases. |

Start with governance. Add harness where governance alone is insufficient. Never rely on harness alone — it is exhausting at scale and assumes failure rather than preventing it.
