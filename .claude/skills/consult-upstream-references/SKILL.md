---
name: consult-upstream-references
description: Consult the closest upstream reference (OpenAI Agents SDK, Guardrails AI, Google ADK, CrewAI, LangChain/LangGraph, LiteLLM, Claude Agent SDK) before designing a subsystem or debugging hard framework behavior — read a fresh source, cite it, then adapt to this ADK. Use instead of guessing how a feature should work.
---

# Consult Upstream References

This ADK is built by adapting proven designs from upstream agent frameworks.
Before you design a subsystem or chase a hard framework bug, read the closest
upstream reference fresh — do not guess from memory, and do not adopt upstream
types or patterns wholesale. Reference, then adapt to this ADK's invariants.

## 1. When to consult

- Designing or substantially changing a subsystem (runner, graphs, swarms,
  handoffs, providers, sandbox, sessions, memory, tools).
- Chasing a hard bug whose root cause is "how is this framework meant to
  behave?".
- Any time you would otherwise guess at an API shape, a default, or control
  flow. A fresh read beats a guess.

## 2. The map — subsystem → closest upstream

| This ADK | Closest upstream reference |
|---|---|
| Runner, agent loop, handoffs, tool-use, guardrail control flow | OpenAI Agents SDK (`openai/openai-agents-python`) |
| Guardrail validators + `on_fail`/action model (block / fix / transform / span-redaction), audit | Guardrails AI (`guardrails-ai/guardrails`) — study the `Validator` / `FailResult.fix_value` / `OnFailAction` model, then **adapt to this ADK's invariants; never copy wholesale** |
| Graphs, state-machine orchestration | LangGraph (`langchain-ai/langgraph`) |
| Swarms, multi-agent collaboration | CrewAI (`crewAIInc/crewAI`) |
| Provider / model / parameter mapping | LiteLLM (`BerriAI/litellm`) |
| Anthropic-native (sandbox, memory, hosted tools, caching, thinking) | Claude Agent SDK + Anthropic SDK |
| Sessions, runtime, agent-config patterns | Google ADK (`google/adk-python`) |

## 3. How to read a fresh source (preference order)

1. **context7 MCP** — current, version-aware docs. Call `resolve-library-id`
   for the library, then `query-docs` with your specific question. Best for
   API and usage.
2. **The installed SDK in site-packages** — the most accurate source for the
   versions this repo actually pins:
   ```bash
   python -c "import agents, os; print(os.path.dirname(agents.__file__))"
   ```
   then Read/Grep the typed client. (`add-llm-provider` step 0 already reads
   the installed SDK for providers.)
3. **Web-fetch raw GitHub / docs** (Claude `WebFetch`, Kimi `FetchURL`) —
   `raw.githubusercontent.com/<org>/<repo>/main/<path>` for source; the
   LangChain and CrewAI documentation domains are already allowlisted.

## 4. Cite, then adapt

- **Cite the specific source you read** (repo path or URL + what you took) in
  the design / plan / PR / commit — a fresh read, not a training-data memory.
- **Adapt to this ADK's invariants** — the three type layers, no implicit
  behavior, cost-conservative defaults, named parameters. Translate the idea
  into this codebase's shape rather than importing the upstream's.
- **Never adopt upstream types wholesale.** For example, the standing
  prohibition on OpenAI's `Model` / `AnyLLMModel` ABC: this ADK's `LLM` ABC is
  framework-owned and typed against framework types.
