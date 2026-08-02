---
name: add-hosted-tool
description: Procedure to add a new provider-hosted tool (web search, code exec, file search, etc.) as a typed HostedTool subclass and wire every provider converter. Use when surfacing a provider-native capability as a framework tool.
---

# Add a Provider-Hosted Tool

Constraints live in `.claude/rules/tools-guardrails.md` (loads when you
edit `tools/`). This is the ordered procedure.

## 0. Decide it earns a class

Add a subclass ONLY when ≥2 providers ship the capability natively, OR 1
provider ships it with typed knobs worth surfacing. NEVER for beta-only
shapes or zero-knob single-provider markers — those use
`LLMConfig.extra_body`.

## 1. Define the subclass

In `src/troopai/adk/tools/hosted/` (base: `hosted_tool.py`), add the
concrete class:

- `@dataclass(kw_only=True)`, inheriting `HostedTool`.
- `SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]]` from
  `{"anthropic", "openai-responses", "openai-chatcompletions", "gemini",
  "litellm"}`.
- Class docstring with a "Provider matrix" section.
- Concept name, never a provider name (`WebSearchTool`, not
  `AnthropicWebSearchTool`).
- Attributes that apply to only some providers tagged
  `**<Provider> only.**` / `**<P1> + <P2>.**` in their docstring.

## 2. Wire EVERY provider converter

For each provider in `llms/<provider>/<provider>_converter.py`
`convert_tools`:

- If supported: translate to the matching wire param, reading only the
  attributes that provider honours (`logger.debug` the ignored ones).
- If unsupported: `raise UnsupportedHostedToolError(tool,
  "<provider-id>", supported_providers=tool.SUPPORTED_PROVIDERS)`.
- **Silent drops are forbidden** — every converter handles every
  concrete subclass explicitly.

## 3. Response side

If the capability emits provider-native output items, ensure a matching
response-side `@dataclass(frozen=True)` exists in `types/tools/` and is
carried by the right RunItem (see the `add-run-item` skill).

## 4. Complete + verify

Tests in `tests/unit/tools/` covering each provider's translate-or-raise
path; update the provider matrix table in
`.claude/rules/tools-guardrails.md`; doc + example. Run the `code-hygiene-gate` skill.
No `NotImplementedError`. Surface missing credentials when running the
example — never mark verified without a real run.
