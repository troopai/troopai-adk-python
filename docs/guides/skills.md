(guides/skills)=

# 🎒 Skills

A skill is a **composable bundle** of instructions, tools, resources, and
governance that you attach to an `Agent`. Where a tool is a single Python
callable, a skill is a full capability stack — domain expertise (markdown
instructions), the functions that realise it (tools), the policies that
govern those functions (guardrails, timeouts, retry budgets), and optional
reference material (resources).

Skills implement the composition-over-inheritance principle: building a
"research-capable agent" means attaching a `ResearchSkill`, not subclassing
`Agent`.

---

## Anatomy of a skill

`Skill` is a plain dataclass under `src/troopai/adk/skills/skill.py`.
Every field is shown below alongside its purpose.

| Field | Type | Purpose |
|---|---|---|
| `name` | `str` | Unique identifier. Required. |
| `description` | `str` | Short summary used by discovery tools and `SkillSet.find()`. |
| `instructions` | `str \| None` | Markdown injected into the agent's system prompt on activation. |
| `tools` | `list[Tool]` | Tools merged into the agent's tool list on activation. |
| `guardrails` | `ToolGuardrails \| None` | Skill-level input/output guardrails prepended to each tool's own guardrails. |
| `enabled` | `bool \| Callable[[RunContext], bool]` | Static flag or dynamic predicate that controls whether the skill is active. |
| `metadata` | `SkillMetadata \| None` | Version, author, tags, license — used for discovery and filtering. |
| `governance` | `SkillGovernance \| None` | Default `timeout`, `max_result_tokens`, and `max_retries` applied to every tool in the skill. |
| `resources` | `dict[str, str] \| None` | Named files or content strings accessible via `SkillDiscoveryToolset`. |
| `resource_root` | `Path \| None` | Absolute root that bounds resource path resolution (set by directory loaders). |

Two companion types carry optional detail:

- **`SkillMetadata(version, author, tags, license)`** — immutable, used for
  cataloguing and `SkillSet.filter_by_tag()`.
- **`SkillGovernance(timeout, max_result_tokens, max_retries)`** — values act
  as defaults; a tool's own setting always takes precedence.

---

## Minimal example

```python
from troopai.adk.agents import Agent
from troopai.adk.llms import LLMConfig
from troopai.adk.run import Runner
from troopai.adk.skills import Skill, SkillGovernance
from troopai.adk.tools import function_tool


@function_tool(name="fetch_article", description="Fetch the text of a web article by URL.")
def fetch_article(url: str) -> str:
    """Return article body (stub)."""
    return f"[article at {url}]"


@function_tool(name="summarise", description="Summarise a block of text in one paragraph.")
def summarise(text: str) -> str:
    """Return a one-paragraph summary (stub)."""
    return f"Summary of: {text[:80]}…"


research_skill = Skill(
    name="research",
    description="Web research and summarisation",
    instructions="""When researching a topic:
1. Fetch the most authoritative source with `fetch_article`.
2. Summarise the result with `summarise`.
3. Cite the source URL in your reply.""",
    tools=[fetch_article, summarise],
    governance=SkillGovernance(
        timeout=30.0,
        max_result_tokens=1024,
        max_retries=2,
    ),
)

agent = Agent(
    name="Research Assistant",
    system_prompt="You are a research assistant. Use your tools to answer questions.",
    skills=[research_skill],
    llm_config=LLMConfig(temperature=0.2),
)

result = Runner.run(agent, "Summarise the main claims in https://example.com/paper")
print(result.final_output)
```

`Agent(skills=[research_skill])` is the only declaration needed. The Runner
unpacks the skill at run time — merging `research_skill.tools` into the
agent's tool list and injecting `research_skill.instructions` into the system
prompt.

---

## Skill vs Tool

```{admonition} Quick rule
:class: tip

Reach for a **tool** when you need to expose a single function.
Reach for a **skill** when you need a full *capability stack* — instructions
that teach the LLM *how* to use the tools, governance that constrains them,
and guardrails that protect them.
```

| Aspect | `Skill` | `Tool` (`FunctionTool`) |
|---|---|---|
| Granularity | Bundle of related capabilities. | Single callable. |
| Instructions | Markdown injected into the system prompt. | Description string only. |
| Governance | `SkillGovernance` applied as defaults to every bundled tool. | Per-tool only. |
| Guardrails | Skill-level guardrails prepend each tool's own list. | Per-tool only. |
| Reuse | One `Skill` instance, many agents. | One `FunctionTool` per site unless explicitly shared. |
| Composition | Multiple skills per agent; skills may share tools. | Multiple tools per agent. |

A tool is a primitive; a skill is a package. A skill may contain zero tools
(instructions-only), many tools, or tools that a bare agent also carries.

For the full treatment see the Concepts page: {doc}`../concepts/index`.

---

## Skill activation strategies

The `SkillActivation` enum (exported from `troopai.adk.skills`) controls
*when* skill instructions enter the system prompt.

**`LAZY`** (default on `Agent.skill_activation`)

Instructions are injected only when one of the skill's tools is called for
the first time. Tool descriptions are always visible in the tool list, so
the LLM can discover them without any extra overhead. Turns that never call
a skill's tools pay nothing for that skill's instruction text.

```python
from troopai.adk.skills import SkillActivation

agent = Agent(
    name="Assistant",
    system_prompt="…",
    skills=[research_skill, billing_skill, safety_skill],
    skill_activation=SkillActivation.LAZY,  # default — no need to set explicitly
)
```

**`EAGER`**

All skill instructions are concatenated into the system prompt before the
first LLM call. Use this only when every skill is relevant to every turn —
for example an agent with a single, always-applicable skill.

```python
agent = Agent(
    name="Research Assistant",
    system_prompt="…",
    skills=[research_skill],
    skill_activation=SkillActivation.EAGER,
)
```

The LAZY default is intentional: it aligns with the cost-conservative
principle — agents never pay per-turn token cost for capabilities the LLM
didn't need.

---

## How the Runner composes skills

When a run starts the Runner calls two composition functions inside the loop:

1. **`resolve_system_prompt`** — For EAGER activation, it appends all enabled
   skill instructions under an `## Available Skills` heading. For LAZY, it
   excludes skill instructions at run start and tracks a `skill_tool_map`
   (tool name → skill name) for deferred injection.

2. **`build_tools`** — Merges skill tools with the agent's own tools.
   For each tool in each enabled skill, `SkillGovernance` values are applied
   as defaults (tool's own `timeout`, `max_result_tokens`, `max_retries`
   take precedence), and skill-level `ToolGuardrails` entries are prepended
   to each tool's guardrail list.

**Name conflicts:** if two skills declare a tool with the same name, the
Runner merges them in declaration order — the first tool with a given name
wins. Design skills so their tool names are scoped (e.g.
`research_fetch_article`, not `fetch`).

**Dynamic enablement:** `Skill.enabled` accepts a sync or async callable
`(RunContext) -> bool`. The Runner evaluates this at activation time, so
skills can be toggled by feature flags, tenant configuration, or run context.

**Lifecycle hook:** when LAZY activation fires on first tool call, the Runner
emits `on_skill_activated(context, agent, skill_name)` so hooks can record
telemetry or update audit logs without polling.

---

## Built-in skills

No skills ship by default. The framework ships the skill *machinery*
(`Skill`, `SkillSet`, `SkillActivation`, `SkillGovernance`,
`SkillDiscoveryToolset`) but no pre-built capability stacks. Every skill in
your application is user-defined, which keeps the surface minimal and the
token budget in the developer's hands.

---

## Per-skill governance and guardrails

`SkillGovernance` sets *defaults* for every tool bundled in the skill:

```python
from troopai.adk.skills import Skill, SkillGovernance

billing_skill = Skill(
    name="billing",
    description="Invoice lookup and payment operations",
    tools=[lookup_invoice, charge_card, issue_refund],
    governance=SkillGovernance(
        timeout=10.0,         # each tool gets 10 s unless it sets its own timeout
        max_result_tokens=512,
        max_retries=1,        # billing calls: fail fast, don't retry silently
    ),
)
```

Tool-level values override skill-level values — `governance` is always a
floor, never a ceiling.

Skill-level guardrails run *before* each tool's own guardrails:

```python
from troopai.adk.tools.tool_guardrails import ToolGuardrails, tool_input_guardrail, ToolInputGuardrailData, ToolGuardrailFunctionOutput


@tool_input_guardrail(name="billing_pii_check")
async def billing_pii_check(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Block raw card numbers from reaching billing tools."""
    if "4" in str(data.agent_output) and len(str(data.agent_output)) > 15:
        return ToolGuardrailFunctionOutput.reject_content("Raw card numbers are not accepted.")
    return ToolGuardrailFunctionOutput.allow()


billing_skill = Skill(
    name="billing",
    description="Invoice lookup and payment operations",
    tools=[lookup_invoice, charge_card, issue_refund],
    guardrails=ToolGuardrails(input=[billing_pii_check]),
)
```

Every tool inside `billing_skill` will have `billing_pii_check` prepended to
its input guardrail chain. Per-skill governance and guardrails are the primary
mechanism for the governance-bundling principle described in
{doc}`../architecture/governance`.

---

## Skill sources

Skills can be defined inline or loaded from external sources.

**Inline (code-defined)** — the default for application code:

```python
skill = Skill(name="research", description="Web research", tools=[fetch_article])
```

**Directory** — load from a folder containing a `SKILL.md` file. Compatible
with the LangChain / CrewAI / Google ADK skill directory format:

```python
skill = Skill.from_directory("./skills/code-review/")
```

The `SKILL.md` file uses YAML front matter for metadata and a markdown body
for instructions:

```yaml
---
name: code-review
description: Expert Python code review with security focus
tags: python, security, review
---

When reviewing code:
1. Check for security vulnerabilities first.
2. Then check for performance issues.
3. Finally suggest style improvements.
```

The directory loader sets `resource_root` automatically so that any scripts
under `scripts/` are bounded within the skill tree.

**Remote URL** — load from a SKILL.md hosted anywhere:

```python
skill = await Skill.from_url("https://skills.example.com/research/SKILL.md")
```

---

## Reusable skill libraries

Skills are plain dataclasses — define them once in a shared module and
import them wherever agents are built:

```python
# myapp/skills/__init__.py
from troopai.adk.skills import Skill, SkillGovernance, SkillMetadata
from myapp.tools.research import fetch_article, summarise
from myapp.tools.billing import lookup_invoice, charge_card

research_skill = Skill(
    name="research",
    description="Web research and summarisation",
    tools=[fetch_article, summarise],
    metadata=SkillMetadata(version="1.0.0", tags=("research",)),
    governance=SkillGovernance(timeout=30.0, max_result_tokens=1024),
)

billing_skill = Skill(
    name="billing",
    description="Invoice lookup and payment processing",
    tools=[lookup_invoice, charge_card],
    metadata=SkillMetadata(version="1.0.0", tags=("billing", "payments")),
    governance=SkillGovernance(timeout=10.0, max_retries=1),
)
```

```python
# myapp/agents/support.py
from troopai.adk.agents import Agent
from myapp.skills import billing_skill, research_skill

support_agent = Agent(
    name="Support",
    system_prompt="You handle customer support queries.",
    skills=[research_skill, billing_skill],
)
```

```python
# myapp/agents/billing_only.py
from troopai.adk.agents import Agent
from myapp.skills import billing_skill

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt="You handle billing queries only.",
    skills=[billing_skill],
)
```

Use `SkillSet` to keep related skills together and query them at build time:

```python
from troopai.adk.skills import SkillSet

finance_skills = SkillSet(name="finance", skills=[billing_skill, tax_skill, fx_skill])

# Find by name
inv = finance_skills.find("billing")

# Filter by tag
payment_skills = finance_skills.filter_by_tag("payments")
```

---

## LLM-driven discovery with `SkillDiscoveryToolset`

For agents that hold a large, open-ended library of skills, you can let the
LLM itself decide which skill to load. `SkillDiscoveryToolset` generates
`FunctionTool`s that the LLM calls to introspect the skill catalogue:

```python
from troopai.adk.skills import Skill, SkillDiscoveryToolset
from troopai.adk.skills import RECOMMENDED_SKILL_INSTRUCTIONS, prompt_with_skill_instructions

skills = [research_skill, billing_skill, safety_skill]
discovery = SkillDiscoveryToolset(skills=skills)

agent = Agent(
    name="General Assistant",
    system_prompt=prompt_with_skill_instructions(
        "You are a general-purpose assistant."
    ),
    skills=skills,
    tools=[*discovery.tools()],
)
```

The generated tools:

| Tool | What it does |
|---|---|
| `list_skills` | Returns names and descriptions of all skills. |
| `load_skill` | Returns full instructions and tool list for a named skill. |
| `load_skill_resource` | Reads a resource file from a skill (only generated if any skill has resources). |
| `run_skill_script` | Executes a Python or Bash script from a skill's resources. Disabled by default; set `enable_scripts=True` to opt in. Requires human approval on each call. |

Discovery tools are strictly **opt-in** — nothing is auto-injected.
`prompt_with_skill_instructions` prepends the recommended usage instructions
to any prompt string.

---

## Common skill patterns

**Research skill** — web search, article fetch, summariser, with a strict
instruction chain (fetch → summarise → cite). Governance caps token output
and timeout so a slow source doesn't stall the agent loop.

**Safety-review skill** — instructions for how to evaluate text for policy
violations; tools that call a moderation endpoint. Skill-level guardrails
reject inputs that are obviously out of policy before they reach the tools.
Can be combined with an `on_skill_activated` hook that logs every moderation
invocation to an audit sink.

**Billing skill** — payment and invoice tools with a tight governance budget
(`max_retries=1`, `timeout=10.0`) so failures are loud and fast. A
skill-level PII input guardrail blocks raw card numbers from ever reaching
tool arguments. Shared across customer-facing and internal agents from a
single definition in a `finance` skill library.

---

## See also

- {doc}`../concepts/index` — Skills vs Tools comparison table (§ "🎒 Skills vs 🔧 Tools")
- {doc}`../architecture/governance` — Audit substrate and tool permissions
- `examples/skills/` — Runnable end-to-end examples (`skills_agent_with_skills.py`,
  `skills_customer_support.py`, `skills_directory.py`, `skills_discovery.py`)
