(foundations/three-limits)=

# 📐 Three Mathematical Limits That Shape This ADK

> Every AI agent system bumps against three formal results. This page
> states each, sketches why it holds, and shows the codebase decision
> it forced. We don't try to engineer past these limits; we engineer
> **with** them.

```{figure} ../_static/images/foundations/three-limits-synthesis.svg
:alt: The three theorems mapped to three engineering responses in the ADK.
:width: 70%
:class: themed
:align: center

How the three theorems shape three engineering responses in the ADK.
```

(halting-problem)=

## ⏱️ Turing's Halting Problem (1936)

**Statement.** No algorithm decides, for every (program, input) pair,
whether the program halts.

**Why it holds (sketch).** Suppose such a decider `H(p, x)` exists.
Build a new program

```python
def D(p):
    if H(p, p):
        while True:
            pass     # loop forever
    return            # halt
```

Then `D(D)` halts if and only if `H(D, D)` says it does **not**. That's
a direct contradiction; therefore `H` cannot exist. This is the classical
*diagonalisation* argument, identical in shape to Cantor's diagonal for
the reals.

**Implication for agents.** An LLM agent loop is a program whose halting
depends on inputs the loop cannot inspect in advance — tool results,
model continuations, retrieval hits, downstream agent decisions. You
cannot *statically* prove "this agent terminates on this prompt".

**Codebase tie-in.**

- `max_turns` is a non-optional budget on `Runner.arun`. The Runner does
  not trust the loop to self-terminate.
- `max_handoffs`, `max_retries`, and the `*_budget` fields are the same
  pattern at finer granularity.
- The project's cost-conservative defaults — every off/smallest/bounded
  default — are the Halting Problem made operational.

```{mermaid}
flowchart LR
  start([prompt]) --> step{LLM step}
  step -->|tool calls| tools[run tools]
  tools --> step
  step -->|final| done([result])
  step -. ?halts .-> guard[max_turns guard]
  guard --> done
  classDef undecidable fill:#ffd6d6,stroke:#c00,stroke-dasharray:5
  class step undecidable
```

## 🧮 Rice's Theorem (1953)

**Statement.** Every non-trivial *semantic* property of programs is
undecidable.

> A property is **semantic** if it depends on what the program does,
> not on how it's written. A property is **non-trivial** if some
> programs have it and others don't.

**Why it holds (sketch).** Any non-trivial semantic property reduces
to the Halting Problem. Pick a property `P` true for some programs and
false for others. Build a transformation that produces a program `Q(p)`
which has property `P` if and only if `p` halts on some fixed input.
A decider for `P` would then decide halting — which we just showed is
impossible. Contradiction.

**Implication for agents.** "Does this agent achieve goal G correctly
for every input?" is a non-trivial semantic property → undecidable.
You cannot *statically* verify agent correctness in general.

**Codebase tie-in.**

- `src/troopai/adk/evals/` is a first-class subsystem, not an
  afterthought. Where formal verification is excluded by Rice,
  empirical evaluation is the only credible substitute.
- LLM-as-judge graders, agent-as-judge graders, and the structured
  eval harness exist because correctness is *measured*, not *proven*.

### On "AGI"

The label *Artificial General Intelligence* suggests a system that
solves arbitrary goals correctly. Halting + Rice together exclude that
target as a formal possibility: any sufficiently expressive agent is
Turing-equivalent, and "solves arbitrary goals correctly" is the
semantic property *par excellence*. The limit is not "we haven't built
it yet" — it is "no Turing-complete computational substrate can in
principle decide arbitrary goal-satisfaction".

We therefore treat **"AGI" as a marketing label**, not an engineering
target. The engineering targets that survive Halting + Rice are:

1. Bounded loops with explicit budgets (Halting).
2. Empirical evaluation over formal verification (Rice).
3. Specialised competence over universal claims (No Free Lunch, next).

```{mermaid}
flowchart TB
  prog([any agent program])
  prog --> region{semantic property?}
  region -->|trivial| dec[decidable<br/>e.g. syntactic checks]
  region -->|non-trivial| undec[Rice region<br/>e.g. correctness, safety]
  undec --> evals[evals subsystem<br/>empirical only]
  classDef rice fill:#ffd6d6,stroke:#c00
  class undec rice
```

## 🍱 No Free Lunch (Wolpert & Macready, 1997)

**Statement.** Averaged over all possible problem distributions, every
optimisation algorithm has identical expected performance.

**Why it holds (sketch).** Combinatorial counting: for every problem
on which algorithm `A` beats algorithm `B`, there exists a "twin"
problem (with the loss surface rearranged) on which `B` beats `A` by
the same margin. Across all problems, the wins cancel exactly.

**Implication for agents.** A single general-purpose agent that beats
a *targeted* specialist on every task is mathematically excluded.
Specialisation is the price of measurable competence.

**Codebase tie-in.**

- **Handoffs** route to the agent best-fit for a sub-problem.
- **Swarms** cycle specialised agents iteratively.
- **Graphs** orchestrate state-machine-shaped specialisation manifolds.
- **Skills** package narrow capability stacks (instructions + tools +
  governance).
- **Sandbox-isolated tool experts** treat tool execution as a specialised
  competence.

Every concurrency / composition primitive in the ADK is a manifold along
which you specialise. You **compose narrow experts**; you don't
**inflate one generalist**.

```{mermaid}
flowchart LR
  subgraph "distribution domain"
    d1[task family 1]
    d2[task family 2]
    d3[task family 3]
  end
  d1 --> spec1[specialist A]
  d2 --> spec2[specialist B]
  d3 --> spec3[specialist C]
  spec1 --> orch{orchestrator}
  spec2 --> orch
  spec3 --> orch
```

## 🧷 Synthesis

| Limit            | Engineering response                              | Codebase manifestation                                            |
| ---------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| Halting Problem  | Bound everything; no implicit self-termination.   | `max_turns`, `max_handoffs`, `*_budget`, cost-conservative dflts. |
| Rice's Theorem   | Measure, don't prove.                             | First-class `evals/` subsystem, hybrid graders.                   |
| No Free Lunch    | Specialise; compose; never aggregate.             | Handoffs, swarms, graphs, skills, sandbox tool isolation.         |

## ⚠️ What these limits do *not* say

:::{warning}
Misquoted versions of these theorems do a lot of damage. Be precise.
:::

- **Halting and Rice are about *universal* deciders, not specific
  programs.** Most loops in practice terminate; budgets are a safety
  net, not a prediction. Many specific properties of specific programs
  are perfectly decidable.
- **No Free Lunch averages over *all* distributions.** Your domain is
  not all distributions. Specialisation wins on the distributions you
  care about — that's exactly the lever.
- **None of these say agents are useless.** They say agents are
  *bounded*. Bounded ≠ broken.

## Further reading

- Turing, A. M. (1936). *On Computable Numbers, with an Application to
  the Entscheidungsproblem*.
- Rice, H. G. (1953). *Classes of Recursively Enumerable Sets and Their
  Decision Problems*.
- Wolpert, D. H., & Macready, W. G. (1997). *No Free Lunch Theorems for
  Optimization*.
- Hopcroft, Motwani & Ullman (2006). *Introduction to Automata Theory,
  Languages, and Computation* — for the formal computability framing.
