"""Budget-aware tool management and cost optimization patterns.

Demonstrates several context management features:

1. **Dynamic tool enabling** — FunctionTool.enabled callback disables
   expensive tools when the token budget runs low.
2. **Tool result size limits** — FunctionTool.max_result_tokens prevents
   a single tool from bloating the message history.
3. **ExecutionAwareToolContext** — Opt-in execution state visibility for
   tools that need to make budget-aware decisions at runtime.
4. **CacheStrategy.STABLE** — Keep disabled tools in the tool list with
   [UNAVAILABLE] prefix to preserve prompt cache.

These patterns keep ToolContext and RunContext cleanly separated while
allowing the Runner to make cost-driven decisions.

Usage:
    python examples/tools/tool_context_management.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from dataclasses import dataclass

from troopai.adk.agents import Agent
from troopai.adk.context import CacheStrategy, ContextManagementConfig
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.context import RunContext
from troopai.adk.tools.function_tool import FunctionTool, function_tool
from troopai.adk.tools.tool_context import ExecutionAwareToolContext, ToolContext
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. User context — carries application state (NOT passed to the LLM)
# ---------------------------------------------------------------------------


@dataclass
class AppContext:
    user_id: str
    budget_mode: str = "normal"  # "normal" | "economy"


# ---------------------------------------------------------------------------
# 2. Tools with cost controls
# ---------------------------------------------------------------------------


@function_tool(name="quick_search", description="Fast cached search (low cost)")
def quick_search(ctx: ToolContext[AppContext], query: str) -> str:
    """Lightweight search using a local cache or simple index."""
    return f"[quick_search] Cached result for '{query}'"


# max_result_tokens limits how much context this tool can consume.
# Even if deep_search returns 5,000 tokens, the Runner truncates
# to 500 before inserting into message history.
@function_tool(
    name="deep_search",
    description="Thorough RAG search (high cost)",
    max_result_tokens=500,
)
def deep_search(ctx: ToolContext[AppContext], query: str) -> str:
    """Expensive retrieval-augmented generation pipeline."""
    # Simulate a large result that would normally bloat context
    return f"[deep_search] Comprehensive result for '{query}': " + "x" * 3000


# ---------------------------------------------------------------------------
# 3. ExecutionAwareToolContext — opt-in budget visibility
#
# This tool receives execution state snapshots (usage, turn count,
# message count, token estimate) as read-only copies.
# ---------------------------------------------------------------------------


@function_tool(name="smart_search", description="Search with budget awareness")
def smart_search(ctx: ExecutionAwareToolContext[AppContext], query: str) -> str:
    """Search that adjusts verbosity based on context window usage."""
    if ctx.tokens > 50_000:
        # Context window is getting full — return concise result
        return f"[smart_search:brief] '{query}' → found 3 relevant results"
    return f"[smart_search:full] Detailed results for '{query}' with citations and excerpts"


# ---------------------------------------------------------------------------
# 4. Dynamic enabled callback — budget gating
# ---------------------------------------------------------------------------

TOKEN_BUDGET = 100_000  # Switch to economy mode above this threshold


def _is_within_budget(context: RunContext[AppContext]) -> bool:
    """Return True while the run has tokens to spare."""
    if context is None:
        return True
    return context.usage.total_tokens < TOKEN_BUDGET


# Wrap deep_search with a dynamic enabled callback.
budget_aware_deep_search = FunctionTool(
    name="deep_search",
    description="Thorough RAG search (high cost, disabled when budget is low)",
    on_invoke=deep_search.on_invoke,
    schema=deep_search.schema,
    enabled=_is_within_budget,
    max_result_tokens=500,  # Still limit even when enabled
)


# ---------------------------------------------------------------------------
# 5. Agent uses all tools — Runner handles budget gating + result limits
# ---------------------------------------------------------------------------

agent = Agent(
    name="Research Agent",
    system_prompt=(
        "You are a research assistant. Use deep_search for thorough answers, "
        "quick_search for fast lookups, and smart_search when you want to "
        "adapt based on context usage. Use whichever tools are available."
    ),
    tools=[quick_search, budget_aware_deep_search, smart_search],
)


async def main() -> None:
    context = AppContext(user_id="demo-user")

    # CacheStrategy.STABLE preserves prompt cache when tools toggle on/off
    result = await Runner.arun(
        agent,
        "Search for recent advances in quantum computing",
        context=context,
        run_config=RunConfig(
            verbose=VerboseConfig(),
            context_management=ContextManagementConfig(
                cache_strategy=CacheStrategy.STABLE,
            ),
        ),
    )

    logger.info(f"\nFinal output: {result.final_output}")
    logger.info(f"Total tokens used: {result.context.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
