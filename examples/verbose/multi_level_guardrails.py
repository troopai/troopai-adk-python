"""Verbose output — multi-level guardrail visualization.

Tool-scoped guardrail hooks (``on_tool_input_guardrail_*`` /
``on_tool_output_guardrail_*``) have no CrewAI equivalent
(CrewAI only surfaces agent-level guardrails). This example shows
them side-by-side with agent-level input/output guardrails so the
panel stack renders both levels distinctly.

Colour legend:

* green border = ``allow`` (pass)
* red border   = ``reject_content`` or ``raise_exception`` (trip)
* yellow       = unknown / warn

Try it:

    python examples/verbose/multi_level_guardrails.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import re

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.tools.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolGuardrails,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)

logger = logging.getLogger(__name__)


# ── Tool-level guardrails ────────────────────────────────────────────


@tool_input_guardrail()
def reject_sql_injection(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    # Illustrative demo regex — production guardrails should use a real
    # parser or a managed SQLi detector (parameterised queries remain
    # the only actual defence).
    raw = str(data.context.tool_arguments)
    if re.search(r";\s*DROP\s+TABLE", raw, re.IGNORECASE):
        return ToolGuardrailFunctionOutput.reject_content("Input rejected: suspicious SQL keyword (demo regex).")
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail()
def reject_ssn_in_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    output = str(data.output)
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", output):
        return ToolGuardrailFunctionOutput.reject_content("Output filtered: contains Social Security Number.")
    return ToolGuardrailFunctionOutput.allow()


@function_tool(
    name="lookup_customer",
    description="Look up a customer record in the database.",
    guardrails=ToolGuardrails(
        input=[reject_sql_injection],
        output=[reject_ssn_in_output],
    ),
)
def lookup_customer(query: str) -> str:
    # Deliberately returns a fictional SSN-shaped placeholder so the
    # output guardrail trips. The pattern matches the US SSN regex but
    # the value is reserved-for-documentation (000-00-0000) — never
    # copy real SSNs into example code.
    return f"Customer match for '{query}': John Doe, SSN: 000-00-0000."


async def main() -> None:
    agent = Agent(
        name="SupportAgent",
        llm="gpt-4o-mini",
        system_prompt="You are a customer support agent. Use lookup_customer to find records. Answer concisely.",
        tools=[lookup_customer],
    )

    result = await Runner.arun(
        agent,
        "Find the customer record for 'alice'.",
        run_config=RunConfig(verbose=VerboseConfig(mode="auto")),
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
