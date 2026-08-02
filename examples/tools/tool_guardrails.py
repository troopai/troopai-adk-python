"""Tool-level guardrails: input validation, output filtering, and error recovery.

Demonstrates the three guardrail behaviors at the tool level:

1. **allow** — Input/output passes validation, tool proceeds normally.
2. **reject_content** — Tool call is rejected but execution continues.
   The rejection message goes back to the LLM as the tool result.
3. **raise_exception** — Execution halts immediately with
   ``ToolGuardrailTripwireTriggered``.

Also demonstrates **tool error recovery**: when ``fail_on_tool_error=False``,
runtime exceptions (like ``ZeroDivisionError``) are caught and sent back to
the LLM as error messages, letting the model adapt instead of crashing.

Both input and output guardrails support all three behaviors.  Input
guardrails run before execution (Layer 1), output guardrails run
after execution (Layer 3).

Usage:
    python examples/tools/tool_guardrails.py
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

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.tools.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolGuardrails,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Input guardrail: reject SQL injection attempts
# ---------------------------------------------------------------------------


@tool_input_guardrail()
def reject_sql_injection(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Check tool input for SQL injection patterns."""
    raw = str(data.context.tool_arguments)
    sql_patterns = [r";\s*DROP\s+TABLE", r"'\s*OR\s+'1'\s*=\s*'1", r"UNION\s+SELECT"]
    for pattern in sql_patterns:
        if re.search(pattern, raw, re.IGNORECASE):
            return ToolGuardrailFunctionOutput.reject_content("Input rejected: potential SQL injection detected.")
    return ToolGuardrailFunctionOutput.allow()


# ---------------------------------------------------------------------------
# 2. Output guardrail: reject PII in tool output
# ---------------------------------------------------------------------------


@tool_output_guardrail()
def reject_pii_in_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Check tool output for PII patterns (SSN, credit card)."""
    output = str(data.output)
    # Simple SSN pattern
    if re.search(r"\b\d{3}-\d{2}-\d{4}\b", output):
        return ToolGuardrailFunctionOutput.reject_content("Output filtered: contains Social Security Number.")
    # Simple credit card pattern
    if re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", output):
        return ToolGuardrailFunctionOutput.reject_content("Output filtered: contains credit card number.")
    return ToolGuardrailFunctionOutput.allow()


# ---------------------------------------------------------------------------
# 3. Output guardrail: halt on critical content
# ---------------------------------------------------------------------------


@tool_output_guardrail()
def halt_on_critical(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Halt execution if tool output contains critical markers."""
    output = str(data.output)
    if "[CRITICAL]" in output:
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info={"reason": "Critical content in tool output", "output": output},
        )
    return ToolGuardrailFunctionOutput.allow()


# ---------------------------------------------------------------------------
# Tools with guardrails attached
# ---------------------------------------------------------------------------


@function_tool(
    name="search_database",
    description="Search the customer database by query",
    guardrails=ToolGuardrails(
        input=[reject_sql_injection],
        output=[reject_pii_in_output, halt_on_critical],
    ),
)
def search_database(query: str) -> str:
    """Simulate a database search."""
    # In a real app, this would query a database
    return f"Found 3 results for query: {query}"


@function_tool(
    name="get_user_profile",
    description="Get a user's profile information",
    guardrails=ToolGuardrails(output=[reject_pii_in_output]),
)
def get_user_profile(user_id: str) -> str:
    """Simulate fetching a user profile."""
    # Simulated response with PII — the output guardrail will catch this
    return f"User {user_id}: John Doe, SSN: 123-45-6789, Email: john@example.com"


# ---------------------------------------------------------------------------
# Agent using guarded tools
# ---------------------------------------------------------------------------


agent = Agent(
    name="Customer Support",
    system_prompt="Help customers find information. Use the available tools.",
    tools=[search_database, get_user_profile],
)


# ---------------------------------------------------------------------------
# 4. Tool error recovery: ZeroDivisionError
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Output guardrail: validate business logic on tool results
# ---------------------------------------------------------------------------


@tool_output_guardrail()
def validate_discount_result(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Reject discount results that are logically nonsensical."""
    output = str(data.output)
    # Extract the number from "Discount: 42.5%"
    match = re.search(r"Discount:\s*([-\d.]+)%", output)
    if match is not None:
        value = float(match.group(1))
        if value < 0:
            return ToolGuardrailFunctionOutput.reject_content(
                f"Invalid result: negative discount ({value:.1f}%) means "
                f"the sale price is higher than the original. "
                f"Please check the prices."
            )
        if value > 100:
            return ToolGuardrailFunctionOutput.reject_content(
                f"Invalid result: discount of {value:.1f}% exceeds 100%. The sale price cannot be negative."
            )
    return ToolGuardrailFunctionOutput.allow()


@function_tool(
    name="calculate_discount",
    description="Calculate discount percentage from original and sale price.",
)
def calculate_discount(original_price: float, sale_price: float) -> str:
    """Calculate the discount — may raise ZeroDivisionError if original is 0."""
    discount = (original_price - sale_price) / original_price * 100
    return f"Discount: {discount:.1f}%"


@function_tool(
    name="calculate_discount_guarded",
    description="Calculate discount percentage (with validation).",
    guardrails=ToolGuardrails(output=[validate_discount_result]),
)
def calculate_discount_guarded(original_price: float, sale_price: float) -> str:
    """Calculate the discount with output guardrail validation."""
    discount = (original_price - sale_price) / original_price * 100
    return f"Discount: {discount:.1f}%"


calculator_agent = Agent(
    name="Pricing Assistant",
    system_prompt=(
        "You help customers with pricing calculations. Use the "
        "calculate_discount tool. If the tool returns an error, "
        "explain the issue to the user."
    ),
    tools=[calculate_discount],
)

guarded_calculator_agent = Agent(
    name="Pricing Assistant (Guarded)",
    system_prompt=(
        "You help customers with pricing calculations. Use the "
        "calculate_discount_guarded tool. If the tool returns an error "
        "or rejection, explain the issue to the user."
    ),
    tools=[calculate_discount_guarded],
)


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Normal query — passes guardrails
    logger.info("=== Normal query ===")
    result = await Runner.arun(agent, "Search for order #12345", run_config=run_config)
    logger.info(f"Output: {result.final_output}\n")

    # The output guardrail would filter PII if the tool returns it
    logger.info("=== Query returning PII (output guardrail filters it) ===")
    result = await Runner.arun(agent, "Get profile for user U-001", run_config=run_config)
    logger.info(f"Output: {result.final_output}\n")

    # Tool error recovery: ZeroDivisionError
    # With fail_on_tool_error=False, the exception is caught and sent
    # back to the LLM as "Error executing tool: division by zero".
    # The LLM sees this and explains the issue instead of crashing.
    logger.info("=== Tool error recovery (ZeroDivisionError) ===")
    result = await Runner.arun(
        calculator_agent,
        "What's the discount if the original price is $0 and sale price is $10?",
        run_config=RunConfig(verbose=VerboseConfig(), fail_on_tool_error=False),
    )
    logger.info(f"Output: {result.final_output}\n")

    # Output guardrail catches logically nonsensical results.
    # The tool runs successfully but returns "Discount: -100.0%" because
    # the sale price ($200) is higher than the original ($100).
    # The guardrail rejects this and the LLM explains the issue.
    logger.info("=== Output guardrail (negative discount rejected) ===")
    result = await Runner.arun(
        guarded_calculator_agent,
        "What's the discount if the original price is $100 and the sale price is $200?",
        run_config=run_config,
    )
    logger.info(f"Output: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
