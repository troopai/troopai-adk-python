"""
Example: Agent Using Skills — End-to-End

Demonstrates a real agent using skills to handle tasks across multiple
domains.  The agent has two skills:

1. **weather** — Provides a weather lookup tool with instructions on
   how to format weather reports
2. **math** — Provides a calculator tool with instructions on showing
   work step-by-step

The agent receives a user query and uses the appropriate skill's tools
to answer.  Skills bundle instructions (domain expertise) with tools
(capabilities) — the agent gets both when skills are attached.

Three patterns demonstrated:
- Pattern 1: Multi-skill agent with EAGER activation
- Pattern 2: Multi-skill agent with LAZY activation (cost optimization)
- Pattern 3: Skill governance + guardrails in action
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import operator

from troopai.adk.agents import Agent
from troopai.adk.llms import LLMConfig
from troopai.adk.run import RunConfig, Runner
from troopai.adk.skills import Skill, SkillActivation, SkillGovernance, SkillMetadata
from troopai.adk.tools import function_tool
from troopai.adk.tools.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolGuardrails,
    ToolInputGuardrailData,
    tool_input_guardrail,
)
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Tools for the weather skill
# =============================================================================


@function_tool(
    name="get_weather",
    description="Get the current weather for a city. Returns temperature, conditions, and humidity.",
)
def get_weather(city: str) -> str:
    """Simulate weather lookup."""
    weather_data = {
        "paris": {"temp": 18, "conditions": "Partly cloudy", "humidity": 65},
        "tokyo": {"temp": 24, "conditions": "Sunny", "humidity": 45},
        "new york": {"temp": 12, "conditions": "Rainy", "humidity": 82},
        "london": {"temp": 14, "conditions": "Overcast", "humidity": 78},
    }
    key = city.lower()
    if key in weather_data:
        d = weather_data[key]
        return f"Weather in {city}: {d['temp']}°C, {d['conditions']}, {d['humidity']}% humidity"
    return f"Weather data not available for {city}. Available cities: {', '.join(weather_data)}"


@function_tool(
    name="get_forecast",
    description="Get a 3-day weather forecast for a city.",
)
def get_forecast(city: str) -> str:
    """Simulate forecast lookup."""
    return f"3-day forecast for {city}: Day 1: 17°C Partly cloudy | Day 2: 19°C Sunny | Day 3: 15°C Light rain"


# =============================================================================
# Tools for the math skill
# =============================================================================

# Simple safe arithmetic operations
_OPS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


@function_tool(
    name="calculate",
    description="Perform a math operation. Supports: add, subtract, multiply, divide.",
)
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a safe math operation."""
    op_func = _OPS.get(operation.lower())
    if op_func is None:
        return f"Error: Unknown operation '{operation}'. Use: add, subtract, multiply, divide"
    if operation == "divide" and b == 0:
        return "Error: Division by zero"
    result = op_func(a, b)
    return f"{a} {operation} {b} = {result}"


@function_tool(
    name="unit_convert",
    description="Convert between common units. Supports km/miles, kg/lbs, celsius/fahrenheit.",
)
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units."""
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
        ("celsius", "fahrenheit"): lambda v: (v * 9 / 5) + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.2f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} is not supported"


# =============================================================================
# Skill-level guardrail
# =============================================================================


@tool_input_guardrail(name="math_input_sanitizer")
async def math_sanitizer(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Block dangerous inputs to math tools."""
    raw_input = str(data.agent_output)
    dangerous = ["import", "exec", "__", "os.", "sys."]
    for keyword in dangerous:
        if keyword in raw_input.lower():
            return ToolGuardrailFunctionOutput.reject_content(f"Input contains forbidden keyword: {keyword}")
    return ToolGuardrailFunctionOutput.allow()


# =============================================================================
# Skill definitions
# =============================================================================

weather_skill = Skill(
    name="weather",
    description="Weather lookup and forecasting capabilities",
    instructions="""When providing weather information:
1. Use get_weather to check current conditions first
2. If the user asks about future weather, use get_forecast
3. Always mention temperature in both Celsius and Fahrenheit
4. Include a brief clothing recommendation based on conditions
5. If a city is not found, suggest nearby available cities""",
    tools=[get_weather, get_forecast],
    metadata=SkillMetadata(
        version="1.0.0",
        tags=("weather", "forecast", "travel"),
    ),
    governance=SkillGovernance(
        timeout=10.0,
        max_result_tokens=500,
    ),
)

math_skill = Skill(
    name="math",
    description="Mathematical calculations and unit conversions",
    instructions="""When solving math problems:
1. Show your work step-by-step before using the calculate tool
2. For unit conversions, use the unit_convert tool
3. Round final answers to 2 decimal places
4. If a problem has multiple steps, break it down and solve each part
5. Explain what each calculation means in plain language""",
    tools=[calculate, unit_convert],
    metadata=SkillMetadata(
        version="1.0.0",
        tags=("math", "calculation", "conversion"),
    ),
    governance=SkillGovernance(
        timeout=5.0,
        max_retries=2,
    ),
    guardrails=ToolGuardrails(input=[math_sanitizer]),
)


# =============================================================================
# Pattern 1: Multi-skill agent with EAGER activation
# =============================================================================

eager_agent = Agent(
    name="Travel Assistant",
    system_prompt=(
        "You are a helpful travel planning assistant. You can check weather "
        "conditions and do math calculations for trip budgeting. "
        "Use your tools when relevant. Be concise and practical."
    ),
    skills=[weather_skill, math_skill],
    skill_activation=SkillActivation.EAGER,
    llm_config=LLMConfig(temperature=0.3),
)


# =============================================================================
# Pattern 2: Multi-skill agent with LAZY activation
# =============================================================================


lazy_agent = Agent(
    name="Lazy Travel Assistant",
    system_prompt=(
        "You are a helpful travel planning assistant. You can check weather "
        "conditions and do math calculations for trip budgeting. "
        "Use your tools when relevant. Be concise and practical."
    ),
    skills=[weather_skill, math_skill],
    skill_activation=SkillActivation.LAZY,
    llm_config=LLMConfig(temperature=0.3),
)


# =============================================================================
# Run examples
# =============================================================================


async def run_eager_example() -> None:
    """EAGER: Both skill instructions are in the system prompt from the start."""
    logger.info("=" * 60)
    logger.info("Pattern 1: EAGER Activation")
    logger.info("=" * 60)

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Weather query
    result = await Runner.arun(
        eager_agent,
        "What's the weather like in Paris right now?",
        run_config=run_config,
    )
    logger.info("Weather query result:\n%s", result.final_output)
    logger.info("Tokens used: %s", result.context.usage.total_tokens)

    # Math query
    result2 = await Runner.arun(
        eager_agent,
        "If a hotel in Tokyo costs 15000 yen per night and I'm staying 5 nights, "
        "how much is that total? Also convert 15000 yen to USD (assume 1 USD = 150 yen).",
        run_config=run_config,
    )
    logger.info("\nMath query result:\n%s", result2.final_output)
    logger.info("Tokens used: %s", result2.context.usage.total_tokens)


async def run_lazy_example() -> None:
    """LAZY: Skill instructions load only when the LLM calls a skill's tool."""
    logger.info("=" * 60)
    logger.info("Pattern 2: LAZY Activation (cost optimization)")
    logger.info("=" * 60)

    run_config = RunConfig(verbose=VerboseConfig())

    # Only the weather skill instructions will load (since only weather tools are called)
    result = await Runner.arun(
        lazy_agent,
        "What's the weather like in Tokyo?",
        run_config=run_config,
    )
    logger.info("Weather query result:\n%s", result.final_output)
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def main() -> None:
    await run_eager_example()
    logger.info("")
    await run_lazy_example()


if __name__ == "__main__":
    asyncio.run(main())
