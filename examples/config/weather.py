"""Tool body and output schema for the declarative-config example.

These live in an importable sibling module (not inside the runnable script) so
``agent.json`` can reference them by a normal dotted path —
``weather.get_weather`` and ``weather.WeatherReport`` — instead of the
non-portable ``__main__:`` form. The loader puts this file's directory on the
import path while it resolves the config, so the bare ``weather`` module name
resolves no matter where the config is loaded from.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from troopai.adk.tools import function_tool


class WeatherReport(BaseModel):
    """Structured weather answer the agent must return."""

    city: str = Field(description="The city the report is about.")
    summary: str = Field(description="A one-sentence weather summary.")


@function_tool
def get_weather(city: str) -> str:
    """Return a (canned) weather report for a city."""
    return f"It is 21°C and sunny in {city}."
