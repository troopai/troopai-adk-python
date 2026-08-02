"""Tests for skill_rendering: enabled evaluation and instruction rendering.

Focus: a dynamic (callable) ``Skill.enabled`` gate cannot be evaluated
without a run context, and must fail *closed* — the skill's tools and
instructions are withheld rather than injected on an unchecked gate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from troopai.adk.run.context import RunContext
from troopai.adk.skills import Skill
from troopai.adk.skills.skill_rendering import check_skill_enabled, render_skill_instructions
from troopai.adk.utils.typedef import MaybeAwaitable


def _skill(
    name: str,
    enabled: bool | Callable[[RunContext[Any]], MaybeAwaitable[bool]],
    instructions: str = "do the thing",
) -> Skill:
    return Skill(name=name, description=f"{name} description", instructions=instructions, enabled=enabled)


def _always_true(ctx: RunContext[Any]) -> bool:
    del ctx
    return True


def _always_false(ctx: RunContext[Any]) -> bool:
    del ctx
    return False


async def _async_true(ctx: RunContext[Any]) -> bool:
    del ctx
    return True


class TestCheckSkillEnabled:
    @pytest.mark.asyncio
    async def test_static_true_without_context(self) -> None:
        assert await check_skill_enabled(_skill("s", True), None) is True

    @pytest.mark.asyncio
    async def test_static_false_without_context(self) -> None:
        assert await check_skill_enabled(_skill("s", False), None) is False

    @pytest.mark.asyncio
    async def test_callable_without_context_fails_closed(self) -> None:
        # A dynamic gate needs the run context to evaluate; without it the
        # skill must be treated as disabled, not blindly enabled.
        assert await check_skill_enabled(_skill("s", _always_true), None) is False

    @pytest.mark.asyncio
    async def test_callable_with_context_evaluated(self) -> None:
        ctx: RunContext[Any] = RunContext(context=None)
        assert await check_skill_enabled(_skill("s", _always_true), ctx) is True
        assert await check_skill_enabled(_skill("s", _always_false), ctx) is False

    @pytest.mark.asyncio
    async def test_async_callable_with_context(self) -> None:
        ctx: RunContext[Any] = RunContext(context=None)
        assert await check_skill_enabled(_skill("s", _async_true), ctx) is True


class TestRenderSkillInstructions:
    @pytest.mark.asyncio
    async def test_callable_gated_skill_excluded_without_context(self) -> None:
        # No context => the dynamic gate can't be honored => the skill's
        # instructions must NOT be injected into the system prompt.
        rendered = await render_skill_instructions([_skill("gated", _always_true, "secret steps")], None)
        assert rendered is None

    @pytest.mark.asyncio
    async def test_static_skill_rendered_without_context(self) -> None:
        rendered = await render_skill_instructions([_skill("plain", True, "visible steps")], None)
        assert rendered is not None
        assert "visible steps" in rendered

    @pytest.mark.asyncio
    async def test_callable_gated_skill_included_with_context(self) -> None:
        ctx: RunContext[Any] = RunContext(context=None)
        rendered = await render_skill_instructions([_skill("gated", _always_true, "now visible")], ctx)
        assert rendered is not None
        assert "now visible" in rendered
