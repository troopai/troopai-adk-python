"""Tests for :class:`~troopai.adk.workflows.restate.llm.RestateLLM`.

All tests skip when ``restate`` is not installed.
See test_get_restate_context.py for the narrowed-RuntimeError tests
that run without the SDK.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

restate = pytest.importorskip("restate")
# isort: split
from troopai.adk.llms.llm import LLM
from troopai.adk.workflows.engine import ModelActivityConfig
from troopai.adk.workflows.restate.llm import RestateLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm() -> MagicMock:
    """Return a ``MagicMock`` with ``LLM`` spec and a controlled ``acomplete``."""
    mock = MagicMock(spec=LLM)
    mock.acomplete = AsyncMock(return_value=MagicMock())
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRestateLLMStoresWrapped:
    def test_restate_llm_stores_wrapped(self) -> None:
        """``wrapped`` field holds the LLM passed at construction."""
        inner = _make_mock_llm()
        llm = RestateLLM(
            wrapped=inner,
            activity_config=ModelActivityConfig(),
        )

        assert llm.wrapped is inner

    def test_restate_llm_stores_activity_config(self) -> None:
        """``activity_config`` field holds the config passed at construction."""
        inner = _make_mock_llm()
        config = ModelActivityConfig(maximum_attempts=5)
        llm = RestateLLM(
            wrapped=inner,
            activity_config=config,
        )

        assert llm.activity_config is config


class TestRestateLLMDelegatesOutsideContext:
    async def test_restate_llm_delegates_outside_context(self) -> None:
        """``acomplete`` forwards to ``wrapped.acomplete`` when context is None."""
        inner = _make_mock_llm()
        llm = RestateLLM(
            wrapped=inner,
            activity_config=ModelActivityConfig(),
        )

        with patch(
            "troopai.adk.workflows.restate.llm.get_restate_context",
            return_value=None,
        ):
            result = await llm.acomplete("hello", stream=False)

        inner.acomplete.assert_awaited_once_with("hello", None, None, None, stream=False)
        assert result is inner.acomplete.return_value

    async def test_restate_llm_delegates_outside_context_with_config(self) -> None:
        """``acomplete`` passes llm_config through to wrapped when context is None."""
        from troopai.adk.llms.llm_config import LLMConfig

        inner = _make_mock_llm()
        llm = RestateLLM(
            wrapped=inner,
            activity_config=ModelActivityConfig(),
        )
        cfg = LLMConfig(temperature=0.5)

        with patch(
            "troopai.adk.workflows.restate.llm.get_restate_context",
            return_value=None,
        ):
            result = await llm.acomplete("hi", llm_config=cfg, stream=False)

        inner.acomplete.assert_awaited_once_with("hi", cfg, None, None, stream=False)
        assert result is inner.acomplete.return_value

    async def test_restate_llm_delegates_stream_outside_context(self) -> None:
        """``acomplete(stream=True)`` forwards with stream=True when context is None."""
        inner = _make_mock_llm()
        llm = RestateLLM(
            wrapped=inner,
            activity_config=ModelActivityConfig(),
        )

        with patch(
            "troopai.adk.workflows.restate.llm.get_restate_context",
            return_value=None,
        ):
            await llm.acomplete("stream test", stream=True)

        inner.acomplete.assert_awaited_once_with("stream test", None, None, None, stream=True)


class TestRestateLLMStreamInsideHandler:
    async def test_stream_true_inside_handler_raises_not_implemented(self) -> None:
        """``acomplete(stream=True)`` inside a handler fails loudly.

        ``ctx.run()`` returns a serialized journaled snapshot, so an async
        iterator cannot be produced.  Rather than silently returning a
        non-iterable ``LLMResponse`` (which would crash the caller's
        ``async for`` later, obscurely), the call must raise immediately.
        """
        inner = _make_mock_llm()
        llm = RestateLLM(
            wrapped=inner,
            activity_config=ModelActivityConfig(),
        )

        ctx = MagicMock()
        ctx.run = AsyncMock()

        with (
            patch(
                "troopai.adk.workflows.restate.llm.get_restate_context",
                return_value=ctx,
            ),
            pytest.raises(NotImplementedError, match="stream=True"),
        ):
            await llm.acomplete("stream test", stream=True)

        # The guard must fire before any journaling / wrapped invocation.
        ctx.run.assert_not_awaited()
        inner.acomplete.assert_not_awaited()
