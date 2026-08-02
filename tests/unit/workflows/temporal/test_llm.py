"""Tests for :class:`~troopai.adk.workflows.temporal.llm.TemporalLLM`.

Covers:
- Outside-workflow path: ``acomplete`` delegates directly to the wrapped LLM.
- ``install`` replaces ``agent.llm`` with a ``TemporalLLM`` wrapping the original.
- ``install`` skips agents that already have a ``TemporalLLM`` (no double-wrapping).
- ``__post_init__`` sets ``model_name`` to ``str(wrapped)`` when left empty.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.llms.llm import LLM
from troopai.adk.workflows.engine import ModelActivityConfig
from troopai.adk.workflows.temporal.llm import TemporalLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(str_repr: str = "mock-llm") -> MagicMock:
    """Return a ``MagicMock`` with ``LLM`` spec and a controlled ``__str__``."""
    mock = MagicMock(spec=LLM)
    mock.__str__ = MagicMock(return_value=str_repr)
    mock.acomplete = AsyncMock(return_value=MagicMock())
    return mock


def _make_agent(llm: LLM | None = None, handoffs: list | None = None) -> MagicMock:
    """Return a lightweight mock agent with ``llm`` and ``handoffs`` attributes."""
    agent = MagicMock()
    agent.name = "TestAgent"
    agent.llm = llm
    agent.handoffs = handoffs
    return agent


class _AddressLLM(LLM):
    """Minimal real LLM whose ``str()`` is an object address but exposes ``.model``.

    Used to prove ``model_name`` derivation prefers the stable ``model`` id
    over the ``<... object at 0x...>`` address that ``str()`` yields.
    """

    model = "claude-address-x"

    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> Any:
        raise NotImplementedError


class _NoModelLLM(LLM):
    """Minimal real LLM with no ``model`` attribute (str() fallback path)."""

    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Delegation outside workflow
# ---------------------------------------------------------------------------


class TestTemporalLLMDelegatesOutsideWorkflow:
    async def test_acomplete_calls_wrapped_when_outside_workflow(self) -> None:
        """``acomplete`` forwards to ``wrapped.acomplete`` when not in a workflow."""
        wrapped = _make_mock_llm()
        llm = TemporalLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        fake_workflow = MagicMock()
        fake_workflow.in_workflow.return_value = False
        original = sys.modules.get("temporalio.workflow")
        sys.modules["temporalio.workflow"] = fake_workflow
        try:
            result = await llm.acomplete("hello", stream=False)
        finally:
            if original is None:
                del sys.modules["temporalio.workflow"]
            else:
                sys.modules["temporalio.workflow"] = original

        wrapped.acomplete.assert_awaited_once_with("hello", None, None, None, stream=False)
        assert result is wrapped.acomplete.return_value

    async def test_acomplete_delegates_outside_workflow_no_temporalio(self) -> None:
        """``acomplete`` delegates directly when ``temporalio`` is not installed."""
        wrapped = _make_mock_llm()
        llm = TemporalLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        original = sys.modules.get("temporalio.workflow")
        # Simulate ImportError by removing the module from sys.modules.
        sys.modules["temporalio.workflow"] = None  # type: ignore[assignment]
        try:
            result = await llm.acomplete("hello")
        finally:
            if original is None:
                sys.modules.pop("temporalio.workflow", None)
            else:
                sys.modules["temporalio.workflow"] = original

        wrapped.acomplete.assert_awaited_once()
        assert result is wrapped.acomplete.return_value


# ---------------------------------------------------------------------------
# model_name defaults to str(wrapped)
# ---------------------------------------------------------------------------


class TestTemporalLLMModelNameDefaults:
    def test_model_name_defaults_to_str_of_wrapped(self) -> None:
        """``__post_init__`` sets ``model_name = str(wrapped)`` when left empty."""
        wrapped = _make_mock_llm(str_repr="my-provider/gpt-4o")
        llm = TemporalLLM(wrapped=wrapped, activity_config=ModelActivityConfig())

        assert llm.model_name == "my-provider/gpt-4o"

    def test_explicit_model_name_is_kept(self) -> None:
        """An explicit ``model_name`` is never overwritten by ``__post_init__``."""
        wrapped = _make_mock_llm(str_repr="some-repr")
        llm = TemporalLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="custom-key",
        )

        assert llm.model_name == "custom-key"


# ---------------------------------------------------------------------------
# install() — basic replacement
# ---------------------------------------------------------------------------


class TestTemporalLLMInstallReplacesAgentLLM:
    def test_install_replaces_agent_llm(self) -> None:
        """``install`` wraps the agent's LLM with a ``TemporalLLM`` instance."""
        original_llm = _make_mock_llm()
        agent = _make_agent(llm=original_llm)

        TemporalLLM.install(agent, activity_config=ModelActivityConfig())

        assert isinstance(agent.llm, TemporalLLM)
        assert agent.llm.wrapped is original_llm

    def test_install_uses_provided_activity_config(self) -> None:
        """The provided ``ModelActivityConfig`` is forwarded to the ``TemporalLLM``."""
        original_llm = _make_mock_llm()
        agent = _make_agent(llm=original_llm)
        config = ModelActivityConfig(maximum_attempts=7, start_to_close_timeout=600)

        TemporalLLM.install(agent, activity_config=config)

        assert agent.llm.activity_config is config

    def test_install_uses_default_config_when_none(self) -> None:
        """``install`` defaults to ``ModelActivityConfig()`` when ``activity_config`` is ``None``."""
        original_llm = _make_mock_llm()
        agent = _make_agent(llm=original_llm)

        TemporalLLM.install(agent)

        assert isinstance(agent.llm, TemporalLLM)
        assert isinstance(agent.llm.activity_config, ModelActivityConfig)

    def test_install_sets_model_name_from_wrapped_str(self) -> None:
        """``model_name`` defaults to ``str(wrapped)`` when not supplied to ``install``."""
        original_llm = _make_mock_llm(str_repr="gpt-4o-mini")
        agent = _make_agent(llm=original_llm)

        TemporalLLM.install(agent)

        assert agent.llm.model_name == "gpt-4o-mini"

    def test_install_explicit_model_name_propagated(self) -> None:
        """An explicit ``model_name`` passed to ``install`` is applied to the wrapper."""
        original_llm = _make_mock_llm()
        agent = _make_agent(llm=original_llm)

        TemporalLLM.install(agent, model_name="registry-key")

        assert agent.llm.model_name == "registry-key"


# ---------------------------------------------------------------------------
# install() — skips already-wrapped agents
# ---------------------------------------------------------------------------


class TestTemporalLLMInstallSkipsAlreadyWrapped:
    def test_install_twice_does_not_double_wrap(self) -> None:
        """Calling ``install`` twice leaves the LLM wrapped exactly once."""
        original_llm = _make_mock_llm()
        agent = _make_agent(llm=original_llm)

        TemporalLLM.install(agent)
        first_wrapper = agent.llm

        TemporalLLM.install(agent)

        assert agent.llm is first_wrapper
        assert isinstance(first_wrapper.wrapped, MagicMock)
        assert not isinstance(first_wrapper.wrapped, TemporalLLM)

    def test_install_skips_none_llm(self) -> None:
        """``install`` is a no-op when the agent has no LLM (``llm=None``)."""
        agent = _make_agent(llm=None)

        TemporalLLM.install(agent)

        assert agent.llm is None


# ---------------------------------------------------------------------------
# install() — handoff graph traversal
# ---------------------------------------------------------------------------


class TestTemporalLLMInstallTraversesHandoffs:
    def test_install_traverses_bare_agent_handoffs(self) -> None:
        """``install`` recurses into bare ``Agent`` entries in ``agent.handoffs``."""
        root_llm = _make_mock_llm()
        child_llm = _make_mock_llm()

        child_agent = _make_agent(llm=child_llm, handoffs=None)
        root_agent = _make_agent(llm=root_llm, handoffs=[child_agent])

        TemporalLLM.install(root_agent)

        assert isinstance(root_agent.llm, TemporalLLM)
        assert isinstance(child_agent.llm, TemporalLLM)
        assert child_agent.llm.wrapped is child_llm

    def test_install_traverses_handoff_objects_via_target(self) -> None:
        """``install`` recurses into ``.target`` of handoff objects (e.g. ``Handoff``)."""
        root_llm = _make_mock_llm()
        child_llm = _make_mock_llm()

        child_agent = _make_agent(llm=child_llm, handoffs=None)

        # Simulate a Handoff dataclass: has .target but NOT .llm.
        class _FakeHandoff:
            def __init__(self, target: Any) -> None:
                self.target = target

        handoff_obj = _FakeHandoff(child_agent)

        root_agent = _make_agent(llm=root_llm, handoffs=[handoff_obj])

        TemporalLLM.install(root_agent)

        assert isinstance(root_agent.llm, TemporalLLM)
        assert isinstance(child_agent.llm, TemporalLLM)

    def test_install_handles_circular_handoff_refs(self) -> None:
        """``install`` does not loop forever on circular handoff references."""
        llm_a = _make_mock_llm()
        llm_b = _make_mock_llm()

        agent_a = _make_agent(llm=llm_a)
        agent_b = _make_agent(llm=llm_b)

        # Circular: a → b → a
        agent_a.handoffs = [agent_b]
        agent_b.handoffs = [agent_a]

        # Should complete without infinite recursion.
        TemporalLLM.install(agent_a)

        assert isinstance(agent_a.llm, TemporalLLM)
        assert isinstance(agent_b.llm, TemporalLLM)


# ---------------------------------------------------------------------------
# TemporalLLM is itself an LLM subclass
# ---------------------------------------------------------------------------


class TestTemporalLLMIsLLMSubclass:
    def test_temporal_llm_is_llm_instance(self) -> None:
        """``TemporalLLM`` satisfies the ``LLM`` ABC contract."""
        wrapped = _make_mock_llm()
        llm = TemporalLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="x",
        )

        assert isinstance(llm, LLM)

    def test_temporal_llm_has_acomplete(self) -> None:
        """``TemporalLLM`` exposes ``acomplete`` as required by the ``LLM`` ABC."""
        wrapped = _make_mock_llm()
        llm = TemporalLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="x",
        )

        assert callable(llm.acomplete)


# ---------------------------------------------------------------------------
# _dict_to_llm_response helper
# ---------------------------------------------------------------------------


class TestDictToLLMResponse:
    def test_converts_text_part(self) -> None:
        """``_dict_to_llm_response`` reconstructs a ``LLMResponseText`` part."""
        from troopai.adk.types.responses.llm_response import LLMResponseText
        from troopai.adk.workflows.temporal.llm import _dict_to_llm_response

        data = {
            "response_id": "r1",
            "model": "gpt-4o",
            "response": [{"type": "text", "text": "Hello!", "annotations": None}],
            "usage": None,
            "finish_reason": "stop",
            "timestamp": None,
        }

        result = _dict_to_llm_response(data)

        assert result.response_id == "r1"
        assert result.model == "gpt-4o"
        assert len(result.response) == 1
        assert isinstance(result.response[0], LLMResponseText)
        assert result.response[0].text == "Hello!"

    def test_converts_function_call_part(self) -> None:
        """``_dict_to_llm_response`` reconstructs a ``LLMResponseFunctionToolCall`` part."""
        from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall
        from troopai.adk.workflows.temporal.llm import _dict_to_llm_response

        data = {
            "response_id": "r2",
            "model": "gpt-4o",
            "response": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "search",
                    "arguments": '{"q": "test"}',
                    "id": None,
                    "status": None,
                }
            ],
            "usage": None,
            "finish_reason": "tool_calls",
            "timestamp": None,
        }

        result = _dict_to_llm_response(data)

        assert len(result.response) == 1
        part = result.response[0]
        assert isinstance(part, LLMResponseFunctionToolCall)
        assert part.name == "search"

    def test_unknown_part_type_raises_value_error(self) -> None:
        """Unknown part type raises ValueError instead of silently truncating the response.

        Regression: the old code logged a warning and dropped the unknown part, committing
        a truncated response to durable history. On replay, Temporal would reuse the
        truncated response forever. Raising ValueError causes the activity to fail and
        retry (or fail fast for non-retryable errors), which is safer.
        """
        from troopai.adk.workflows.temporal.llm import _dict_to_llm_response

        data = {
            "response_id": "r3",
            "model": "x",
            "response": [{"type": "unknown_future_part", "data": "..."}],
            "usage": None,
            "finish_reason": None,
            "timestamp": None,
        }

        with pytest.raises(ValueError, match="unknown_future_part"):
            _dict_to_llm_response(data)

    def test_round_trips_usage(self) -> None:
        """Usage survives the round-trip with nested token details rebuilt.

        Regression: _dict_to_llm_response hardcoded usage=None, so durable runs
        never accumulated usage (UsageLimitExceeded never fired, tenant dollar
        budgets never charged).
        """
        import dataclasses

        from troopai.adk.types.tokens.llm_usage import LLMUsage
        from troopai.adk.types.tokens.tokens import InputTokensDetails
        from troopai.adk.workflows.temporal.llm import _dict_to_llm_response

        usage = LLMUsage(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=InputTokensDetails(cached_tokens=10, cache_creation_input_tokens=0),
        )
        data = {
            "response_id": "r",
            "model": "m",
            "response": [],
            "usage": dataclasses.asdict(usage),
            "finish_reason": "stop",
            "timestamp": None,
        }

        result = _dict_to_llm_response(data)

        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        # Nested detail rebuilt as the real type (not a dict) so accumulation works.
        assert isinstance(result.usage.input_tokens_details, InputTokensDetails)
        assert result.usage.input_tokens_details.cached_tokens == 10
        combined = result.usage + result.usage
        assert combined.total_tokens == 300

    def test_round_trips_text_annotations(self) -> None:
        """Text-part annotations rebuild as ``LLMResponseAnnotation`` so to_param works.

        Regression: asdict flattened each annotation to a dict; storing the dict
        made a later ``to_param()`` (which calls asdict on each annotation) raise.
        """
        from troopai.adk.types.responses.llm_response import (
            LLMResponseAnnotation,
            LLMResponseText,
        )
        from troopai.adk.workflows.temporal.llm import _dict_to_llm_response

        data = {
            "response_id": "r",
            "model": "m",
            "response": [
                {
                    "type": "text",
                    "text": "cited",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "http://x",
                            "title": "t",
                            "start_index": 0,
                            "end_index": 5,
                        }
                    ],
                }
            ],
            "usage": None,
            "finish_reason": "stop",
            "timestamp": None,
        }

        result = _dict_to_llm_response(data)

        part = result.response[0]
        assert isinstance(part, LLMResponseText)
        assert part.annotations is not None
        assert isinstance(part.annotations[0], LLMResponseAnnotation)
        assert part.annotations[0].url == "http://x"
        # The original bug bites here: to_param() calls asdict on each annotation
        # — it must not raise on a reconstructed part.
        part.to_param()


# ---------------------------------------------------------------------------
# model_name derivation (stable key, not an object address)
# ---------------------------------------------------------------------------


class TestTemporalLLMModelNameDerivation:
    """``model_name`` must derive from the wrapped LLM's stable ``model`` id.

    Regression: it defaulted to ``str(wrapped)`` — a ``<LiteLLM object at
    0x...>`` address for providers without a custom ``__str__``. That address
    is non-deterministic across processes, so it never matches
    ``register_model()`` on the worker and breaks replay.
    """

    def test_derives_from_wrapped_model_not_address(self) -> None:
        """When the wrapped LLM exposes ``.model``, it becomes the registry key."""
        llm = TemporalLLM(wrapped=_AddressLLM(), activity_config=ModelActivityConfig())
        assert llm.model_name == "claude-address-x"
        assert "0x" not in llm.model_name

    def test_falls_back_to_str_without_model(self) -> None:
        """No ``.model`` → documented ``str(wrapped)`` fallback."""
        wrapped = _NoModelLLM()
        llm = TemporalLLM(wrapped=wrapped, activity_config=ModelActivityConfig())
        assert llm.model_name == str(wrapped)

    def test_explicit_model_name_wins(self) -> None:
        """An explicit ``model_name`` is never overridden by derivation."""
        llm = TemporalLLM(
            wrapped=_AddressLLM(),
            activity_config=ModelActivityConfig(),
            model_name="explicit-key",
        )
        assert llm.model_name == "explicit-key"

    def test_install_derives_from_wrapped_model(self) -> None:
        """``install`` also derives from ``.model`` (not the object address)."""
        agent = _make_agent(llm=_AddressLLM())
        TemporalLLM.install(agent)
        assert agent.llm.model_name == "claude-address-x"
        assert "0x" not in agent.llm.model_name


# ---------------------------------------------------------------------------
# Cost delegation (tenant budgets must reach the wrapped provider)
# ---------------------------------------------------------------------------


class TestTemporalLLMDelegatesCost:
    """``TemporalLLM`` must forward cost/estimate_cost to the wrapped LLM.

    Regression: the base ``LLM.cost`` returns ``None``, so without an override
    a durable run's tenant dollar budget is silently never charged.
    """

    def test_cost_delegates_to_wrapped(self) -> None:
        """``cost`` returns the wrapped provider's price for the call."""
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        wrapped = MagicMock(spec=LLM)
        wrapped.model = "gpt-4o"
        wrapped.cost.return_value = 0.42
        llm = TemporalLLM(wrapped=wrapped, activity_config=ModelActivityConfig())
        usage = LLMUsage(input_tokens=10, output_tokens=5)

        assert llm.cost("gpt-4o", usage) == 0.42
        wrapped.cost.assert_called_once_with("gpt-4o", usage)

    def test_estimate_cost_delegates_to_wrapped(self) -> None:
        """``estimate_cost`` forwards args and returns the wrapped estimate."""
        wrapped = MagicMock(spec=LLM)
        wrapped.model = "gpt-4o"
        sentinel = object()
        wrapped.estimate_cost.return_value = sentinel
        llm = TemporalLLM(wrapped=wrapped, activity_config=ModelActivityConfig())

        result = llm.estimate_cost([], "gpt-4o", max_output_tokens=100)

        assert result is sentinel
        wrapped.estimate_cost.assert_called_once_with([], "gpt-4o", max_output_tokens=100)

    def test_cost_none_from_wrapped_is_passed_through(self) -> None:
        """A wrapped LLM with no price table yields ``None`` (never raises)."""
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        wrapped = MagicMock(spec=LLM)
        wrapped.model = "gpt-4o"
        wrapped.cost.return_value = None
        llm = TemporalLLM(wrapped=wrapped, activity_config=ModelActivityConfig())

        assert llm.cost("gpt-4o", LLMUsage(input_tokens=1, output_tokens=1)) is None
