"""Tests for :mod:`troopai.adk.workflows.engine`.

Covers:
- Default field values for :class:`ModelActivityConfig`.
- Custom construction of :class:`ModelActivityConfig`.
- Default field values for :class:`ToolActivityConfig`.
- Runtime-checkability of :class:`DurableEngine`.
- Frozen (immutable) enforcement on :class:`ModelActivityConfig`.
"""

from __future__ import annotations

import dataclasses

import pytest

from troopai.adk.workflows.engine import (
    DurableEngine,
    ModelActivityConfig,
    ToolActivityConfig,
)


class TestModelActivityConfigDefaults:
    def test_model_activity_config_defaults(self) -> None:
        cfg = ModelActivityConfig()

        assert cfg.start_to_close_timeout == 300
        assert cfg.heartbeat_timeout == 60
        assert cfg.maximum_attempts == 1
        assert cfg.initial_interval == 1
        assert cfg.backoff_coefficient == 2.0
        assert cfg.non_retryable_error_types == ("ClientError",)


class TestModelActivityConfigCustom:
    def test_model_activity_config_custom(self) -> None:
        cfg = ModelActivityConfig(
            start_to_close_timeout=120,
            heartbeat_timeout=30,
            maximum_attempts=5,
            initial_interval=2,
            backoff_coefficient=1.5,
            non_retryable_error_types=("ClientError", "ValidationError"),
        )

        assert cfg.start_to_close_timeout == 120
        assert cfg.heartbeat_timeout == 30
        assert cfg.maximum_attempts == 5
        assert cfg.initial_interval == 2
        assert cfg.backoff_coefficient == 1.5
        assert cfg.non_retryable_error_types == ("ClientError", "ValidationError")


class TestToolActivityConfigDefaults:
    def test_tool_activity_config_defaults(self) -> None:
        cfg = ToolActivityConfig()

        assert cfg.start_to_close_timeout == 30
        assert cfg.maximum_attempts == 1


class TestActivityConfigNoRetryByDefault:
    """Default activity configs MUST NOT retry without explicit opt-in.

    Regression: a retry re-sends the full prompt (LLM) or re-runs the call
    (tool) and re-bills tokens.  The cost-conservative default is a single
    attempt; developers raise ``maximum_attempts`` to opt into retries.
    """

    def test_model_config_default_does_not_retry(self) -> None:
        assert ModelActivityConfig().maximum_attempts == 1

    def test_tool_config_default_does_not_retry(self) -> None:
        assert ToolActivityConfig().maximum_attempts == 1

    def test_model_config_retry_is_opt_in(self) -> None:
        assert ModelActivityConfig(maximum_attempts=3).maximum_attempts == 3

    def test_tool_config_retry_is_opt_in(self) -> None:
        assert ToolActivityConfig(maximum_attempts=2).maximum_attempts == 2


class TestDurableEngineProtocol:
    def test_durable_engine_protocol_is_runtime_checkable(self) -> None:
        # A minimal concrete class that satisfies the Protocol structurally.
        class _FakeEngine:
            def wrap_llm(self, llm, *, config=None):  # type: ignore[override]
                _ = config
                return llm

            def wrap_tool(self, tool, *, config=None):  # type: ignore[override]
                _ = config
                return tool

            def in_durable_context(self) -> bool:
                return False

        engine = _FakeEngine()
        assert isinstance(engine, DurableEngine)

    def test_non_conforming_object_is_not_durable_engine(self) -> None:
        class _NotAnEngine:
            pass

        assert not isinstance(_NotAnEngine(), DurableEngine)


class TestModelActivityConfigFrozen:
    def test_model_activity_config_frozen(self) -> None:
        cfg = ModelActivityConfig()

        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.maximum_attempts = 99  # type: ignore[misc]


class TestTemporalDurableEngine:
    """TemporalDurableEngine must satisfy the DurableEngine Protocol.

    Regression: TemporalLLM and RestateLLM are LLM wrappers, not DurableEngine
    implementations.  Adding TemporalDurableEngine provides the concrete
    DurableEngine facade for the Temporal backend.
    """

    def test_temporal_durable_engine_is_durable_engine(self) -> None:
        """TemporalDurableEngine is a runtime-checkable DurableEngine."""
        pytest.importorskip("temporalio")
        from troopai.adk.workflows.temporal.engine import TemporalDurableEngine

        engine = TemporalDurableEngine()
        assert isinstance(engine, DurableEngine)

    def test_temporal_durable_engine_wrap_llm_returns_temporal_llm(self) -> None:
        """wrap_llm returns a TemporalLLM wrapping the supplied LLM."""
        pytest.importorskip("temporalio")
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM
        from troopai.adk.workflows.temporal.engine import TemporalDurableEngine
        from troopai.adk.workflows.temporal.llm import TemporalLLM

        mock_llm = MagicMock(spec=LLM)
        engine = TemporalDurableEngine()
        wrapped = engine.wrap_llm(mock_llm, config=ModelActivityConfig())

        assert isinstance(wrapped, TemporalLLM)
        assert wrapped.wrapped is mock_llm

    def test_temporal_durable_engine_in_durable_context_outside_workflow(self) -> None:
        """in_durable_context returns False when not inside a Temporal workflow."""
        import sys
        from unittest.mock import MagicMock

        pytest.importorskip("temporalio")
        from troopai.adk.workflows.temporal.engine import TemporalDurableEngine

        engine = TemporalDurableEngine()

        fake_wf = MagicMock()
        fake_wf.in_workflow.return_value = False
        original = sys.modules.get("temporalio.workflow")
        sys.modules["temporalio.workflow"] = fake_wf
        try:
            result = engine.in_durable_context()
        finally:
            if original is None:
                sys.modules.pop("temporalio.workflow", None)
            else:
                sys.modules["temporalio.workflow"] = original

        assert result is False

    def test_wrap_tool_preserves_name_and_schema(self) -> None:
        """wrap_tool preserves the tool's real name and schema.

        Regression: it fed the generic ``on_invoke(ctx, input)`` closure to
        ``activity_tool``, collapsing every tool to the name ``on_invoke_tool``
        with a ``(ctx, input)``-derived schema — a name collision plus schema
        degradation, and an undecorated closure that ``execute_activity``
        rejects at runtime.
        """
        pytest.importorskip("temporalio")
        from troopai.adk.tools.function_tool import function_tool
        from troopai.adk.workflows.temporal.engine import TemporalDurableEngine

        async def lookup(city: str) -> str:
            return f"weather in {city}"

        tool = function_tool(lookup)
        engine = TemporalDurableEngine()
        wrapped = engine.wrap_tool(tool, config=ToolActivityConfig())

        assert wrapped.name == "lookup"
        assert wrapped.name != "on_invoke_tool"
        schema = wrapped.get_json_schema()
        assert "city" in schema.get("properties", {})


class TestRestateDurableEngine:
    """RestateDurableEngine must satisfy the DurableEngine Protocol."""

    def test_restate_durable_engine_is_durable_engine(self) -> None:
        """RestateDurableEngine is a runtime-checkable DurableEngine."""
        import importlib
        import sys
        import types

        fake_restate = types.ModuleType("restate")
        original = sys.modules.get("restate")
        sys.modules["restate"] = fake_restate
        try:
            import troopai.adk.workflows.restate.engine as restate_engine_mod

            importlib.reload(restate_engine_mod)
            engine = restate_engine_mod.RestateDurableEngine()
            is_engine = isinstance(engine, DurableEngine)
        finally:
            if original is None:
                sys.modules.pop("restate", None)
            else:
                sys.modules["restate"] = original
            importlib.reload(restate_engine_mod)

        assert is_engine

    def test_restate_durable_engine_wrap_llm_returns_restate_llm(self) -> None:
        """wrap_llm returns a RestateLLM wrapping the supplied LLM."""
        import importlib
        import sys
        import types
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM

        fake_restate = types.ModuleType("restate")
        original = sys.modules.get("restate")
        sys.modules["restate"] = fake_restate
        try:
            import troopai.adk.workflows.restate.engine as restate_engine_mod
            import troopai.adk.workflows.restate.llm as restate_llm_mod

            importlib.reload(restate_llm_mod)
            importlib.reload(restate_engine_mod)
            mock_llm = MagicMock(spec=LLM)
            engine = restate_engine_mod.RestateDurableEngine()
            wrapped = engine.wrap_llm(mock_llm, config=ModelActivityConfig())
            is_restate_llm = isinstance(wrapped, restate_llm_mod.RestateLLM)
            wrapped_inner = wrapped.wrapped
        finally:
            if original is None:
                sys.modules.pop("restate", None)
            else:
                sys.modules["restate"] = original
            importlib.reload(restate_llm_mod)
            importlib.reload(restate_engine_mod)

        assert is_restate_llm
        assert wrapped_inner is mock_llm
