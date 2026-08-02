"""Tests for :mod:`troopai.adk.workflows.temporal.plugin`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

temporalio = pytest.importorskip("temporalio")

from troopai.adk.llms.llm import LLM
from troopai.adk.workflows.temporal.determinism import DEFAULT_PASSTHROUGH_MODULES
from troopai.adk.workflows.temporal.plugin import TroopAITemporalPlugin


class TestPluginDefaultPassthroughModules:
    def test_plugin_default_passthrough_modules(self) -> None:
        """Default passthrough_modules matches DEFAULT_PASSTHROUGH_MODULES exactly."""
        plugin = TroopAITemporalPlugin()
        assert plugin.passthrough_modules == DEFAULT_PASSTHROUGH_MODULES

    def test_default_includes_pydantic(self) -> None:
        plugin = TroopAITemporalPlugin()
        assert "pydantic" in plugin.passthrough_modules

    def test_default_includes_litellm(self) -> None:
        plugin = TroopAITemporalPlugin()
        assert "litellm" in plugin.passthrough_modules

    def test_default_includes_troopai(self) -> None:
        plugin = TroopAITemporalPlugin()
        assert "troopai" in plugin.passthrough_modules


class TestPluginCustomPassthroughModules:
    def test_plugin_custom_passthrough_modules(self) -> None:
        """Extra modules are appended after the defaults."""
        plugin = TroopAITemporalPlugin(extra_passthrough_modules=["numpy", "scipy"])
        assert "numpy" in plugin.passthrough_modules
        assert "scipy" in plugin.passthrough_modules

    def test_defaults_still_present_with_extras(self) -> None:
        plugin = TroopAITemporalPlugin(extra_passthrough_modules=["pandas"])
        for module in DEFAULT_PASSTHROUGH_MODULES:
            assert module in plugin.passthrough_modules

    def test_extra_appended_after_defaults(self) -> None:
        plugin = TroopAITemporalPlugin(extra_passthrough_modules=["mylib"])
        default_count = len(DEFAULT_PASSTHROUGH_MODULES)
        assert plugin.passthrough_modules[:default_count] == DEFAULT_PASSTHROUGH_MODULES
        assert plugin.passthrough_modules[default_count:] == ("mylib",)

    def test_passthrough_is_tuple(self) -> None:
        plugin = TroopAITemporalPlugin(extra_passthrough_modules=["x"])
        assert isinstance(plugin.passthrough_modules, tuple)


class TestPluginRegisterModel:
    def test_plugin_register_model(self) -> None:
        """Registered model appears in model_registry."""
        plugin = TroopAITemporalPlugin()
        mock_llm = MagicMock(spec=LLM)
        plugin.register_model("my-model", mock_llm)
        assert plugin.model_registry["my-model"] is mock_llm

    def test_register_model_propagates_to_activity_registry(self) -> None:
        """register_model() also updates the activity module registry."""
        from troopai.adk.workflows.temporal.activity import get_model

        plugin = TroopAITemporalPlugin()
        mock_llm = MagicMock(spec=LLM)
        plugin.register_model("propagation-test-model", mock_llm)
        assert get_model("propagation-test-model") is mock_llm

    def test_register_model_overwrites(self) -> None:
        """Registering the same name twice replaces the previous entry."""
        plugin = TroopAITemporalPlugin()
        first = MagicMock(spec=LLM)
        second = MagicMock(spec=LLM)
        plugin.register_model("overwrite-model", first)
        plugin.register_model("overwrite-model", second)
        assert plugin.model_registry["overwrite-model"] is second


class TestPluginBuildWorkerKwargsKeys:
    def test_plugin_build_worker_kwargs_keys(self) -> None:
        """build_worker_kwargs() returns only keys Worker() actually accepts.

        Worker has no data_converter parameter — the converter is a client
        setting — so it must NOT appear here or Worker(**kwargs) raises.
        """
        plugin = TroopAITemporalPlugin()
        kwargs = plugin.build_worker_kwargs()
        assert set(kwargs.keys()) == {"workflow_runner"}
        import inspect

        from temporalio.worker import Worker

        worker_params = inspect.signature(Worker.__init__).parameters
        for key in kwargs:
            assert key in worker_params, f"Worker() does not accept {key!r}"

    def test_build_worker_kwargs_workflow_runner_type(self) -> None:
        """workflow_runner is a SandboxedWorkflowRunner instance."""
        from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

        plugin = TroopAITemporalPlugin()
        kwargs = plugin.build_worker_kwargs()
        assert isinstance(kwargs["workflow_runner"], SandboxedWorkflowRunner)

    def test_build_worker_kwargs_registers_models(self) -> None:
        """Models pre-loaded in model_registry are synced to the activity registry."""
        from troopai.adk.workflows.temporal.activity import get_model

        mock_llm = MagicMock(spec=LLM)
        plugin = TroopAITemporalPlugin(model_registry={"preloaded-model": mock_llm})
        plugin.build_worker_kwargs()
        assert get_model("preloaded-model") is mock_llm


class TestPluginAbcSurfaces:
    """TroopAITemporalPlugin composes via temporalio's plugin chains."""

    def test_implements_both_plugin_abcs(self) -> None:
        from temporalio.client import Plugin as ClientPlugin
        from temporalio.worker import Plugin as WorkerPlugin

        plugin = TroopAITemporalPlugin()
        assert isinstance(plugin, ClientPlugin)
        assert isinstance(plugin, WorkerPlugin)

    def test_configure_client_installs_data_converter(self) -> None:
        plugin = TroopAITemporalPlugin()
        config = plugin.configure_client({})  # type: ignore[typeddict-item]  # partial config suffices for the chain
        assert "data_converter" in config

    def test_configure_worker_installs_sandboxed_runner(self) -> None:
        from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

        plugin = TroopAITemporalPlugin()
        config = plugin.configure_worker({})  # type: ignore[typeddict-item]  # partial config suffices for the chain
        assert isinstance(config["workflow_runner"], SandboxedWorkflowRunner)

    def test_configure_worker_syncs_models(self) -> None:
        from troopai.adk.workflows.temporal.activity import get_model

        mock_llm = MagicMock(spec=LLM)
        plugin = TroopAITemporalPlugin(model_registry={"abc-sync-model": mock_llm})
        plugin.configure_worker({})  # type: ignore[typeddict-item]  # partial config suffices for the chain
        assert get_model("abc-sync-model") is mock_llm

    def test_configure_replayer_mirrors_worker_and_client(self) -> None:
        from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

        plugin = TroopAITemporalPlugin()
        config = plugin.configure_replayer({})  # type: ignore[typeddict-item]  # partial config suffices for the chain
        assert isinstance(config["workflow_runner"], SandboxedWorkflowRunner)
        assert "data_converter" in config

    @pytest.mark.asyncio
    async def test_run_worker_delegates_to_next(self) -> None:
        plugin = TroopAITemporalPlugin()
        worker = MagicMock()
        seen: list[object] = []

        async def fake_next(w: object) -> None:
            seen.append(w)

        await plugin.run_worker(worker, fake_next)
        assert seen == [worker]
