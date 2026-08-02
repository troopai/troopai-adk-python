"""Worker wiring plugin for TroopAI ADK Temporal integration.

:class:`TroopAITemporalPlugin` implements BOTH of temporalio's plugin
surfaces — ``temporalio.client.Plugin`` and ``temporalio.worker.Plugin``
— so one instance wires the TroopAI data converter into the client and
the sandboxed workflow runner into every worker created from it, and it
composes with other Temporal plugins via the standard ``plugins=[...]``
lists::

    from temporalio.client import Client
    from temporalio.worker import Worker

    from troopai.adk.workflows.temporal.plugin import TroopAITemporalPlugin
    from troopai.adk.llms import LiteLLM

    plugin = TroopAITemporalPlugin()
    plugin.register_model("gpt-4o", LiteLLM(model="gpt-4o"))

    client = await Client.connect("localhost:7233", plugins=[plugin])
    # Client plugins that also implement the worker surface are applied
    # to workers created from that client automatically — no need to
    # repeat plugins= on the Worker.
    worker = Worker(client, task_queue="my-queue", workflows=[...], activities=[...])

For manual wiring without the plugin chain, :meth:`build_worker_kwargs`
returns the ``workflow_runner`` entry for the ``Worker`` constructor;
the data converter is a *client* setting — pass
:func:`~troopai.adk.workflows.temporal.serialization.build_troopai_data_converter`
to ``Client.connect(data_converter=...)``.

References:
    Temporal Python worker docs:
    https://docs.temporal.io/develop/python/core-application#run-a-dev-worker
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, override

# Module-top import is deliberate: this package requires the temporal
# extra before import (see the package docstring), and ABC bases must
# exist at class-definition time.
from temporalio.client import Plugin as ClientPlugin
from temporalio.worker import Plugin as WorkerPlugin

from troopai.adk.workflows.temporal.activity import register_model as activity_register_model
from troopai.adk.workflows.temporal.determinism import (
    DEFAULT_PASSTHROUGH_MODULES,
    build_sandbox_restrictions,
)
from troopai.adk.workflows.temporal.serialization import build_troopai_data_converter

if TYPE_CHECKING:
    from temporalio.client import ClientConfig, ConnectConfig, ServiceClient, WorkflowHistory
    from temporalio.worker import (
        Replayer,
        ReplayerConfig,
        Worker,
        WorkerConfig,
        WorkflowReplayResult,
    )

    from troopai.adk.llms.llm import LLM

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TroopAITemporalPlugin(ClientPlugin, WorkerPlugin):
    """Client + worker plugin wiring TroopAI Temporal workflows.

    On the client side, installs the TroopAI data converter. On the
    worker side, installs a sandboxed workflow runner whose passthrough
    modules cover the TroopAI ADK and its dependencies, and syncs the
    model registry into the activity-level registry. Replayer
    configuration mirrors the worker so replays decode and execute the
    same way live runs did.

    Attributes:
        extra_passthrough_modules: Additional module names to pass through the
            Temporal sandbox beyond the built-in defaults.
        model_registry: Mapping from model name to
            :class:`~troopai.adk.llms.llm.LLM` instance.  Populated via
            :meth:`register_model`.
        passthrough_modules: Complete set of sandbox passthrough modules,
            computed from :data:`~troopai.adk.workflows.temporal.determinism.DEFAULT_PASSTHROUGH_MODULES`
            plus :attr:`extra_passthrough_modules` in ``__post_init__``.
    """

    extra_passthrough_modules: Sequence[str] = dataclasses.field(default_factory=tuple)
    """Additional sandbox passthrough modules beyond the built-in defaults."""

    model_registry: dict[str, LLM] = dataclasses.field(default_factory=dict)
    """Mapping from model name to :class:`~troopai.adk.llms.llm.LLM` instance."""

    passthrough_modules: tuple[str, ...] = dataclasses.field(init=False)
    """Complete set of sandbox passthrough modules, computed in ``__post_init__``."""

    def __post_init__(self) -> None:
        self.passthrough_modules = DEFAULT_PASSTHROUGH_MODULES + tuple(self.extra_passthrough_modules)
        logger.info(
            "TroopAITemporalPlugin initialized with %d passthrough modules",
            len(self.passthrough_modules),
        )

    def register_model(self, name: str, llm: LLM) -> None:
        """Register an LLM instance by name.

        Adds the instance to :attr:`model_registry` and also registers it
        in the module-level activity registry so that
        :func:`~troopai.adk.workflows.temporal.activity.invoke_model_activity`
        can resolve it at runtime.

        Args:
            name: Registry key — must match the ``model_name`` field used in
                workflow activity inputs.
            llm: The :class:`~troopai.adk.llms.llm.LLM` instance to register.
        """
        self.model_registry[name] = llm
        activity_register_model(name, llm)
        logger.info("TroopAITemporalPlugin registered model %r", name)

    # -- temporalio.client.Plugin surface ---------------------------------

    @override
    def configure_client(self, config: ClientConfig) -> ClientConfig:
        """Install the TroopAI data converter on the client.

        The converter is a client-level setting in temporalio; workers
        and workflow handles created from the client inherit it.
        """
        config["data_converter"] = build_troopai_data_converter()
        logger.debug("TroopAITemporalPlugin: data converter installed on client config")
        return config

    @override
    async def connect_service_client(
        self,
        config: ConnectConfig,
        next: Callable[[ConnectConfig], Awaitable[ServiceClient]],
    ) -> ServiceClient:
        """Continue the connection chain unchanged."""
        return await next(config)

    # -- temporalio.worker.Plugin surface ----------------------------------

    @override
    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        """Install the sandboxed workflow runner and sync registered models."""
        from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

        for name, llm in self.model_registry.items():
            activity_register_model(name, llm)
            logger.debug("configure_worker: synced model %r to activity registry", name)

        restrictions = build_sandbox_restrictions(tuple(self.extra_passthrough_modules))
        config["workflow_runner"] = SandboxedWorkflowRunner(restrictions=restrictions)
        logger.debug("TroopAITemporalPlugin: sandboxed workflow runner installed on worker config")
        return config

    @override
    async def run_worker(self, worker: Worker, next: Callable[[Worker], Awaitable[None]]) -> None:
        """Continue the worker execution chain unchanged."""
        await next(worker)

    @override
    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:
        """Mirror the worker runner and client converter for replays.

        A replay must decode payloads and sandbox workflow code exactly
        as the live worker did, or histories fail to verify.
        """
        from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

        restrictions = build_sandbox_restrictions(tuple(self.extra_passthrough_modules))
        config["workflow_runner"] = SandboxedWorkflowRunner(restrictions=restrictions)
        config["data_converter"] = build_troopai_data_converter()
        return config

    @override
    def run_replayer(
        self,
        replayer: Replayer,
        histories: AsyncIterator[WorkflowHistory],
        next: Callable[
            [Replayer, AsyncIterator[WorkflowHistory]],
            AbstractAsyncContextManager[AsyncIterator[WorkflowReplayResult]],
        ],
    ) -> AbstractAsyncContextManager[AsyncIterator[WorkflowReplayResult]]:
        """Continue the replayer chain unchanged."""
        return next(replayer, histories)

    # -- Manual wiring ------------------------------------------------------

    def build_worker_kwargs(self) -> dict[str, Any]:
        """Return kwargs to pass directly to a Temporal ``Worker`` constructor.

        Registers all models from :attr:`model_registry` into the activity
        registry and assembles the ``workflow_runner`` entry. The data
        converter is NOT included — ``Worker`` does not accept one; it is
        a client setting (``Client.connect(data_converter=...)`` or the
        plugin chain via :meth:`configure_client`).

        Returns:
            A ``dict`` with the ``"workflow_runner"`` key ready to unpack
            into ``Worker(..., **plugin.build_worker_kwargs())``.
        """
        from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

        for name, llm in self.model_registry.items():
            activity_register_model(name, llm)
            logger.info("build_worker_kwargs: synced model %r to activity registry", name)

        restrictions = build_sandbox_restrictions(tuple(self.extra_passthrough_modules))
        runner = SandboxedWorkflowRunner(restrictions=restrictions)

        logger.info("build_worker_kwargs: workflow_runner assembled")
        return {"workflow_runner": runner}
