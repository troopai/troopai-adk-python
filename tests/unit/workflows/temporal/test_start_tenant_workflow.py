from __future__ import annotations

from typing import Any

from troopai.adk.workflows.temporal.routing import (
    MappingTaskQueueRouter,
    start_tenant_workflow,
)


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(self, workflow: Any, *args: Any, **kwargs: Any) -> str:
        self.calls.append({"workflow": workflow, "args": args, "kwargs": kwargs})
        return "handle"


async def test_dispatch_uses_resolved_queue() -> None:
    client = _RecordingClient()
    router = MappingTaskQueueRouter(mapping={"premium": "premium-q"}, default="shared")

    # Workflow args are forwarded the Temporal way (arg=/args=), not positionally.
    await start_tenant_workflow(client, "MyWorkflow", arg="arg1", tenant_id="premium", router=router, id="wf-1")

    call = client.calls[0]
    assert call["kwargs"]["task_queue"] == "premium-q"
    assert call["kwargs"]["id"] == "wf-1"
    assert call["kwargs"]["arg"] == "arg1"


async def test_unknown_tenant_dispatches_to_default_queue() -> None:
    client = _RecordingClient()
    router = MappingTaskQueueRouter(mapping={}, default="shared")
    await start_tenant_workflow(client, "W", tenant_id="x", router=router, id="wf-2")
    assert client.calls[0]["kwargs"]["task_queue"] == "shared"


async def test_none_tenant_dispatches_to_default_queue() -> None:
    client = _RecordingClient()
    router = MappingTaskQueueRouter(mapping={"premium": "premium-q"}, default="shared")
    await start_tenant_workflow(client, "W", tenant_id=None, router=router, id="wf-3")
    assert client.calls[0]["kwargs"]["task_queue"] == "shared"
