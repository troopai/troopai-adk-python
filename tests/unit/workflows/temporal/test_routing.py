from __future__ import annotations

from troopai.adk.workflows.temporal.routing import (
    MappingTaskQueueRouter,
    TenantTaskQueueRouter,
    start_tenant_workflow,
)


def test_mapping_router_is_a_router() -> None:
    router = MappingTaskQueueRouter(mapping={}, default="shared")
    assert isinstance(router, TenantTaskQueueRouter)


def test_resolves_mapped_tenant() -> None:
    router = MappingTaskQueueRouter(mapping={"premium": "premium-q"}, default="shared")
    assert router.resolve("premium") == "premium-q"


def test_unknown_tenant_falls_back_to_default() -> None:
    router = MappingTaskQueueRouter(mapping={"premium": "premium-q"}, default="shared")
    assert router.resolve("free") == "shared"


def test_none_tenant_uses_default() -> None:
    router = MappingTaskQueueRouter(mapping={"premium": "premium-q"}, default="shared")
    assert router.resolve(None) == "shared"


def test_start_tenant_workflow_docstring_warns_about_child_isolation() -> None:
    """start_tenant_workflow docstring must warn that child workflows are not auto-isolated.

    Design-fork finding: child workflows started inside the routed workflow
    without using this helper land on the default queue.  The docstring must
    document this so callers know to propagate the tenant_id manually.
    """
    doc = start_tenant_workflow.__doc__ or ""
    assert "child" in doc.lower(), "start_tenant_workflow docstring must mention child workflow isolation"
    assert "task_queue" in doc or "tenant" in doc.lower(), (
        "docstring must guide callers on propagating tenant isolation to child workflows"
    )
