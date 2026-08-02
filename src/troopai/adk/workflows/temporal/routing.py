"""Per-tenant Temporal task-queue routing.

Routes a tenant's workflow onto a tenant-specific task queue at dispatch
time. Because Temporal activities inherit their workflow's task queue,
this isolates all of a tenant's model/tool activities onto dedicated
worker pools — no activity-path changes are required.

Temporal task queues: https://docs.temporal.io/task-queue
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from temporalio.client import Client, WorkflowHandle

logger = logging.getLogger(__name__)


@runtime_checkable
class TenantTaskQueueRouter(Protocol):
    """Maps a tenant identifier to a Temporal task-queue name."""

    def resolve(self, tenant_id: str | None) -> str:
        """Return the task-queue name for ``tenant_id``."""
        ...


@dataclass(frozen=True)
class MappingTaskQueueRouter:
    """Router backed by a static ``tenant_id`` -> queue mapping.

    Unknown or ``None`` tenants fall back to ``default``.

    Attributes:
        mapping: Explicit per-tenant queue assignments.
        default: Queue for tenants absent from ``mapping`` (and for
            untenanted dispatches).
    """

    mapping: Mapping[str, str]
    """Explicit per-tenant queue assignments."""
    default: str
    """Fallback queue for unmapped / untenanted dispatches."""

    def resolve(self, tenant_id: str | None) -> str:
        """Return the task-queue name for ``tenant_id``.

        Args:
            tenant_id: The tenant to look up, or ``None`` for untenanted
                dispatches.

        Returns:
            The mapped queue name, or ``default`` when the tenant is absent
            from ``mapping`` or is ``None``.
        """
        if tenant_id is None:
            return self.default
        return self.mapping.get(tenant_id, self.default)


async def start_tenant_workflow(
    client: Client,
    workflow: Any,
    *,
    tenant_id: str | None,
    router: TenantTaskQueueRouter,
    id: str,
    **kwargs: Any,
) -> WorkflowHandle:
    """Start ``workflow`` on the tenant's task queue.

    Resolves the queue via ``router`` and forwards everything to
    ``client.start_workflow``. Activities inherit the workflow's task
    queue, so the whole run is isolated onto the tenant's worker pool.

    **Child workflow tenant isolation**: any child workflow started *inside*
    the routed workflow without using this helper will NOT be isolated onto
    the tenant's task queue — it will land on the default queue instead.
    To preserve per-tenant isolation for fan-out, use this helper (or pass
    ``task_queue=router.resolve(tenant_id)`` explicitly) when starting each
    child workflow.

    Args:
        client: A connected Temporal client.
        workflow: The workflow function/class or its name.
        tenant_id: Tenant whose queue should run this workflow.
        router: Resolver mapping ``tenant_id`` -> task-queue name.
        id: Workflow id (Temporal requires one).
        **kwargs: Forwarded to ``client.start_workflow``. Pass the workflow
            argument(s) the Temporal way: ``arg=<single>`` or
            ``args=[<multiple>]``.

    Returns:
        The started workflow handle.
    """
    task_queue = router.resolve(tenant_id)
    logger.info(
        "dispatch tenant=%s workflow_id=%s task_queue=%s",
        tenant_id,
        id,
        task_queue,
    )
    return await client.start_workflow(workflow, id=id, task_queue=task_queue, **kwargs)


__all__ = ["MappingTaskQueueRouter", "TenantTaskQueueRouter", "start_tenant_workflow"]
