"""Per-tenant Temporal task-queue routing.

Premium tenants get a dedicated queue (and worker pool); everyone else
shares a default queue. Run one worker per queue. This shows the dispatch
side; see examples/temporal/basic_agent.py for worker setup.
"""

from __future__ import annotations

import logging
from typing import Any

from troopai.adk.workflows.temporal.routing import (
    MappingTaskQueueRouter,
    start_tenant_workflow,
)

logger = logging.getLogger(__name__)

# Premium tenants -> dedicated queue; all others -> the shared default.
ROUTER = MappingTaskQueueRouter(
    mapping={"premium-tenant": "troopai-premium"},
    default="troopai-shared",
)


async def dispatch(
    client: Any,
    workflow: Any,
    prompt: str,
    *,
    tenant_id: str,
    workflow_id: str,
) -> Any:
    """Start a tenant's workflow on its task queue.

    Workflow arguments are passed the Temporal way (``arg=`` for a single
    argument), since ``start_tenant_workflow`` forwards ``**kwargs`` to
    ``client.start_workflow``.
    """
    return await start_tenant_workflow(
        client,
        workflow,
        arg=prompt,
        tenant_id=tenant_id,
        router=ROUTER,
        id=workflow_id,
    )
