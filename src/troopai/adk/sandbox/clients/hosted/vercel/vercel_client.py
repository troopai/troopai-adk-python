"""VercelSandboxClient — hosted bridge backed by the Vercel sandbox provider.

REST-only path (httpx). Users who want each provider's official Python
SDK can subclass and override ``create`` / ``resume``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    ExecTransportError,
    SandboxConfigurationError,
    SandboxStartFailed,
)
from troopai.adk.sandbox.clients.base import (
    reject_unsupported_snapshot_store,
    warn_discarded_snapshot,
)
from troopai.adk.sandbox.clients.hosted.remote_vm import (
    RemoteVMSandboxClient,
    RemoteVMSandboxClientOptions,
    build_httpx_client,
    map_http_error_to_sandbox_error,
    retry_async_call,
)
from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_session import (
    RemoteVMSandboxSession,
)
from troopai.adk.sandbox.clients.session import BaseSandboxSession
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor
from troopai.adk.types.sandbox.session_state import SandboxSessionState

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

logger = logging.getLogger(__name__)

__all__ = ["VercelSandboxClient", "VercelSandboxClientOptions"]

_DEFAULT_BASE_URL = "https://api.vercel.com"


class VercelSandboxClientOptions(RemoteVMSandboxClientOptions):
    """Options for the Vercel hosted bridge."""

    api_key: str | None = None
    """Vercel auth token / API key. Required at create-time."""

    project_id: str | None = None
    """Vercel project ID; required if team_id is set."""

    team_id: str | None = None
    """Optional team ID query param."""


class VercelSandboxClient(RemoteVMSandboxClient):
    """Hosted-bridge client backed by Vercel."""

    backend_id = "vercel"
    # Approximate compute rate (USD/min); override per current Vercel pricing:
    # https://vercel.com/pricing
    cost = SandboxCostDescriptor(usd_per_minute=0.12)
    capabilities = SandboxBackendCapabilities(network=True, persistent=False)

    def __init__(self, *, http_client: Any = None) -> None:
        super().__init__()
        self._injected_http = http_client

    def _auth_headers(self, options: VercelSandboxClientOptions) -> dict[str, str]:
        if options.api_key is None or len(options.api_key) == 0:
            raise SandboxStartFailed(
                backend_id="vercel",
                reason="VercelSandboxClientOptions.api_key is required at create-time.",
            )
        return {
            "Authorization": "Bearer " + options.api_key,
            "Content-Type": "application/json",
        }

    @override
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: VercelSandboxClientOptions,  # type: ignore[override]
    ) -> BaseSandboxSession:
        reject_unsupported_snapshot_store(snapshot_store, self.backend_id)
        warn_discarded_snapshot(snapshot, self.backend_id, logger)
        if manifest is not None:
            logger.warning(
                "backend %r does not materialize a workspace manifest remotely; the "
                "configured manifest is discarded (select a backend that materializes "
                "manifests, or remove config.manifest)",
                self.backend_id,
            )
        del snapshot, manifest
        headers = self._auth_headers(options)
        owns_http = self._injected_http is None
        http = self._injected_http or build_httpx_client(
            base_url=options.base_url or _DEFAULT_BASE_URL,
            headers=headers,
            timeout=options.request_timeout,
        )
        try:
            body: dict[str, Any] = {"projectID": options.project_id} if options.project_id is not None else {}
            params: dict[str, str] = {}
            if options.team_id is not None and len(options.team_id) > 0:
                if options.project_id is None:
                    raise SandboxStartFailed(
                        backend_id="vercel",
                        reason="VercelSandboxClientOptions.project_id is required when team_id is set.",
                    )
                params["teamId"] = options.team_id

            async def _create() -> Any:
                return await http.post("/v1/sandboxes", json=body, params=params)

            try:
                response = await retry_async_call(_create, max_retries=options.max_retries)
            except (
                SandboxStartFailed,
                SandboxConfigurationError,
                ExecTimeoutError,
                ExecTransportError,
            ):
                raise
            except Exception as exc:
                raise SandboxStartFailed(
                    backend_id="vercel",
                    reason=f"Vercel create failed: {exc}",
                ) from exc
            if response.status_code >= 400:
                raise map_http_error_to_sandbox_error(
                    response.status_code,
                    response.text,
                    backend_id="vercel",
                )
            payload = response.json()
            sandbox_id = payload.get("sandboxId") or payload.get("id")
            if not isinstance(sandbox_id, str) or len(sandbox_id) == 0:
                raise SandboxStartFailed(
                    backend_id="vercel",
                    reason=f"Vercel create did not return a sandbox ID; payload={payload!r}",
                )
            return RemoteVMSandboxSession(
                http_client=http,
                session_endpoint=f"/v1/sandboxes/{sandbox_id}",
                backend_id="vercel",
                session_id=sandbox_id,
                max_retries=options.max_retries,
            )
        except BaseException:
            if owns_http:
                await http.aclose()
            raise

    @override
    async def resume(self, state: SandboxSessionState) -> BaseSandboxSession:
        sandbox_id = state.provider_payload.get("sandbox_id")
        if not isinstance(sandbox_id, str) or len(sandbox_id) == 0:
            raise SandboxStartFailed(
                backend_id="vercel",
                reason="Vercel resume requires provider_payload['sandbox_id']",
            )
        api_key = state.provider_payload.get("api_key")
        base_url = state.provider_payload.get("base_url", _DEFAULT_BASE_URL)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if isinstance(api_key, str) and len(api_key) > 0:
            headers["Authorization"] = "Bearer " + api_key
        http = self._injected_http or build_httpx_client(
            base_url=base_url if isinstance(base_url, str) else _DEFAULT_BASE_URL,
            headers=headers,
        )
        return RemoteVMSandboxSession(
            http_client=http,
            session_endpoint=f"/v1/sandboxes/{sandbox_id}",
            backend_id="vercel",
            session_id=sandbox_id,
        )
