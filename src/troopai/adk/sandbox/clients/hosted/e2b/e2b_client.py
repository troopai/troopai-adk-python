"""E2bSandboxClient — hosted bridge backed by the E2B sandbox provider.

E2B exposes a REST-ish API plus its own ``e2b`` Python SDK. This client
takes the REST path so it doesn't pull in the SDK as a required dep —
the [sandbox-e2b] extra installs httpx only. Users who want the typed
SDK surface can subclass and override ``create`` / ``resume``.
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
    from troopai.adk.types.sandbox.cost import SandboxBillingRecord
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

logger = logging.getLogger(__name__)

__all__ = ["E2bSandboxClient", "E2bSandboxClientOptions"]

_DEFAULT_E2B_BASE_URL = "https://api.e2b.dev"


class E2bSandboxClientOptions(RemoteVMSandboxClientOptions):
    """Options for the E2B hosted bridge."""

    api_key: str | None = None
    """E2B API key (X-API-KEY header)."""

    template_id: str = "base"
    """E2B sandbox template ID. Defaults to E2B's 'base' template."""

    region: str | None = None
    """Optional region preference; provider may ignore."""


class E2bSandboxClient(RemoteVMSandboxClient):
    """Hosted-bridge client backed by E2B."""

    backend_id = "e2b"
    # Approximate compute rate (USD/min); override per current E2B pricing:
    # https://e2b.dev/pricing
    cost = SandboxCostDescriptor(usd_per_minute=0.06)
    capabilities = SandboxBackendCapabilities(network=True, persistent=True)

    def __init__(self, *, http_client: Any = None) -> None:
        """Construct the client.

        ``http_client`` injects a pre-built httpx.AsyncClient for tests;
        when ``None`` each ``create`` call builds its own.
        """
        super().__init__()
        self._injected_http = http_client

    def _auth_headers(self, options: E2bSandboxClientOptions) -> dict[str, str]:
        if options.api_key is None or len(options.api_key) == 0:
            raise SandboxStartFailed(
                backend_id="e2b",
                reason="E2bSandboxClientOptions.api_key is required at create-time.",
            )
        return {
            "X-API-KEY": options.api_key,
            "Content-Type": "application/json",
        }

    @override
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: E2bSandboxClientOptions,  # type: ignore[override]
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
            base_url=options.base_url or _DEFAULT_E2B_BASE_URL,
            headers=headers,
            timeout=options.request_timeout,
        )
        try:
            body: dict[str, Any] = {"templateID": options.template_id}
            if options.region is not None:
                body["region"] = options.region

            async def _create() -> Any:
                return await http.post("/sandboxes", json=body)

            try:
                response = await retry_async_call(
                    _create,
                    max_retries=options.max_retries,
                )
            except SandboxStartFailed:
                raise
            except (
                SandboxConfigurationError,
                ExecTimeoutError,
                ExecTransportError,
            ):
                # Typed SandboxError subclasses already carry context; let them bubble.
                raise
            except Exception as exc:
                raise SandboxStartFailed(
                    backend_id="e2b",
                    reason=f"E2B create failed: {exc}",
                ) from exc
            if response.status_code >= 400:
                raise map_http_error_to_sandbox_error(
                    response.status_code,
                    response.text,
                    backend_id="e2b",
                )
            payload = response.json()
            sandbox_id = payload.get("sandboxID") or payload.get("id")
            if not isinstance(sandbox_id, str) or len(sandbox_id) == 0:
                raise SandboxStartFailed(
                    backend_id="e2b",
                    reason=f"E2B create did not return a sandbox ID; payload={payload!r}",
                )
            session_endpoint = f"/sandboxes/{sandbox_id}"
            return RemoteVMSandboxSession(
                http_client=http,
                session_endpoint=session_endpoint,
                backend_id="e2b",
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
                backend_id="e2b",
                reason="E2B resume requires provider_payload['sandbox_id']",
            )
        api_key = state.provider_payload.get("api_key")
        base_url = state.provider_payload.get("base_url", _DEFAULT_E2B_BASE_URL)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if isinstance(api_key, str) and len(api_key) > 0:
            headers["X-API-KEY"] = api_key
        http = self._injected_http or build_httpx_client(
            base_url=base_url if isinstance(base_url, str) else _DEFAULT_E2B_BASE_URL,
            headers=headers,
        )
        return RemoteVMSandboxSession(
            http_client=http,
            session_endpoint=f"/sandboxes/{sandbox_id}",
            backend_id="e2b",
            session_id=sandbox_id,
        )

    @override
    async def fetch_billing(self, session: BaseSandboxSession) -> SandboxBillingRecord | None:
        """Live billing for an E2B sandbox — returns ``None`` by design.

        E2B meters compute usage at the account level; it does not expose a
        per-sandbox, provider-reported dollar-cost endpoint. Live billing is
        therefore unavailable here, and the framework's
        ``SandboxUsage.computed_cost_usd`` (the static rate card times wall-
        clock duration) is the per-run cost estimate. This override is the
        seam where a per-sandbox cost endpoint would be wired if E2B adds
        one. E2B pricing: https://e2b.dev/pricing
        """
        del session
        return None
