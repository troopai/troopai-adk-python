"""ModalSandboxClient — hosted bridge.

Requires the [sandbox-modal] extra. Extends
RemoteVMSandboxClient for shared HTTP / retry / port-forward
machinery; overrides only provider-specific create / resume / auth.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.hosted.modal.modal_client import (
    ModalSandboxClient,
    ModalSandboxClientOptions,
)

__all__ = ["ModalSandboxClient", "ModalSandboxClientOptions"]
