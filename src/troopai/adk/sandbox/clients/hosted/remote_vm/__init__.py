"""RemoteVMSandboxClient — intermediate base for hosted bridges.

Owns httpx client construction, retry policy (tenacity), transport-
error mapping to typed SandboxError subclasses, ExecResult
normalization from provider JSON, and port-forwarding abstraction.
Hosted bridges (E2B, Vercel, Modal, ...) override only ``create``,
``delete``, ``resume``, auth headers, and proprietary command transport.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_client import (
    RemoteVMSandboxClient,
    RemoteVMSandboxClientOptions,
    build_httpx_client,
    map_http_error_to_sandbox_error,
    parse_exec_result,
    retry_async_call,
)
from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_session import (
    RemoteVMSandboxSession,
)

__all__ = [
    "RemoteVMSandboxClient",
    "RemoteVMSandboxClientOptions",
    "RemoteVMSandboxSession",
    "build_httpx_client",
    "map_http_error_to_sandbox_error",
    "parse_exec_result",
    "retry_async_call",
]
