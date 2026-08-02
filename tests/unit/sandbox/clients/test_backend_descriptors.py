import pytest

from troopai.adk.sandbox.clients.docker.docker_client import DockerSandboxClient
from troopai.adk.sandbox.clients.hosted.blaxel.blaxel_client import BlaxelSandboxClient
from troopai.adk.sandbox.clients.hosted.cloudflare.cloudflare_client import CloudflareSandboxClient
from troopai.adk.sandbox.clients.hosted.daytona.daytona_client import DaytonaSandboxClient
from troopai.adk.sandbox.clients.hosted.e2b.e2b_client import E2bSandboxClient
from troopai.adk.sandbox.clients.hosted.modal.modal_client import ModalSandboxClient
from troopai.adk.sandbox.clients.hosted.runloop.runloop_client import RunloopSandboxClient
from troopai.adk.sandbox.clients.hosted.vercel.vercel_client import VercelSandboxClient
from troopai.adk.sandbox.clients.k8s.k8s_client import K8sPodSandboxClient
from troopai.adk.sandbox.clients.local.subprocess_client import LocalSubprocessSandboxClient
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor

_FREE_BACKENDS = [LocalSubprocessSandboxClient, DockerSandboxClient, K8sPodSandboxClient]
_PRICED_BACKENDS = [
    E2bSandboxClient,
    ModalSandboxClient,
    DaytonaSandboxClient,
    VercelSandboxClient,
    CloudflareSandboxClient,
    BlaxelSandboxClient,
    RunloopSandboxClient,
]
_PERSISTENT_BACKENDS = [
    DockerSandboxClient,
    K8sPodSandboxClient,
    E2bSandboxClient,
    ModalSandboxClient,
    DaytonaSandboxClient,
    BlaxelSandboxClient,
    RunloopSandboxClient,
]
_EPHEMERAL_BACKENDS = [
    LocalSubprocessSandboxClient,
    VercelSandboxClient,
    CloudflareSandboxClient,
]


@pytest.mark.parametrize("client_cls", _FREE_BACKENDS + _PRICED_BACKENDS)
def test_every_backend_declares_cost_and_network(client_cls):
    assert isinstance(client_cls.cost, SandboxCostDescriptor)
    assert isinstance(client_cls.capabilities, SandboxBackendCapabilities)
    assert client_cls.capabilities.network is True


@pytest.mark.parametrize("client_cls", _FREE_BACKENDS)
def test_self_hosted_backends_are_free(client_cls):
    assert client_cls.cost.free is True


@pytest.mark.parametrize("client_cls", _PRICED_BACKENDS)
def test_hosted_backends_are_priced(client_cls):
    assert client_cls.cost.free is False
    assert client_cls.cost.usd_per_minute > 0


@pytest.mark.parametrize("client_cls", _PERSISTENT_BACKENDS)
def test_persistent_backends_declare_persistence(client_cls):
    assert client_cls.capabilities.persistent is True


@pytest.mark.parametrize("client_cls", _EPHEMERAL_BACKENDS)
def test_ephemeral_backends_declare_no_persistence(client_cls):
    assert client_cls.capabilities.persistent is False
