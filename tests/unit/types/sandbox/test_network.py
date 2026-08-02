"""Tests for ``troopai.adk.types.sandbox.network``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.network import NetworkPolicy, PortForwardRule


class TestNetworkPolicyDefaults:
    def test_default_is_deny_with_dns(self) -> None:
        p = NetworkPolicy()
        assert p.deny_default is True
        assert p.allow_dns is True
        assert p.allow_hosts == ()
        assert p.allow_ports == ()
        assert p.port_forwards == ()

    def test_open_factory(self) -> None:
        p = NetworkPolicy.open()
        assert p.deny_default is False

    def test_closed_factory(self) -> None:
        p = NetworkPolicy.closed()
        assert p.deny_default is True
        assert p.allow_dns is False


class TestNetworkPolicyValidation:
    def test_empty_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            NetworkPolicy(allow_hosts=("",))

    def test_port_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="1..65535"):
            NetworkPolicy(allow_ports=(0,))

    def test_high_port_rejected(self) -> None:
        with pytest.raises(ValueError, match="1..65535"):
            NetworkPolicy(allow_ports=(70000,))


class TestPortForwardRule:
    def test_construction(self) -> None:
        r = PortForwardRule(local_port=8080)
        assert r.local_port == 8080
        assert r.remote_port is None
        assert r.protocol == "tcp"

    def test_local_port_validation(self) -> None:
        with pytest.raises(ValueError, match="local_port"):
            PortForwardRule(local_port=0)

    def test_remote_port_validation(self) -> None:
        with pytest.raises(ValueError, match="remote_port"):
            PortForwardRule(local_port=8080, remote_port=70000)

    def test_udp_protocol(self) -> None:
        r = PortForwardRule(local_port=53, protocol="udp")
        assert r.protocol == "udp"
