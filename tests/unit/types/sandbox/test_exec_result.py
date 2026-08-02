"""Tests for ``troopai.adk.types.sandbox.exec_result``."""

from __future__ import annotations

import dataclasses

import pytest

from troopai.adk.types.sandbox.exec_result import (
    ExecResult,
    ExposedPortEndpoint,
    PtyHandle,
)


class TestExecResult:
    def test_construction(self) -> None:
        r = ExecResult(stdout=b"hello\n", stderr=b"", exit_code=0, duration_ms=42)
        assert r.stdout == b"hello\n"
        assert r.stderr == b""
        assert r.exit_code == 0
        assert r.duration_ms == 42

    def test_ok_property(self) -> None:
        assert ExecResult(stdout=b"", stderr=b"", exit_code=0).ok is True
        assert ExecResult(stdout=b"", stderr=b"", exit_code=1).ok is False
        assert ExecResult(stdout=b"", stderr=b"", exit_code=137).ok is False

    def test_decoded_stdout_utf8(self) -> None:
        r = ExecResult(stdout="hêllo".encode(), stderr=b"", exit_code=0)
        assert r.decoded_stdout() == "hêllo"

    def test_decoded_stderr_strict_by_default_raises_on_bad_bytes(self) -> None:
        # Strict default: malformed bytes surface as UnicodeDecodeError
        # rather than being silently replaced. Sandbox forensic invariant.
        r = ExecResult(stdout=b"", stderr=b"\xff\xfe\xfd", exit_code=1)
        with pytest.raises(UnicodeDecodeError):
            r.decoded_stderr()

    def test_decoded_stderr_replace_is_opt_in(self) -> None:
        r = ExecResult(stdout=b"", stderr=b"\xff\xfe\xfd", exit_code=1)
        decoded = r.decoded_stderr(errors="replace")
        # U+FFFD substitution kicked in; method did not raise.
        assert len(decoded) > 0

    def test_is_frozen(self) -> None:
        r = ExecResult(stdout=b"", stderr=b"", exit_code=0)
        # frozen=True makes assignment a static type error AND a runtime
        # FrozenInstanceError; the `type: ignore[misc]` silences the
        # static side so we can verify the runtime guard fires.
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.exit_code = 1  # type: ignore[misc]


class TestExposedPortEndpoint:
    def test_construction(self) -> None:
        e = ExposedPortEndpoint(host="sandbox-1.example.com", port=8080)
        assert e.host == "sandbox-1.example.com"
        assert e.port == 8080
        assert e.tls is False
        assert e.query == ""

    def test_url_for_http_non_default_port(self) -> None:
        e = ExposedPortEndpoint(host="h", port=8080)
        assert e.url_for("http") == "http://h:8080"

    def test_url_for_http_default_port_omits_port(self) -> None:
        e = ExposedPortEndpoint(host="h", port=80)
        assert e.url_for("http") == "http://h"

    def test_url_for_https_default_port_omits_port(self) -> None:
        e = ExposedPortEndpoint(host="h", port=443)
        assert e.url_for("https") == "https://h"

    def test_url_for_with_query(self) -> None:
        e = ExposedPortEndpoint(host="h", port=8080, query="token=abc")
        assert e.url_for() == "http://h:8080?token=abc"

    def test_url_for_wss_uses_https_default(self) -> None:
        e = ExposedPortEndpoint(host="h", port=443, tls=True)
        assert e.url_for("wss") == "wss://h"

    def test_url_for_empty_scheme_rejected(self) -> None:
        e = ExposedPortEndpoint(host="h", port=80)
        with pytest.raises(ValueError, match="non-empty"):
            e.url_for("")

    def test_url_for_ftp_scheme_rejected(self) -> None:
        # FTP has different authority + port conventions; rendering
        # generically would silently produce a wrong URL.
        e = ExposedPortEndpoint(host="h", port=21)
        with pytest.raises(ValueError, match="unsupported scheme"):
            e.url_for("ftp")

    def test_url_for_redis_scheme_rejected(self) -> None:
        e = ExposedPortEndpoint(host="h", port=6379)
        with pytest.raises(ValueError, match="unsupported scheme"):
            e.url_for("redis")

    def test_empty_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            ExposedPortEndpoint(host="", port=80)

    def test_port_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="1..65535"):
            ExposedPortEndpoint(host="h", port=0)

    def test_port_above_65535_rejected(self) -> None:
        with pytest.raises(ValueError, match="1..65535"):
            ExposedPortEndpoint(host="h", port=70000)


class TestPtyHandle:
    def test_construction(self) -> None:
        h = PtyHandle(
            session_id="abc",
            command="bash -l",
            backend_payload={"docker_exec_id": "deadbeef"},
        )
        assert h.session_id == "abc"
        assert h.command == "bash -l"
        # backend_payload is opaque — we don't introspect it here.

    def test_is_frozen(self) -> None:
        h = PtyHandle(session_id="x", command="y", backend_payload=None)
        # frozen=True: static type error AND runtime FrozenInstanceError.
        # The `type: ignore[misc]` silences static so we can verify
        # the runtime guard.
        with pytest.raises(dataclasses.FrozenInstanceError):
            h.session_id = "z"  # type: ignore[misc]

    def test_repr_redacts_backend_payload(self) -> None:
        # backend_payload may carry provider tokens / exec IDs / socket
        # paths; the repr override redacts so tracebacks + log captures
        # never leak it.
        h = PtyHandle(
            session_id="abc",
            command="bash -l",
            backend_payload={"token": "SUPER_SECRET", "exec_id": "deadbeef"},
        )
        rendered = repr(h)
        assert "SUPER_SECRET" not in rendered
        assert "deadbeef" not in rendered
        assert "<opaque>" in rendered
        assert "abc" in rendered  # session_id is OK to render
        assert "bash -l" in rendered  # command is OK to render
