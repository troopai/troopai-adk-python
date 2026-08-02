"""Tests for ``A2AContinuationToken`` and ``A2ATaskStatus``."""

import dataclasses
import json

import pytest

from troopai.adk.a2a import A2AContinuationToken, A2ATaskStatus


class TestA2AContinuationToken:
    def test_construction_requires_all_fields(self) -> None:
        tok = A2AContinuationToken(
            task_id="t1",
            context_id="c1",
            remote_url="http://example.com",
        )
        assert tok.task_id == "t1"
        assert tok.context_id == "c1"
        assert tok.remote_url == "http://example.com"

    def test_is_frozen(self) -> None:
        tok = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tok.task_id = "t2"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        b = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        c = A2AContinuationToken(task_id="t2", context_id="c1", remote_url="http://x")
        assert a == b
        assert a != c

    def test_hashable_when_frozen(self) -> None:
        tok = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        # Frozen dataclasses are hashable.
        s = {tok}
        assert tok in s

    def test_json_round_trip(self) -> None:
        # Continuation tokens MUST survive JSON serialisation so they
        # can be persisted to a queue / database / cache and resumed
        # from a fresh process. This is the load-bearing invariant.
        tok = A2AContinuationToken(
            task_id="task-abc",
            context_id="ctx-xyz",
            remote_url="https://agents.example.com:8443",
        )
        encoded = json.dumps(dataclasses.asdict(tok))
        restored = A2AContinuationToken(**json.loads(encoded))
        assert restored == tok

    @pytest.mark.parametrize("field", ["task_id", "context_id", "remote_url"])
    def test_empty_identifier_rejected(self, field: str) -> None:
        # An empty identifier produces deferred, confusing failures downstream;
        # reject it at construction (the single deserialization gate).
        kwargs = {"task_id": "t1", "context_id": "c1", "remote_url": "https://x"}
        kwargs[field] = ""
        with pytest.raises(ValueError, match=f"{field} must be non-empty"):
            A2AContinuationToken(**kwargs)

    @pytest.mark.parametrize("bad_url", ["file:///etc/passwd", "gopher://x", "ftp://x", "x"])
    def test_non_http_scheme_rejected(self, bad_url: str) -> None:
        # remote_url is carried in a serializable token and may arrive from an
        # untrusted source — reject non-HTTP(S) schemes as a baseline SSRF guard.
        with pytest.raises(ValueError, match="http:// or https:// scheme"):
            A2AContinuationToken(task_id="t1", context_id="c1", remote_url=bad_url)

    def test_https_and_http_schemes_accepted(self) -> None:
        for url in ("http://x", "https://agents.example.com:8443/path"):
            tok = A2AContinuationToken(task_id="t1", context_id="c1", remote_url=url)
            assert tok.remote_url == url

    def test_untrusted_json_with_bad_scheme_rejected_on_deserialize(self) -> None:
        # The deserialization path (A2AContinuationToken(**data)) runs __post_init__,
        # so a tampered persisted token cannot smuggle a file:// URL through.
        tampered = {"task_id": "t1", "context_id": "c1", "remote_url": "file:///secret"}
        with pytest.raises(ValueError, match="http:// or https:// scheme"):
            A2AContinuationToken(**tampered)


class TestA2ATaskStatus:
    def test_completed_carries_result(self) -> None:
        status = A2ATaskStatus(
            task_id="t1",
            context_id="c1",
            state="completed",
            result="The answer is 42.",
        )
        assert status.state == "completed"
        assert status.result == "The answer is 42."
        assert status.message is None

    def test_failed_carries_message_no_result(self) -> None:
        status = A2ATaskStatus(
            task_id="t1",
            context_id="c1",
            state="failed",
            message="Internal server error",
        )
        assert status.state == "failed"
        assert status.result is None
        assert status.message == "Internal server error"

    def test_is_frozen(self) -> None:
        status = A2ATaskStatus(task_id="t1", context_id="c1", state="working")
        with pytest.raises(dataclasses.FrozenInstanceError):
            status.state = "completed"  # type: ignore[misc]
