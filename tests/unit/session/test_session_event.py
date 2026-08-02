"""Tests for SessionEvent dataclass and factory."""

import pytest

from troopai.adk.session.session_event import create_session_event


class TestCreateSessionEvent:
    def test_generates_id(self):
        event = create_session_event(author="user", content={"role": "user", "content": "hi"})
        assert event.id  # non-empty UUID string
        assert len(event.id) == 36  # UUID format

    def test_generates_timestamp(self):
        event = create_session_event(author="user", content={"role": "user", "content": "hi"})
        assert event.timestamp > 0

    def test_unique_ids(self):
        e1 = create_session_event(author="user", content={"role": "user", "content": "a"})
        e2 = create_session_event(author="user", content={"role": "user", "content": "b"})
        assert e1.id != e2.id

    def test_author_preserved(self):
        event = create_session_event(author="my-agent", content={"role": "assistant", "content": "ok"})
        assert event.author == "my-agent"

    def test_content_preserved(self):
        content = {"role": "user", "content": "hello"}
        event = create_session_event(author="user", content=content)
        assert event.content == content

    def test_default_empty_state_delta(self):
        event = create_session_event(author="user", content={"role": "user", "content": "hi"})
        assert event.state_delta == {}

    def test_with_state_delta(self):
        delta = {"theme": "dark"}
        event = create_session_event(
            author="user",
            content={"role": "user", "content": "hi"},
            state_delta=delta,
        )
        assert event.state_delta == {"theme": "dark"}


class TestSessionEvent:
    def test_frozen(self):
        event = create_session_event(author="user", content={"role": "user", "content": "hi"})
        with pytest.raises(AttributeError):
            event.author = "other"  # type: ignore[misc]

    def test_to_content(self):
        content = {"role": "user", "content": "hello"}
        event = create_session_event(author="user", content=content)
        assert event.to_content() is content
