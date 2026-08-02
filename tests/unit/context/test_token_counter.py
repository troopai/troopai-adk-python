"""Tests for TokenCounter."""

from unittest.mock import patch

from troopai.adk.context.token_counter import TokenCounter


@patch("litellm.token_counter", return_value=42)
def test_count_messages(mock_counter):
    messages = [{"role": "user", "content": "Hello"}]
    result = TokenCounter.count_messages(messages, "gpt-4o")

    assert result == 42
    mock_counter.assert_called_once_with(model="gpt-4o", messages=messages)


@patch("litellm.token_counter", return_value=10)
def test_count_text(mock_counter):
    result = TokenCounter.count_text("Hello world", "gpt-4o")

    assert result == 10
    mock_counter.assert_called_once_with(model="gpt-4o", text="Hello world")


@patch("litellm.token_counter", return_value=150)
def test_estimate_remaining(_mock_counter):
    messages = [{"role": "user", "content": "Hello"}]
    remaining = TokenCounter.estimate_remaining(messages, "gpt-4o", max_context=200)

    assert remaining == 50


@patch("litellm.token_counter", return_value=250)
def test_estimate_remaining_negative(_mock_counter):
    messages = [{"role": "user", "content": "Very long..."}]
    remaining = TokenCounter.estimate_remaining(messages, "gpt-4o", max_context=200)

    assert remaining == -50
