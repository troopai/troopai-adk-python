"""Tests for the sentence-based text splitter."""

from __future__ import annotations

import pytest

from troopai.adk.voice.splitter import sentence_splitter


def test_releases_completed_sentence_once_long_enough():
    split = sentence_splitter(min_sentence_length=10)
    ready, remainder = split("This is a sentence. And more")
    assert ready == "This is a sentence."
    assert remainder == "And more"


def test_holds_back_text_shorter_than_minimum():
    split = sentence_splitter(min_sentence_length=20)
    ready, remainder = split("Hi. Bye")
    assert ready == ""
    assert remainder == "Hi. Bye"


def test_keeps_buffering_without_a_boundary():
    split = sentence_splitter()
    ready, remainder = split("no terminator yet")
    assert ready == ""
    assert remainder == "no terminator yet"


def test_releases_all_but_the_trailing_sentence():
    split = sentence_splitter(min_sentence_length=5)
    ready, remainder = split("One sentence here. Two sentence here. Trailing")
    assert ready == "One sentence here. Two sentence here."
    assert remainder == "Trailing"


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_rejects_non_positive_minimum(bad: int):
    with pytest.raises(ValueError, match="must be positive"):
        sentence_splitter(min_sentence_length=bad)
