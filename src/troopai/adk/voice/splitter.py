"""Incremental text splitting for streamed speech synthesis.

A *text splitter* drives how a stream of partial text deltas is cut into
synthesis-ready segments. As text accumulates, the splitter is called
with the full pending buffer and returns ``(ready_to_speak, remainder)``:
the leading portion safe to send to the text-to-speech model now, and
the trailing portion to keep buffering. Speaking on sentence boundaries
(rather than word-by-word) gives the synthesizer enough context for
natural prosody while still starting playback before the full response
is generated.
"""

from __future__ import annotations

import re
from collections.abc import Callable

TextSplitter = Callable[[str], tuple[str, str]]
"""Maps a pending text buffer to ``(ready_to_speak, remainder)``."""

DEFAULT_MIN_SENTENCE_LENGTH = 20
"""Minimum characters before a completed sentence is released for synthesis."""

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
"""Split after sentence-ending punctuation followed by whitespace."""


def sentence_splitter(min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH) -> TextSplitter:
    """Build a splitter that releases complete sentences once long enough.

    The returned callable splits the buffer on sentence boundaries. When
    at least one boundary is present, every sentence except the final
    (possibly still-incomplete) one is released — but only if the
    released text reaches ``min_sentence_length``, which avoids handing
    the synthesizer clipped fragments like ``"Hi."``.

    Args:
        min_sentence_length: Minimum length of the released text before
            it is spoken. Must be positive.

    Returns:
        A :data:`TextSplitter` callable.

    Raises:
        ValueError: When ``min_sentence_length`` is not positive.
    """
    if min_sentence_length <= 0:
        raise ValueError(f"min_sentence_length must be positive, got {min_sentence_length}")

    def split(buffer: str) -> tuple[str, str]:
        sentences = _SENTENCE_BOUNDARY.split(buffer)
        if len(sentences) > 1:
            ready = " ".join(sentences[:-1])
            if len(ready) >= min_sentence_length:
                return ready, sentences[-1]
        return "", buffer

    return split
