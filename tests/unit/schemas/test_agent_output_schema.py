"""Tests for AgentOutputSchema PEP 604 union handling.

PEP 604 unions (``A | B``) report ``types.UnionType`` from
``get_origin``, while ``typing.Union[A, B]`` reports ``typing.Union``.
``name()`` and the discriminator-repair path must treat both spellings
identically: the API-facing schema name must stay sanitized, and the
Gemini-style discriminator repair must still run.
"""

from __future__ import annotations

import re
from typing import Literal, Union

import pytest
from pydantic import BaseModel, Field

from troopai.adk.schemas.agent_output_schema import AgentOutputSchema


class DiscriminatedA(BaseModel):
    kind: Literal["a"] = "a"
    value: str = Field(description="The value for A.")


class DiscriminatedB(BaseModel):
    kind: Literal["b"] = "b"
    score: int = Field(description="The score for B.")


# OpenAI requires the json_schema name to match this pattern (<=64 chars).
_API_NAME_RE = re.compile(r"[a-zA-Z0-9_-]+")


class TestPep604UnionName:
    """name() must produce an API-safe sanitized name for ``A | B``."""

    def test_pep604_union_name_is_sanitized(self) -> None:
        """``A | B`` yields the same sanitized name as typing.Union."""
        schema = AgentOutputSchema(DiscriminatedA | DiscriminatedB)
        assert schema.name() == "DiscriminatedA_DiscriminatedB"

    def test_pep604_union_name_matches_api_pattern(self) -> None:
        """The name has no spaces, dots, or pipes (rejected by OpenAI)."""
        name = AgentOutputSchema(DiscriminatedA | DiscriminatedB).name()
        assert _API_NAME_RE.fullmatch(name) is not None
        assert " " not in name
        assert "." not in name
        assert "|" not in name

    def test_pep604_and_typing_union_names_agree(self) -> None:
        """Both union spellings produce an identical name()."""
        pep604 = AgentOutputSchema(DiscriminatedA | DiscriminatedB)
        typing_union = AgentOutputSchema(Union[DiscriminatedA, DiscriminatedB])
        assert pep604.name() == typing_union.name()


class TestPep604UnionDiscriminatorRepair:
    """Gemini-style discriminator repair must run for ``A | B`` too."""

    def test_pep604_repair_capitalized_discriminator(self) -> None:
        """'A' → 'a' for a PEP 604 union, same as typing.Union."""
        schema = AgentOutputSchema(DiscriminatedA | DiscriminatedB)
        result = schema.validate_json('{"response": {"kind": "A", "value": "test"}}')
        assert result.kind == "a"
        assert result.value == "test"

    def test_pep604_repair_uppercase_discriminator(self) -> None:
        """'B' → 'b' for a PEP 604 union."""
        schema = AgentOutputSchema(DiscriminatedA | DiscriminatedB)
        result = schema.validate_json('{"response": {"kind": "B", "score": 42}}')
        assert result.kind == "b"
        assert result.score == 42

    def test_pep604_correct_casing_passes_through(self) -> None:
        """Correct casing validates without repair."""
        schema = AgentOutputSchema(DiscriminatedA | DiscriminatedB)
        result = schema.validate_json('{"response": {"kind": "a", "value": "ok"}}')
        assert result.kind == "a"

    def test_pep604_unrepairable_value_raises(self) -> None:
        """A completely wrong discriminator still raises ValueError."""
        schema = AgentOutputSchema(DiscriminatedA | DiscriminatedB)
        with pytest.raises(ValueError, match="does not match"):
            schema.validate_json('{"response": {"kind": "unknown", "value": "x"}}')
