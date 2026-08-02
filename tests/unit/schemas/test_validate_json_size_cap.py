"""Tests for ``AgentOutputSchema.validate_json`` 256 KiB size cap.

The cap is a defence-in-depth bound against pathological LLM output
that would stackoverflow or OOM the validator. It runs **before**
``json.loads`` so the attacker pays the O(1) byte-count cost, not the
O(N) parse cost.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.schemas.agent_output_schema import (
    MAX_SCHEMA_BYTES,
    AgentOutputSchema,
)


class _Payload(BaseModel):
    """Wraps a single string so oversized strings hit the cap cleanly."""

    content: str


class TestValidateJsonSizeCap:
    def test_payload_just_under_cap_validates(self) -> None:
        # 256 KiB - 64 bytes leaves headroom for the JSON envelope
        # (``{"content": "..."}`` plus quoting), so the raw string
        # of ``MAX_SCHEMA_BYTES - 64`` ASCII chars yields a payload
        # just under the cap. ASCII guarantees 1 byte per codepoint.
        filler = "x" * (MAX_SCHEMA_BYTES - 64)
        json_str = f'{{"content": "{filler}"}}'

        # Sanity: we built a payload under the cap
        assert len(json_str.encode("utf-8")) <= MAX_SCHEMA_BYTES

        schema = AgentOutputSchema(_Payload)
        result = schema.validate_json(json_str)
        assert result.content == filler

    def test_payload_over_cap_rejected(self) -> None:
        # 2 KiB of headroom past the cap ensures the encoded payload
        # clears the boundary regardless of envelope overhead.
        filler = "x" * (MAX_SCHEMA_BYTES + 2048)
        json_str = f'{{"content": "{filler}"}}'

        # Sanity: we built a payload over the cap
        assert len(json_str.encode("utf-8")) > MAX_SCHEMA_BYTES

        schema = AgentOutputSchema(_Payload)
        with pytest.raises(ValueError, match="exceeds validate_json size cap"):
            schema.validate_json(json_str)

    def test_multibyte_utf8_counted_in_bytes_not_codepoints(self) -> None:
        # 4-byte emoji inflates byte-count ~4× past codepoint count.
        # A string of (MAX_SCHEMA_BYTES // 4 + 1) emojis is
        # comfortably over the cap in bytes but well under any naive
        # codepoint-based check — proves the guard measures UTF-8 bytes.
        emoji = "\U0001f4a9"  # 4 UTF-8 bytes
        codepoint_count = (MAX_SCHEMA_BYTES // 4) + 1024
        filler = emoji * codepoint_count
        json_str = f'{{"content": "{filler}"}}'

        encoded_bytes = len(json_str.encode("utf-8"))
        assert encoded_bytes > MAX_SCHEMA_BYTES
        assert len(json_str) < MAX_SCHEMA_BYTES  # codepoint count is UNDER the cap

        schema = AgentOutputSchema(_Payload)
        with pytest.raises(ValueError, match="exceeds validate_json size cap"):
            schema.validate_json(json_str)

    def test_empty_payload_not_blocked(self) -> None:
        schema = AgentOutputSchema(_Payload)
        result = schema.validate_json('{"content": ""}')
        assert result.content == ""
