"""Tests for LLMUsage token accounting.

Regression coverage for:
- Dead BeforeValidator on plain @dataclass (fields must accept None at
  construction time; __add__ must handle None safely).
- __add__ correctly accumulates cached_tokens, cache_creation_input_tokens,
  and reasoning_tokens across turns.
"""

from __future__ import annotations

from troopai.adk.types.tokens.llm_usage import LLMSingleRequestUsage, LLMUsage
from troopai.adk.types.tokens.tokens import InputTokensDetails, OutputTokensDetails


class TestLLMUsageDefaults:
    def test_default_construction(self) -> None:
        usage = LLMUsage()
        assert usage.requests == 0
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert isinstance(usage.input_tokens_details, InputTokensDetails)
        assert isinstance(usage.output_tokens_details, OutputTokensDetails)

    def test_explicit_none_fields_accepted(self) -> None:
        """Fields declared Optional must accept None explicitly.

        Regression: previously the type annotation was non-Optional (decorated
        with Annotated[..., BeforeValidator(...)]) but the __add__ had None
        guards, indicating None was actually possible. The type is now
        explicitly Optional so this construction must not raise.
        """
        usage = LLMUsage(input_tokens_details=None, output_tokens_details=None)
        assert usage.input_tokens_details is None
        assert usage.output_tokens_details is None


class TestLLMUsageAdd:
    def _make(
        self,
        *,
        requests: int = 1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int = 0,
        cache_creation: int = 0,
        reasoning_tokens: int = 0,
    ) -> LLMUsage:
        return LLMUsage(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=cached_tokens,
                cache_creation_input_tokens=cache_creation,
            ),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
        )

    def test_add_accumulates_tokens(self) -> None:
        a = self._make(requests=1, input_tokens=100, output_tokens=50, total_tokens=150)
        b = self._make(requests=1, input_tokens=200, output_tokens=75, total_tokens=275)
        result = a + b
        assert result.requests == 2
        assert result.input_tokens == 300
        assert result.output_tokens == 125
        assert result.total_tokens == 425

    def test_add_accumulates_cached_tokens(self) -> None:
        a = self._make(requests=1, total_tokens=100, cached_tokens=10, cache_creation=5)
        b = self._make(requests=1, total_tokens=200, cached_tokens=20, cache_creation=15)
        result = a + b
        assert result.input_tokens_details is not None
        assert result.input_tokens_details.cached_tokens == 30
        assert result.input_tokens_details.cache_creation_input_tokens == 20

    def test_add_accumulates_reasoning_tokens(self) -> None:
        a = self._make(requests=1, total_tokens=100, reasoning_tokens=30)
        b = self._make(requests=1, total_tokens=200, reasoning_tokens=70)
        result = a + b
        assert result.output_tokens_details is not None
        assert result.output_tokens_details.reasoning_tokens == 100

    def test_add_with_none_input_details(self) -> None:
        """__add__ must handle None input_tokens_details gracefully."""
        a = LLMUsage(requests=1, total_tokens=100, input_tokens_details=None, output_tokens_details=None)
        b = LLMUsage(requests=1, total_tokens=200, input_tokens_details=None, output_tokens_details=None)
        result = a + b
        assert result.requests == 2
        assert result.total_tokens == 300
        assert result.input_tokens_details is not None
        assert result.input_tokens_details.cached_tokens == 0
        assert result.output_tokens_details is not None
        assert result.output_tokens_details.reasoning_tokens == 0

    def test_add_single_request_preserved_in_usage_list(self) -> None:
        """A single-request LLMUsage appends to usage list when added."""
        base = LLMUsage()
        single = self._make(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        result = base + single
        assert len(result.usage) == 1
        entry = result.usage[0]
        assert isinstance(entry, LLMSingleRequestUsage)
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.total_tokens == 150

    def test_add_multi_request_extends_usage_list(self) -> None:
        """Multi-request LLMUsage extends the usage list (not append)."""
        existing_entry = LLMSingleRequestUsage(
            input_tokens=50,
            output_tokens=25,
            total_tokens=75,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_creation_input_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        )
        multi = LLMUsage(
            requests=2,
            total_tokens=200,
            usage=[existing_entry],
        )
        base = LLMUsage()
        result = base + multi
        # Multi-request extends; usage list from multi is merged
        assert len(result.usage) == 1
        assert result.usage[0].total_tokens == 75

    def test_add_identity(self) -> None:
        """Adding a zero-usage LLMUsage is a no-op for counters."""
        a = self._make(requests=2, input_tokens=300, output_tokens=150, total_tokens=450)
        zero = LLMUsage()
        result = a + zero
        assert result.requests == 2
        assert result.input_tokens == 300
        assert result.output_tokens == 150
        assert result.total_tokens == 450

    def test_add_does_not_mutate_original_usage_list(self) -> None:
        """__add__ must not mutate the original caller's usage list (shallow-copy bug).

        Regression: ``copy(self)`` shared the ``usage`` list between ``self``
        and ``new_usage``. Every subsequent ``new_usage.usage.append(...)``
        therefore mutated ``self.usage`` too.
        """
        u1 = self._make(requests=1, input_tokens=100, output_tokens=50, total_tokens=150)
        u2 = self._make(requests=1, input_tokens=200, output_tokens=75, total_tokens=275)
        result = u1 + u2
        # The result has one single-request entry (from u2).
        assert len(result.usage) == 1
        # u1's own list must NOT have been mutated.
        assert len(u1.usage) == 0, f"u1.usage was mutated by __add__: {u1.usage!r}"
        # The result list must be a separate object.
        assert result.usage is not u1.usage

    def test_add_chain_does_not_leak_into_earlier_snapshots(self) -> None:
        """Multi-turn accumulation must not corrupt intermediate snapshots."""
        base = LLMUsage()
        turn1 = self._make(requests=1, input_tokens=100, output_tokens=50, total_tokens=150)
        turn2 = self._make(requests=1, input_tokens=200, output_tokens=100, total_tokens=300)

        after_turn1 = base + turn1
        snapshot_len = len(after_turn1.usage)  # should be 1

        # Simulate a second accumulation pass.
        _ = after_turn1 + turn2

        # The snapshot taken after turn 1 must still report exactly 1 entry.
        assert len(after_turn1.usage) == snapshot_len, (
            f"after_turn1.usage was corrupted by the turn-2 addition: {after_turn1.usage!r}"
        )
