"""Tests for ``CompactionCapability`` and its policies (P12)."""

from __future__ import annotations

import pytest

from troopai.adk.sandbox.capabilities.compaction import (
    CompactionCapability,
    CompactionModelInfo,
    DynamicCompactionPolicy,
    StaticCompactionPolicy,
)


class TestCompactionModelInfo:
    @pytest.mark.parametrize(
        "model,expected_window",
        [
            ("claude-haiku-4-5", 200_000),
            ("claude-sonnet-4-6", 200_000),
            ("gpt-4o", 128_000),
            ("gpt-4o-mini", 128_000),
            ("gpt-5", 400_000),
            ("gpt-4.1", 1_047_576),
            ("o1", 200_000),
            ("o3-mini", 200_000),
            ("gemini-1.5-pro", 1_000_000),
            ("openai/gpt-4o", 128_000),  # provider prefix stripped
            ("anthropic/claude-sonnet-4-6", 200_000),
        ],
    )
    def test_known_models(self, model: str, expected_window: int) -> None:
        info = CompactionModelInfo.maybe_for_model(model)
        assert info is not None
        assert info.context_window == expected_window

    def test_unknown_model_returns_none(self) -> None:
        assert CompactionModelInfo.maybe_for_model("totally-fake-model") is None

    def test_for_model_raises_on_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown context window"):
            CompactionModelInfo.for_model("totally-fake-model")


class TestStaticCompactionPolicy:
    def test_default_threshold(self) -> None:
        p = StaticCompactionPolicy()
        assert p.threshold == 240_000
        assert p.compaction_threshold({"model": "anything"}) == 240_000

    def test_custom_threshold(self) -> None:
        p = StaticCompactionPolicy(threshold=100_000)
        assert p.compaction_threshold({}) == 100_000


class TestDynamicCompactionPolicy:
    def test_default_threshold_is_90_percent(self) -> None:
        p = DynamicCompactionPolicy(
            model_info=CompactionModelInfo(context_window=100_000),
        )
        assert p.threshold == 0.9
        assert p.compaction_threshold({}) == 90_000

    def test_custom_threshold(self) -> None:
        p = DynamicCompactionPolicy(
            model_info=CompactionModelInfo(context_window=200_000),
            threshold=0.5,
        )
        assert p.compaction_threshold({}) == 100_000

    def test_threshold_out_of_range_rejected(self) -> None:
        info = CompactionModelInfo(context_window=100_000)
        with pytest.raises(ValueError):
            DynamicCompactionPolicy(model_info=info, threshold=1.5)
        with pytest.raises(ValueError):
            DynamicCompactionPolicy(model_info=info, threshold=-0.1)


class TestCompactionCapabilityType:
    def test_discriminator(self) -> None:
        c = CompactionCapability()
        assert c.type == "compaction"

    def test_no_policy_by_default(self) -> None:
        c = CompactionCapability()
        assert c.policy is None


class TestSamplingParamsAutoSelect:
    def test_known_model_picks_dynamic(self) -> None:
        c = CompactionCapability()
        params = c.sampling_params({"model": "gpt-4o"})
        management = params["context_management"]
        assert management[0]["type"] == "compaction"
        # 128_000 * 0.9 = 115_200
        assert management[0]["compact_threshold"] == 115_200

    def test_unknown_model_falls_back_to_static(self) -> None:
        c = CompactionCapability()
        params = c.sampling_params({"model": "fake-model"})
        assert params["context_management"][0]["compact_threshold"] == 240_000

    def test_missing_model_falls_back_to_static(self) -> None:
        c = CompactionCapability()
        params = c.sampling_params({})
        assert params["context_management"][0]["compact_threshold"] == 240_000

    def test_explicit_policy_overrides_auto(self) -> None:
        c = CompactionCapability(policy=StaticCompactionPolicy(threshold=50_000))
        params = c.sampling_params({"model": "gpt-4o"})
        # 50_000 (static), not 115_200 (dynamic for gpt-4o)
        assert params["context_management"][0]["compact_threshold"] == 50_000


class TestProcessContextTruncation:
    def test_no_marker_returns_unchanged(self) -> None:
        c = CompactionCapability()
        ctx = [
            {"type": "user", "content": "hi"},
            {"type": "assistant", "content": "hello"},
        ]
        assert c.process_context(ctx) == ctx

    def test_marker_truncates_to_suffix(self) -> None:
        c = CompactionCapability()
        ctx = [
            {"type": "user", "content": "first"},
            {"type": "assistant", "content": "response"},
            {"type": "compaction", "summary": "..."},
            {"type": "user", "content": "second"},
        ]
        result = c.process_context(ctx)
        assert len(result) == 2
        assert result[0]["type"] == "compaction"
        assert result[1]["content"] == "second"

    def test_latest_marker_wins(self) -> None:
        c = CompactionCapability()
        ctx = [
            {"type": "compaction", "summary": "v1"},
            {"type": "user", "content": "between"},
            {"type": "compaction", "summary": "v2"},
            {"type": "user", "content": "after"},
        ]
        result = c.process_context(ctx)
        assert result[0]["summary"] == "v2"
        assert len(result) == 2


class TestPolicyValidator:
    def test_dict_static_coerced(self) -> None:
        c = CompactionCapability(policy={"type": "static", "threshold": 100})  # type: ignore[arg-type]
        assert isinstance(c.policy, StaticCompactionPolicy)
        assert c.policy.threshold == 100

    def test_dict_dynamic_coerced(self) -> None:
        c = CompactionCapability(
            policy={  # type: ignore[arg-type]
                "type": "dynamic",
                "model_info": {"context_window": 128_000},
                "threshold": 0.8,
            },
        )
        assert isinstance(c.policy, DynamicCompactionPolicy)
        assert c.policy.threshold == 0.8

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported compaction policy"):
            CompactionCapability(policy={"type": "weird"})  # type: ignore[arg-type]
