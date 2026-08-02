"""Tests for ``cost_usd`` field on :class:`GenerationSpanData`."""

from troopai.adk.types.tracing.span_data import GenerationSpanData


def test_generation_span_data_has_cost_usd() -> None:
    d = GenerationSpanData(model="m", cost_usd=0.004)
    assert d.cost_usd == 0.004
    assert d.export()["cost_usd"] == 0.004


def test_generation_span_data_cost_usd_defaults_none() -> None:
    assert GenerationSpanData(model="m").cost_usd is None
