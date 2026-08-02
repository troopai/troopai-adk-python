"""Tests for CompositeSpan.data fan-out to child spans.

When runner code rebinds ``span.data`` after the LLM call (e.g.
``gen_span.data = dataclasses.replace(gen_span.data, usage=...)``)
the new value must reach every child span so composed tracers observe
the final payload at ``finish()``.
"""

from __future__ import annotations

import dataclasses

from troopai.adk.tracing.multi_tracer import CompositeSpan
from troopai.adk.tracing.spans import Span
from troopai.adk.types.tracing.span_data import GenerationSpanData


def test_composite_data_rebind_reaches_children() -> None:
    d = GenerationSpanData(model="m")
    child_a: Span[GenerationSpanData] = Span(d)
    child_b: Span[GenerationSpanData] = Span(d)
    composite = CompositeSpan([child_a, child_b], d)
    composite.data = dataclasses.replace(composite.data, usage={"input_tokens": 5})
    assert child_a.data.usage == {"input_tokens": 5}
    assert child_b.data.usage == {"input_tokens": 5}
    assert composite.data.usage == {"input_tokens": 5}
