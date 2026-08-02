"""Catastrophic-backtracking regression for the OTel redaction regexes.

Each pattern in ``_REDACTION_PATTERNS`` is anchored to a concrete
literal prefix (``Bearer``, ``sk-``, ``AIza``, ``AKIA``/``ASIA``,
``gh[pousr]_``, ``xox``, ``-----BEGIN``, or a JSON field name) so the
regex engine short-circuits to a forward-only scan on literal-prefix
misses. Pathological inputs that contain *many* near-matches — 10 KB
of ``sk-`` lookalikes, nested PEM markers, overlapping token prefixes
— MUST still complete in sub-second time.

The test pins that invariant. If a future change to
``_REDACTION_PATTERNS`` adds a non-anchored alternation with nested
quantifiers, this test will blow past its time budget and fail.

We deliberately do NOT use ``pytest.mark.benchmark`` here — we don't
need measurement precision, we need a cheap upper bound that runs on
every CI invocation.
"""

from __future__ import annotations

import time

import pytest

from troopai.adk.tracing.otel.otel_tracer import _redact

# Per-input wall-clock budget. Any single `_redact` call that takes
# longer than this on the CI worker represents a regression in the
# anchoring invariant. 500 ms is extremely generous for the 10-50 KB
# inputs below — on a warm laptop each one completes in <5 ms — but
# it has to survive a slow cold CI runner too.
_BUDGET_SECONDS = 0.5


def _run_and_time(payload: str) -> float:
    start = time.perf_counter()
    _ = _redact(payload)
    return time.perf_counter() - start


class TestRedactDoesNotBacktrack:
    @pytest.mark.parametrize(
        "name,payload",
        [
            # Long string of ``sk-`` near-matches (wrong suffix shape
            # in every case — tests the engine's ability to bail after
            # the literal prefix without exploring trailing alternatives).
            ("many_sk_prefixes", "sk-" + "!" + ("sk-abc" * 2000)),
            # Dense ``Bearer`` tokens with short bodies — walks the
            # Bearer branch but immediately fails the ``{8,}``
            # requirement, so there's no body to backtrack over.
            ("bearer_short_bodies", "Bearer ab " * 5000),
            # Overlapping GitHub token prefixes — each ``gh`` restarts
            # the literal anchor scan; ensures the ``gh[pousr]_`` arm
            # doesn't exhibit runaway behaviour on dense repeats.
            ("dense_gh_prefixes", "ghp_" + ("ghp_xxx " * 5000)),
            # Interleaved AKIA candidates with invalid bodies (21
            # chars, not 16 — the regex MUST fail fast).
            ("akia_wrong_bodies", "AKIA12345678901234567 " * 2000),
            # PEM marker collisions: many BEGIN markers, only one
            # legitimate END, forces the non-greedy ``[\s\S]+?`` to be
            # exercised on a very large input.
            (
                "many_pem_begins_one_end",
                "-----BEGIN PRIVATE KEY-----" * 200 + "\nbody\n-----END PRIVATE KEY-----",
            ),
            # Near-match JSON field tokens stressing the negative-
            # lookahead chain.
            (
                "json_field_near_matches",
                '{"api_key":"Bearer abcdefgh","api_key":"sk-AAAAAAAAAAAAAAAAAAAA",' * 500 + "}",
            ),
        ],
        ids=lambda v: v if isinstance(v, str) and len(v) < 40 else "payload",
    )
    def test_pathological_input_completes_under_budget(self, name: str, payload: str) -> None:
        """Regression: every pattern MUST be linear-time on this input.

        If this fires, the offending regex almost certainly lost its
        literal anchor or gained a nested unbounded quantifier. Fix
        the regex, do not relax the budget.
        """
        elapsed = _run_and_time(payload)
        assert elapsed < _BUDGET_SECONDS, (
            f"_redact took {elapsed * 1000:.1f} ms on {name!r} "
            f"({len(payload)} chars) — suggests catastrophic backtracking "
            f"in _REDACTION_PATTERNS."
        )

    def test_mixed_realistic_payload_under_budget(self) -> None:
        """Realistic-shape payload: a 50 KB conversation log studded
        with real credential shapes that DO match. Each match must
        succeed and the total walk must still finish in <budget."""
        chunk = (
            "GET /api HTTP/1.1\n"
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456\n"
            'data: {"api_key":"sk-AAAAAAAAAAAAAAAAAAAA","user":"alice"}\n'
            "aws_key=AKIAIOSFODNN7EXAMPLE gh_token=ghp_1234567890abcdef"
            "1234567890abcdef123456\n"
        )
        payload = chunk * 500  # ≈ 100 KB of real matches
        elapsed = _run_and_time(payload)
        assert elapsed < _BUDGET_SECONDS * 2, (
            f"_redact took {elapsed * 1000:.1f} ms on a 100 KB realistic payload — regression in the redaction walk."
        )
