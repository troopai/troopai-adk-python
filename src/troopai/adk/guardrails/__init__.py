"""Built-in, language-agnostic guardrails reusing the agent-level decorators.

A small, non-overlapping set of reusable guardrails — PII, prompt-injection
(regex and embedding-based), and wrong-language — built on one shared regex
scanner plus an embedding codebook scanner. They return the same
``AgentGuardrailFunctionOutput`` verdicts as hand-written guardrails, so they
register on an agent like any other.

Each factory defaults to ``on_fail=GuardrailAction.RAISE`` (cost-conservative
halt); ``TRANSFORM`` (output anonymization) is always opt-in.

Example:
    from troopai.adk.agents import Agent, AgentGuardrails
    from troopai.adk.guardrails import injection_scan_guardrail, pii_guardrail
    from troopai.adk.types.guardrails import GuardrailAction

    agent = Agent(
        name="Support",
        system_prompt="Help customers.",
        guardrails=AgentGuardrails(
            input=[injection_scan_guardrail()],
            output=[pii_guardrail(on_fail=GuardrailAction.TRANSFORM)],
        ),
    )
"""

from __future__ import annotations

from troopai.adk.guardrails.injection import (
    DEFAULT_INJECTION_PATTERNS,
    fence_untrusted_text,
    injection_scan_guardrail,
)
from troopai.adk.guardrails.language import (
    DEFAULT_LANGUAGE_CODES,
    detect_wrong_language,
    wrong_language_guardrail,
)
from troopai.adk.guardrails.pii import (
    DEFAULT_PII_MASK,
    DEFAULT_PII_PATTERNS,
    mask_pii_spans,
    pii_guardrail,
)
from troopai.adk.guardrails.scan import PatternScanner
from troopai.adk.guardrails.semantic import (
    DEFAULT_INJECTION_EXEMPLARS,
    SemanticMatch,
    SemanticScanner,
    semantic_scan_guardrail,
)

__all__ = [
    "DEFAULT_INJECTION_EXEMPLARS",
    "DEFAULT_INJECTION_PATTERNS",
    "DEFAULT_LANGUAGE_CODES",
    "DEFAULT_PII_MASK",
    "DEFAULT_PII_PATTERNS",
    "PatternScanner",
    "SemanticMatch",
    "SemanticScanner",
    "detect_wrong_language",
    "fence_untrusted_text",
    "injection_scan_guardrail",
    "mask_pii_spans",
    "pii_guardrail",
    "semantic_scan_guardrail",
    "wrong_language_guardrail",
]
