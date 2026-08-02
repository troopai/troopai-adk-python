"""Input prompt-injection scan plus a structural nonce fence for untrusted text.

Two complementary defences:

- ``fence_untrusted_text`` is the language-agnostic, structural protection: it
  wraps untrusted text between delimiters carrying a random nonce, so injected
  instructions cannot forge the closing delimiter and "escape" the fence. It is a
  plain prompt-construction helper, not a guardrail verdict.
- ``injection_scan_guardrail`` is a best-effort, multilingual input scan that
  halts when high-confidence injection markers appear in the prompt (English plus
  the classic injection signature across FR/DE/ES/PT/RU/ZH/JA/HI/AR). It backs up
  the fence; it does not replace it.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Mapping
from typing import Any

from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrailSeverity,
    AgentInputGuardrail,
    AgentInputGuardrailData,
)
from troopai.adk.guardrails.scan import PatternScanner
from troopai.adk.types.guardrails.action import GuardrailAction

__all__ = [
    "DEFAULT_INJECTION_PATTERNS",
    "fence_untrusted_text",
    "injection_scan_guardrail",
]

logger = logging.getLogger(__name__)

DEFAULT_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_previous": re.compile(r"(?i)\b(ignore|disregard|forget)\b.{0,20}\b(previous|prior|above|all)\b"),
    "new_instructions": re.compile(r"(?i)\bnew\s+instructions?\b"),
    "role_override": re.compile(r"(?i)\byou\s+are\s+now\b|\bact\s+as\b|\bsystem\s+prompt\b"),
    "role_tag": re.compile(r"(?im)^\s*(system|assistant|user)\s*:"),
    "exfiltrate": re.compile(r"(?i)\b(reveal|print|repeat)\b.{0,20}\b(prompt|instructions?)\b"),
    # The classic "ignore the previous instructions" signature in more languages
    # — an ignore-verb AND an adjacent *previous/above* reference. The reference,
    # not a bare instruction-noun, is the trigger: several instruction words are
    # English cognates ("instructions", "instru…"), so keying on them would fire
    # on benign prose ("do not ignore these safety instructions"). Requiring the
    # directional qualifier keeps these high-confidence. Broader English phrases
    # above stay English. Spanish/Portuguese share BOTH the verb "ignore" and the
    # qualifier "anterior" with English, so those two additionally require a
    # non-English instruction-noun (instrucción/instrução/…) — otherwise benign
    # English "ignore the anterior wall" would trip them.
    "ignore_previous_fr": re.compile(
        r"(?i)\b(ignore[zr]?|oublie[zr]?|négligez?)\b.{0,30}\b(précédent\w*|antérieur\w*|ci-dessus|au-dessus|ci-avant|plus haut)\b"
    ),
    "ignore_previous_de": re.compile(
        r"(?i)\b(ignoriere\w*|vergiss|missachte)\b.{0,30}\b(vorherig\w*|vorig\w*|obig\w*|vorangehend\w*|weiter\s+oben)\b"
    ),
    "ignore_previous_es": re.compile(
        r"(?i)\b(ignora|ignore|olvida|olvide|omite)\b.{0,25}\b(instruccion\w*|indicacion\w*|[oó]rden\w*|regla\w*|mensaje\w*|directriz|directrices)\b.{0,15}\b(anterior\w*|previa\w*|previo\w*|arriba|encima)\b"
    ),
    "ignore_previous_pt": re.compile(
        r"(?i)\b(ignor[ae]|esque[çc]a|desconsidere?)\b.{0,25}\b(instru[çc]\w*|indica[çc]\w*|ordem|ordens|regra\w*|mensage\w*)\b.{0,15}\b(anterior\w*|acima|pr[ée]vi\w*)\b"
    ),
    "ignore_previous_ru": re.compile(
        r"(?i)\b(игнорир\w*|проигнорир\w*|забудь\w*)\b.{0,30}\b(предыдущ\w*|вышеуказанн\w*|выше)\b"
    ),
    "ignore_previous_zh": re.compile(r"忽略[^\n]{0,12}(之前|上述|所有|先前|以上)[^\n]{0,8}(指令|指示|提示|命令)"),
    "ignore_previous_ja": re.compile(
        r"(以前|上記|前述|すべて|全て|これまで)[^\n]{0,10}(指示|命令|指令)[^\n]{0,8}(無視|忘れ)"
    ),
    "ignore_previous_hi": re.compile(
        r"(पिछल\w*|पूर्व\w*|सभी|उपरोक्त)[^\n]{0,18}(निर्देश\w*|आदेश\w*)[^\n]{0,12}(अनदेखा|नज़रअंदाज|भूल)"
    ),
    "ignore_previous_ar": re.compile(r"(تجاهل|انس\w*|أهمل)[^\n]{0,30}(السابق\w*|أعلاه|السالف\w*)"),
}
"""High-confidence injection markers. Broad English phrases (role assignment,
"new instructions") plus the classic "ignore the previous instructions"
signature across FR/DE/ES/PT/RU/ZH/JA/HI/AR. Best-effort detection that backs up
— never replaces — the language-agnostic ``fence_untrusted_text``; broader
non-English phrasing is not enumerable by regex and is left to the fence and the
embedding-codebook ``semantic_scan_guardrail``."""


def fence_untrusted_text(text: str, *, nonce: str | None = None) -> str:
    """Wrap untrusted ``text`` in nonce-fenced delimiters for prompt construction.

    The random ``nonce`` makes the closing delimiter unforgeable from inside the
    source, so injected text cannot terminate the fence early. Any literal copy
    of the (nonce-bearing) markers already present in ``text`` is stripped as a
    defence in depth. This is a structural helper, not a verdict — it never
    inspects the content for meaning.

    Args:
        text: The untrusted text to isolate.
        nonce: Optional caller-supplied nonce. When ``None``, a fresh random
            token is generated, which is the recommended usage.

    Returns:
        ``text`` wrapped between open/close markers that embed the nonce.

    Raises:
        ValueError: If an explicit empty ``nonce`` is supplied.
    """
    token = nonce if nonce is not None else secrets.token_hex(8)
    if len(token) == 0:
        raise ValueError("fence_untrusted_text nonce must be non-empty")
    open_marker = f"<<UNTRUSTED {token}>>"
    close_marker = f"<<END_UNTRUSTED {token}>>"
    safe = text.replace(open_marker, "").replace(close_marker, "")
    return f"{open_marker}\n{safe}\n{close_marker}"


def injection_scan_guardrail(
    *,
    on_fail: GuardrailAction = GuardrailAction.RAISE,
    patterns: Mapping[str, re.Pattern[str]] | None = None,
    name: str = "injection_scan",
    severity: AgentGuardrailSeverity | None = None,
    run_in_parallel: bool = False,
) -> AgentInputGuardrail[Any]:
    """Build an input guardrail that halts when injection markers appear.

    Args:
        on_fail: Only ``RAISE`` is supported — the prompt is not a replaceable
            artifact, so ``TRANSFORM``/``PASS`` do not apply on the input side.
        patterns: Override the default label → compiled-pattern map.
        name: Guardrail name surfaced in results and tracing.
        severity: Verdict severity (e.g. ``WARNING`` to detect-and-log without
            halting). ``None`` (default) lets the tripwire halt the run.
        run_in_parallel: Defaults to ``False`` so the scan blocks before the
            agent runs and saves tokens when it trips.

    Returns:
        An ``AgentInputGuardrail`` ready to register on an agent.

    Raises:
        ValueError: If ``on_fail`` is not ``RAISE``, or ``patterns`` is empty.
    """
    if on_fail is not GuardrailAction.RAISE:
        raise ValueError(
            "injection_scan_guardrail supports only on_fail=RAISE: a prompt is not a replaceable "
            "artifact, so TRANSFORM/PASS do not apply on the input side."
        )
    if patterns is not None and len(patterns) == 0:
        raise ValueError("injection_scan_guardrail patterns must be non-empty when provided")
    scanner = PatternScanner(patterns=patterns if patterns is not None else DEFAULT_INJECTION_PATTERNS)

    async def check(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
        markers = scanner.scan(str(data.user_prompt))
        if len(markers) == 0:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)
        logger.warning("Prompt-injection markers detected: %s", markers)
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True, severity=severity, output_info={"markers": markers}
        )

    return AgentInputGuardrail(guardrail_function=check, name=name, run_in_parallel=run_in_parallel)
