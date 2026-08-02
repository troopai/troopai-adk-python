"""Output wrong-language guardrail plus a reusable language-mismatch detector.

The wrong-language check catches refusals, untranslated text, and hijacked
output across 75 languages: if the output is not in the expected target
language, the verdict trips. ``detect_wrong_language`` is the reusable core; the
guardrail factory wraps it for the output phase.

``lingua`` is imported lazily inside the detector, and its detector — which
covers every language in ``DEFAULT_LANGUAGE_CODES``, is deterministic, and
abstains rather than guess when a short or ambiguous string leaves its top
guesses too close (so short valid output is skipped, not false-flagged) — is
built once and cached, so the dependency is only needed when the check actually
runs. A missing install surfaces a clear, actionable ``ImportError``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrailSeverity,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
)
from troopai.adk.types.guardrails.action import GuardrailAction

if TYPE_CHECKING:
    from lingua import LanguageDetector

__all__ = [
    "DEFAULT_LANGUAGE_CODES",
    "detect_wrong_language",
    "wrong_language_guardrail",
]

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE_CODES: dict[str, str] = {
    "afrikaans": "af",
    "albanian": "sq",
    "arabic": "ar",
    "armenian": "hy",
    "azerbaijani": "az",
    "basque": "eu",
    "belarusian": "be",
    "bengali": "bn",
    "bokmal": "nb",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "esperanto": "eo",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "ganda": "lg",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "irish": "ga",
    "italian": "it",
    "japanese": "ja",
    "kazakh": "kk",
    "korean": "ko",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "maori": "mi",
    "marathi": "mr",
    "mongolian": "mn",
    "nynorsk": "nn",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "shona": "sn",
    "slovak": "sk",
    "slovene": "sl",
    "somali": "so",
    "sotho": "st",
    "spanish": "es",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "tsonga": "ts",
    "tswana": "tn",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
    "welsh": "cy",
    "xhosa": "xh",
    "yoruba": "yo",
    "zulu": "zu",
    # Common alternate names for the same detector language.
    "farsi": "fa",
    "filipino": "tl",
    "mandarin": "zh",
    "norwegian": "nb",
    "slovenian": "sl",
}
"""Target-language name → ISO 639-1 code, spanning every language the detector
recognises (plus common alternate names). A target absent here skips the check.
``chinese`` collapses Simplified/Traditional to ``zh``; ``norwegian`` maps to
Bokmål (``nb``), the dominant written form."""


@lru_cache(maxsize=1)
def _get_detector() -> LanguageDetector:
    """Build (once) and cache the default ``lingua`` detector over all languages.

    ``lingua`` is imported lazily so the dependency is only needed when the check
    runs; the detector is cached because building it is the expensive step. The
    full-language, high-accuracy detector is the robust default: on the first
    ``detect_language_of`` call it loads the high-accuracy model set (which is
    memory-heavy), so a caller who wants a tighter footprint or lower latency
    should build a subset detector (``from_languages(...)``) and pass it via the
    ``detector`` seam on the public functions below.

    A minimum-relative-distance gate makes the detector abstain (return ``None``)
    when its top guesses are close — otherwise it always commits to a single best
    guess and misclassifies short or ambiguous strings (``"OK"``, ``"Total: 42"``,
    a lone ``"Merci."``), which on a ``RAISE`` guardrail would halt the run on
    valid output. The gate keeps abstention for those while still resolving clear
    sentence-length text (including refusals long enough to identify); a very
    short refusal may itself abstain, which is the safe direction here.

    Raises:
        ImportError: If the optional ``lingua`` package is not installed.
    """
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError as exc:
        raise ImportError(
            "wrong-language detection needs the optional `lingua` package. "
            "Install it with: pip install 'troopai-adk-python[guardrails-lingua]'"
        ) from exc
    return LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0.25).build()


def _detect_language(text: str, detector: LanguageDetector | None = None) -> str | None:
    """Return the detected ISO 639-1 code for ``text``, or ``None`` if unknown.

    Args:
        text: The text to identify.
        detector: An explicit ``lingua`` detector to use. When ``None``, the
            cached default full-language detector is used.

    Returns:
        The lowercase ISO 639-1 code, or ``None`` when the detector cannot
        confidently identify the language.

    Raises:
        ImportError: If the optional ``lingua`` package is not installed.
    """
    active = detector if detector is not None else _get_detector()
    language = active.detect_language_of(text)
    if language is None:
        return None
    return language.iso_code_639_1.name.lower()


def detect_wrong_language(
    text: str,
    target_language: str,
    *,
    language_codes: Mapping[str, str] | None = None,
    detector: LanguageDetector | None = None,
) -> str | None:
    """Return a mismatch reason if ``text`` is not in ``target_language``, else ``None``.

    Skips silently (returns ``None``) when ``text`` is blank, the target is not
    in ``language_codes``, or the detector cannot confidently identify the
    language. Chinese is a single detector language (``zh``).

    Args:
        text: The candidate output text.
        target_language: Expected target-language name (case-insensitive).
        language_codes: Override the default name → ISO 639-1 code map.
        detector: Override the default full-language ``lingua`` detector (e.g. a
            ``from_languages(...)`` subset for a tighter footprint). When
            ``None``, the cached default detector is used.

    Returns:
        A short mismatch reason (``"expected <code>, detected <code>"``), or
        ``None`` when the output is acceptable or the check does not apply.

    Raises:
        ImportError: If the optional ``lingua`` package is not installed.
    """
    codes = language_codes if language_codes is not None else DEFAULT_LANGUAGE_CODES
    if len(text.strip()) == 0:
        return None
    target_code = codes.get(target_language.lower())
    if target_code is None:
        return None
    detected = _detect_language(text, detector)
    if detected is None:
        return None
    if detected == target_code:
        return None
    return f"expected {target_code}, detected {detected}"


def wrong_language_guardrail(
    *,
    target_language: str | Callable[[AgentOutputGuardrailData], str],
    on_fail: GuardrailAction = GuardrailAction.RAISE,
    name: str = "wrong_language",
    severity: AgentGuardrailSeverity | None = None,
    language_codes: Mapping[str, str] | None = None,
    detector: LanguageDetector | None = None,
) -> AgentOutputGuardrail[Any]:
    """Build an output guardrail that trips when the output is in the wrong language.

    Args:
        target_language: Expected target-language name, or a callable resolving
            it per run from the output guardrail data (e.g. from typed context).
        on_fail: Only ``RAISE`` is supported — a mistranslation cannot be
            auto-corrected by masking, so ``TRANSFORM``/``PASS`` do not apply.
        name: Guardrail name surfaced in results and tracing.
        severity: Verdict severity (e.g. ``WARNING`` to detect-and-log without
            halting). ``None`` (default) lets the tripwire halt the run.
        language_codes: Override the default name → ISO 639-1 code map.
        detector: Override the default full-language ``lingua`` detector (e.g. a
            ``from_languages(...)`` subset for a tighter footprint). When
            ``None``, the cached default detector is used.

    Returns:
        An ``AgentOutputGuardrail`` ready to register on an agent.

    Raises:
        ValueError: If ``on_fail`` is not ``RAISE``, or a literal
            ``target_language`` is empty.
    """
    if on_fail is not GuardrailAction.RAISE:
        raise ValueError(
            "wrong_language_guardrail supports only on_fail=RAISE: a mistranslation cannot be "
            "auto-corrected by masking, so TRANSFORM/PASS do not apply."
        )
    if isinstance(target_language, str) and len(target_language) == 0:
        raise ValueError("wrong_language_guardrail target_language must be non-empty")

    async def check(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        output = data.output
        text = output if isinstance(output, str) else str(output)
        target = target_language(data) if callable(target_language) else target_language
        reason = detect_wrong_language(text, target, language_codes=language_codes, detector=detector)
        if reason is None:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)
        return AgentGuardrailFunctionOutput(
            tripwire_triggered=True,
            severity=severity,
            output_info={"issue": "wrong_language", "reason": reason},
        )

    return AgentOutputGuardrail(guardrail_function=check, name=name)
