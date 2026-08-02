(references/api/guardrails)=

# Guardrails

Built-in, language-agnostic guardrails — PII, prompt-injection, and
wrong-language — that register on an agent like any hand-written
guardrail.

## Factories

```{eval-rst}
.. autofunction:: troopai.adk.guardrails.pii_guardrail

.. autofunction:: troopai.adk.guardrails.injection_scan_guardrail

.. autofunction:: troopai.adk.guardrails.semantic_scan_guardrail

.. autofunction:: troopai.adk.guardrails.wrong_language_guardrail
```

## Scanners

```{eval-rst}
.. autoclass:: troopai.adk.guardrails.PatternScanner
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.guardrails.SemanticScanner
   :members:
   :show-inheritance:

.. autoclass:: troopai.adk.guardrails.SemanticMatch
   :members:
   :show-inheritance:
```

## Helpers

```{eval-rst}
.. autofunction:: troopai.adk.guardrails.mask_pii_spans

.. autofunction:: troopai.adk.guardrails.fence_untrusted_text

.. autofunction:: troopai.adk.guardrails.detect_wrong_language
```

## Defaults

```{eval-rst}
.. autodata:: troopai.adk.guardrails.DEFAULT_PII_MASK

.. autodata:: troopai.adk.guardrails.DEFAULT_INJECTION_EXEMPLARS
```

Three further defaults are documented in prose because their source
docstrings do not render through autodoc:

- `DEFAULT_PII_PATTERNS` — cheap, deterministic regex markers for the
  common injected identifiers: email addresses (ASCII and
  internationalized), URLs, and phone numbers. Override via the
  `patterns` argument of `pii_guardrail`.
- `DEFAULT_INJECTION_PATTERNS` — high-confidence injection markers:
  broad English phrases plus the classic "ignore the previous
  instructions" signature across FR/DE/ES/PT/RU/ZH/JA/HI/AR. Override
  via the `patterns` argument of `injection_scan_guardrail`.
- `DEFAULT_LANGUAGE_CODES` — target-language name to ISO 639-1 code,
  spanning every language the detector supports. Override via the
  `language_codes` argument of `wrong_language_guardrail`.

Agent-level guardrail configuration is documented under the
[Guardrails guide](../../guardrails/guardrail_hub.md).
