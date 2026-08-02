# Built-in Guardrail Hub

Reusable, non-overlapping built-in guardrails, reusing the agent-level
decorators (no new container type). The pattern-based guardrails share one
scanner; the regexes/codes are the framework-owned source of truth (the
translation example consumes them, not the other way round).

## Files

| Path | Purpose |
|---|---|
| `scan.py` | `PatternScanner` — the one shared regex scanner (`scan` → labels, `find_spans` → `GuardrailSpan`s). Pre-compiled, validated at construction. |
| `pii.py` | `pii_guardrail` (output) + `DEFAULT_PII_PATTERNS` + `mask_pii_spans`. |
| `injection.py` | `injection_scan_guardrail` (input) + `DEFAULT_INJECTION_PATTERNS` + `fence_untrusted_text`. |
| `semantic.py` | `semantic_scan_guardrail` (input) + `SemanticScanner` + `DEFAULT_INJECTION_EXEMPLARS` — embedding-codebook injection scan (per-sentence max-cosine via the `Embedder` ABC). |
| `language.py` | `wrong_language_guardrail` (output) + `detect_wrong_language` + `DEFAULT_LANGUAGE_CODES`. |

## Key decisions

- **One scanner, reused.** Both `pii` and `injection` construct a
  `PatternScanner`; neither carries its own match loop. `find_spans` is the
  scanner's observation of the checked text — distinct from a verdict's
  `changed_spans` (what a transform reports), even when they coincide here.
- **`on_fail=RAISE` everywhere by default** — the cost-conservative halt.
  `TRANSFORM` is always opt-in.
- **PII TRANSFORM is wholesale.** On `on_fail=TRANSFORM` (str outputs only)
  the validator computes the COMPLETE anonymized string as
  `transformed_output` (default redactor masks each `find_spans` span with
  `[REDACTED_PII]`); `changed_spans` is observability only and is never
  spliced. The verdict also sets `tripwire_triggered=True` as the halt
  fallback for when the runner cannot substitute (non-text output / no
  transform sink). The factory therefore rejects a non-halting severity
  (`WARNING`/`INFO`) paired with `TRANSFORM` — it would silence that fallback
  and leak masked PII. Already-streamed tokens cannot be recalled; the mask
  lands on the final output and the persisted history.
- **Injection: fence + scan.** `fence_untrusted_text` is the structural,
  language-agnostic protection (a nonce makes the closing delimiter
  unforgeable) — a plain prompt helper, not a verdict. The scan is best-effort:
  broad English markers plus the classic "ignore the previous instructions"
  signature across nine languages (FR/DE/ES/PT/RU/ZH/JA/HI/AR). It backs up the
  fence and the semantic scan — it does not replace them; the regex cannot
  enumerate every phrasing, so the fence stays the guarantee. Input has no
  replaceable artifact, so `injection_scan_guardrail` supports only
  `on_fail=RAISE`.
- **PII patterns are internationalized.** The `email` marker matches Unicode
  local parts and IDN domains, not just ASCII; `\d` already covers non-Latin
  numerals, so the `phone` marker is script-agnostic.
- **Wrong-language is reusable + lazy.** `detect_wrong_language` is the
  importable core; `lingua` is imported lazily and its detector — covering all
  75 languages in `DEFAULT_LANGUAGE_CODES`, deterministic, abstaining on short or
  ambiguous strings instead of guessing (a relative-distance gate, so a `RAISE`
  guardrail never trips on valid short output) — is built once and cached, so the
  dependency is only required when the check runs. A `detector` override accepts
  a subset/tuned detector for a tighter footprint. A missing install raises a
  clear `ImportError` pointing at the `guardrails-lingua` extra. `on_fail=RAISE`
  only — a mistranslation cannot be masked away.

## Dependency

`pip install 'troopai-adk-python[guardrails-lingua]'` (umbrella: `[guardrails]`).
The PII + injection scanners are stdlib-only; only the wrong-language check
needs the extra.

- **Semantic scan: training-free codebook, `Embedder`-agnostic.** Adapted
  from Guardrails AI `DetectJailbreak` (max cosine of the prompt embedding
  against `KNOWN_ATTACKS`, default combined threshold 0.81) and NeMo
  Guardrails' embedding-based jailbreak input rail. Framework adaptations:
  the guardrail consumes the provider-agnostic `Embedder` ABC (no torch/onnx
  dep; passing an embedder IS the explicit cost opt-in), and text is scanned
  one window *per sentence* so one injected sentence in a document-sized
  prompt isn't diluted — not by whole-text pooling, and not by a benign
  sentence sharing its window (packing several sentences re-introduces the
  same dilution). Codebook embedded lazily once
  per scanner (asyncio-locked). Exemplars are English by design — a
  multilingual embedding model projects translations/paraphrases into the
  same neighbourhood. Input side ⇒ `on_fail=RAISE` only. Threshold
  distributions differ per embedding model; calibrate before tightening.

## Faithfulness / embedding output validator

An embedding-based *output faithfulness* validator is still deliberately NOT
shipped — it overlaps `wrong_language` and has no consumer. (The input-side
`semantic.py` scan is a different animal: it landed with a confirmed consumer
— the document-translation example — and adds no provider dependency because
it consumes the `Embedder` ABC.)
