# Voice Module

Speech-to-text → agent workflow → text-to-speech pipeline. Provider-agnostic
abstractions live here; concrete provider models live in `llms/<provider>/`.

## Files

| File | Purpose |
|---|---|
| `audio.py` | `AudioInput` (buffered) + `StreamedAudioInput` (live), PCM constants, optional numpy→PCM helpers |
| `stt.py` | `STTModel` / `StreamedTranscriptionSession` ABCs, `STTModelSettings`, `TurnDetection` |
| `tts.py` | `TTSModel` ABC, `TTSModelSettings` |
| `events.py` | `VoiceStreamEvent` union — audio / lifecycle / error |
| `splitter.py` | `sentence_splitter` — cuts streamed text into synthesis segments |
| `workflow.py` | `VoiceWorkflow` ABC + `SingleAgentVoiceWorkflow` (drives the Runner) |
| `result.py` | `StreamedAudioResult` — ordered synthesis + balanced lifecycle |
| `pipeline_config.py` | `VoicePipelineConfig` |
| `pipeline.py` | `VoicePipeline` — entry point; buffered + streamed orchestration |

Concrete models: `llms/openai/openai_stt.py` (`OpenAISTTModel`),
`llms/openai/openai_tts.py` (`OpenAITTSModel`).

## Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Provider models live in `llms/<provider>/`, NOT here | The standing invariant: every provider SDK import is confined to its provider module. `voice/` defines the ABCs (mirroring the `LLM` ABC); the OpenAI models implement them next to the other OpenAI models. |
| 2 | Audio is raw PCM `bytes` on every public surface | Keeps `import voice` and the buffered path free of any array library. `numpy` is an optional convenience (`pcm16_from_*`) for callers converting microphone sample arrays. |
| 3 | STT and TTS models are required and explicit on `VoicePipeline` | No silent fallback to a default provider — the framework never adds a provider (or its dependency, or its cost) the developer did not choose. |
| 4 | Trace capture of transcripts and audio defaults to off | Conservative default: the developer opts *in* to recording sensitive speech, never has to opt out of it. |
| 5 | Spoken text is extracted provider-agnostically from the Runner stream | `SingleAgentVoiceWorkflow` reads `RawResponseStreamEvent` text deltas. Reasoning and tool-call argument deltas are filtered upstream, so the model never speaks its thinking — no provider-specific event-type check needed. |
| 6 | One ordered synthesis task; bounded output queue | A single consumer of the segment queue synthesizes per segment, so audio never interleaves across segments. The output queue is bounded, so a slow listener back-pressures synthesis instead of buffering audio without limit. |
| 7 | `turn_started` / `turn_ended` are always emitted as a balanced pair | Emitted eagerly at turn boundaries, not lazily on first audio — so a turn that produces no audio still brackets cleanly and UI state machines never desync. |
| 8 | `workflow.py` imports the Runner lazily (inside `run`) | Module load of `voice` stays free of any `run` dependency, so the provider models under `llms/openai/` can import the voice ABCs without a circular import. |
| 9 | Realtime STT uses the `websockets` library directly, confined to `llms/openai/` | The transcription-intent handshake needs precise control the SDK's helpers do not expose. It is gated behind the `voice` extra and soft-imported, so the buffered path and the ABCs work without it. |

## Layering

`STTModel` / `TTSModel` are framework-owned ABCs, the speech analog of the
`LLM` ABC. `VoiceStreamEvent` variants are plain `@dataclass` discriminated
unions, mirroring the runner's boundary stream events rather than the
validation-heavy response types. No wire types cross the ABC boundary.

## See Also

- `docs/voice/` — pipeline walkthrough, buffered vs streamed, tracing.
- `examples/voice/` — runnable buffered + streamed examples.
- `llms/openai/CLAUDE.md` — the OpenAI speech model implementations.
- `architecture.md` — provider-isolation and cost-conservative-default invariants.
