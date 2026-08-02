"""Example: a buffered voice round-trip with the OpenAI speech models.

This wires the full voice pipeline — speech-to-text, an agent, and
text-to-speech — and runs one complete turn:

1. Text-to-speech synthesizes a spoken *question* (this stands in for a
   microphone recording, so the example is self-contained).
2. The pipeline transcribes that audio, runs an agent to answer it, and
   synthesizes the spoken *answer*.
3. The answer audio is written to a WAV file you can play.

The speech models are explicit — the pipeline never assumes a provider.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/voice/voice_pipeline.py

Install the voice extra for the realtime/streaming path and microphone
helpers:
    pip install 'troopai-adk-python[voice]'
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import io
import logging
import wave

from troopai.adk.agents import Agent
from troopai.adk.llms.openai import OpenAISTTModel, OpenAITTSModel
from troopai.adk.voice import (
    DEFAULT_SAMPLE_RATE,
    AudioInput,
    SingleAgentVoiceWorkflow,
    TTSModelSettings,
    VoicePipeline,
    VoiceWorkflowCallbacks,
)

logger = logging.getLogger(__name__)

QUESTION = "What is the capital of France? Answer in one short sentence."
OUTPUT_WAV = "voice_pipeline_answer.wav"


class PrintTranscription(VoiceWorkflowCallbacks):
    """Logs the transcript the speech-to-text model recognized."""

    def on_transcription(self, transcription: str) -> None:
        logger.info("[heard] %s", transcription)


async def synthesize_question(tts: OpenAITTSModel) -> bytes:
    """Use text-to-speech to produce the spoken question as PCM bytes."""
    pcm = bytearray()
    async for chunk in tts.run(QUESTION, TTSModelSettings(voice="nova")):
        pcm.extend(chunk)
    return bytes(pcm)


def write_wav(path: str, pcm: bytes) -> None:
    """Write mono 16-bit PCM to a WAV file at the pipeline sample rate."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(DEFAULT_SAMPLE_RATE)
        wav_file.writeframes(pcm)
    with open(path, "wb") as handle:
        handle.write(buffer.getvalue())


async def main() -> None:
    tts = OpenAITTSModel()
    stt = OpenAISTTModel()

    # Stand in for a microphone recording by synthesizing the question.
    question_pcm = await synthesize_question(tts)
    logger.info("Synthesized question audio: %d PCM bytes", len(question_pcm))

    agent = Agent(
        name="VoiceAssistant",
        system_prompt="You are a concise, friendly voice assistant. Keep replies to one sentence.",
        llm="gpt-4o-mini",
    )
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent, callbacks=PrintTranscription()),
        stt_model=stt,
        tts_model=tts,
    )

    result = await pipeline.run(AudioInput(data=question_pcm))

    answer_pcm = bytearray()
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            answer_pcm.extend(event.data)
        elif event.type == "voice_stream_event_lifecycle":
            logger.info("[lifecycle] %s", event.event)
        elif event.type == "voice_stream_event_error":
            logger.error("[error] %s", event.error)

    write_wav(OUTPUT_WAV, bytes(answer_pcm))
    logger.info("Wrote spoken answer (%d PCM bytes) to %s", len(answer_pcm), OUTPUT_WAV)


if __name__ == "__main__":
    asyncio.run(main())
