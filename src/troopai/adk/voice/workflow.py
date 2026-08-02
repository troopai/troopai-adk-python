"""Voice workflows — what an agent *says* in response to a transcript.

A :class:`VoiceWorkflow` is the brain between speech-to-text and
text-to-speech: it receives a transcript and yields the text to speak,
streaming so synthesis can begin before the full answer is generated.

:class:`SingleAgentVoiceWorkflow` is the common case — it drives one
:class:`~troopai.adk.agents.agent.Agent` through the
:class:`~troopai.adk.run.runner.Runner`, accumulating conversation
history across turns and following handoffs. It extracts spoken text
provider-agnostically from the runner's stream: only assistant *text*
deltas are surfaced, so the model's reasoning and tool-call arguments
are never spoken.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

logger = logging.getLogger(__name__)


@runtime_checkable
class VoiceWorkflowCallbacks(Protocol):
    """Optional hooks observing a single-agent voice workflow."""

    def on_transcription(self, transcription: str) -> None:
        """Called with each finalized transcript before the agent runs."""


class VoiceWorkflow(ABC):
    """Maps a transcript to streamed text for synthesis.

    Subclasses implement :meth:`run`. :meth:`on_start` optionally emits a
    greeting before the first transcript (used by continuous sessions).
    """

    @abstractmethod
    def run(self, transcription: str) -> AsyncIterator[str]:
        """Yield text to speak in response to a transcript.

        Args:
            transcription: The finalized transcript of one user turn.

        Yields:
            Text fragments to synthesize, in order.
        """

    async def on_start(self) -> AsyncIterator[str]:
        """Yield optional text to speak before the first transcript.

        The default implementation yields nothing. Override to greet the
        user when a continuous session opens.

        Yields:
            Greeting text fragments, in order.
        """
        greetings: tuple[str, ...] = ()
        for greeting in greetings:
            yield greeting


class SingleAgentVoiceWorkflow(VoiceWorkflow):
    """Drives one agent per turn, keeping history and following handoffs.

    Each :meth:`run` call appends the transcript to the running history,
    streams the agent's spoken text, then folds the turn's items back
    into history so the next turn has full context. When the agent hands
    off, the active agent is updated for subsequent turns.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        context: RunContext | None = None,
        run_config: RunConfig | None = None,
        callbacks: VoiceWorkflowCallbacks | None = None,
    ) -> None:
        self._agent = agent
        self._context = context
        self._run_config = run_config
        self._callbacks = callbacks
        self._input_history: list[LLMInputContentItem] = []

    @override
    async def run(self, transcription: str) -> AsyncIterator[str]:
        # Lazy imports: keep module load of ``voice`` free of any ``run``
        # dependency so the provider models under ``llms/openai`` can
        # import the voice ABCs without a circular import.
        from troopai.adk.run import Runner
        from troopai.adk.run.stream import RawResponseStreamEvent

        if self._callbacks is not None:
            self._callbacks.on_transcription(transcription)

        user_message: LLMInputEasyMessage = {"role": "user", "content": transcription}
        self._input_history.append(user_message)

        result = await Runner.arun(
            self._agent,
            self._input_history,
            context=self._context,
            run_config=self._run_config,
            stream=True,
        )
        async for event in result.stream_events():
            # Only assistant text deltas reach ``RawResponseStreamEvent`` —
            # reasoning and tool-call argument deltas are filtered upstream,
            # so the model never speaks its thinking.
            if isinstance(event, RawResponseStreamEvent) and isinstance(event.data, str) and len(event.data) > 0:
                yield event.data

        self._input_history = result.to_input_list()
        self._agent = result.current_agent
