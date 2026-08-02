"""Exception hierarchy for the voice subsystem.

Every voice error derives from :class:`VoiceError`, which in turn
derives from the framework-wide :class:`~troopai.adk.exceptions.TroopAIError`.
Catching ``TroopAIError`` therefore catches voice failures too, while
``VoiceError`` narrows to the speech pipeline.
"""

from __future__ import annotations

from troopai.adk.exceptions import TroopAIError


class VoiceError(TroopAIError):
    """Base class for every failure raised by the voice subsystem."""


class STTError(VoiceError):
    """Speech-to-text transcription failed."""


class TTSError(VoiceError):
    """Text-to-speech synthesis failed."""


class STTWebsocketError(STTError):
    """A realtime speech-to-text websocket session failed.

    Raised when the realtime transcription websocket cannot be
    established, sends a protocol error, or closes unexpectedly mid
    session.
    """
