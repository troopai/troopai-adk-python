"""URL context hosted tool (Gemini only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from troopai.adk.tools.hosted.hosted_tool import HostedTool


@dataclass(kw_only=True)
class URLContextTool(HostedTool):
    """Provider-hosted URL retrieval and context grounding.

    The model fetches text from URLs the user mentions and grounds its
    response in the retrieved content.

    Provider matrix:
    - **Gemini**: native via the ``url_context`` tool type. No
      tunable knobs at this time.
    - All other providers raise :class:`UnsupportedHostedToolError`.

    Refs:
        - Gemini: https://ai.google.dev/gemini-api/docs/url-context
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ("gemini",)
