"""Stream incremental text from a remote A2A agent.

Usage::

    pip install 'troopai-adk-python[a2a]'
    python examples/a2a/client_streaming.py [URL]
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import sys

from troopai.adk.a2a import A2AAgent, A2ARunner

logger = logging.getLogger(__name__)


async def main(remote_url: str) -> None:
    async with A2AAgent(name="RemoteWriter", url=remote_url) as remote:
        chunks: list[str] = []
        async for event in await A2ARunner.arun(
            remote,
            "Write a short paragraph about the A2A protocol.",
            stream=True,
        ):
            kind = event["type"]
            if kind == "text_delta":
                chunks.append(event["text_delta"])
                logger.info("delta: %s", event["text_delta"])
            elif kind == "status":
                logger.debug("status: %s", event.get("state"))
            elif kind == "completed":
                logger.info(
                    "completed; total length=%d chars",
                    len(event.get("result_text", "")),
                )
                break
            elif kind == "failed":
                logger.error(
                    "stream failed: state=%s message=%s",
                    event.get("state"),
                    event.get("message"),
                )
                break

        logger.info("Final accumulated text: %s", "".join(chunks))


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    asyncio.run(main(url))
