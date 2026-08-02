"""Minimal A2A client: call a remote agent, print the result.

Usage::

    pip install 'troopai-adk-python[a2a]'
    python examples/a2a/client_basic.py [URL]

If ``URL`` is omitted, defaults to ``http://localhost:8080`` (start
``server_basic.py`` in another terminal first).
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

from troopai.adk.a2a import A2AAgent, A2ARunner, A2ATaskError, A2ATransportError

logger = logging.getLogger(__name__)


async def main(remote_url: str) -> None:
    async with A2AAgent(name="RemoteAgent", url=remote_url) as remote:
        try:
            result = await A2ARunner.arun(remote, "Say hello in three different languages.")
        except A2ATransportError as exc:
            logger.error("Network error talking to %s: %s", remote_url, exc)
            return
        except A2ATaskError as exc:
            logger.error(
                "Remote task %s ended in state %s: %s",
                exc.task_id,
                exc.state,
                exc.remote_message,
            )
            return

        logger.info("Remote response: %s", result.text)
        logger.info("task_id=%s context_id=%s", result.task_id, result.context_id)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    asyncio.run(main(url))
