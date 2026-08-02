"""Submit a long-running A2A task; poll for completion via the typed token.

The token is a frozen, JSON-serialisable dataclass — persist it to a
queue / database / cache and resume from any process. This example
just polls in-process every second.

Usage::

    pip install 'troopai-adk-python[a2a]'
    python examples/a2a/client_background.py [URL]
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import dataclasses
import json
import logging
import sys

from troopai.adk.a2a import A2AAgent, A2AContinuationToken, A2ARunner, A2ATaskError

logger = logging.getLogger(__name__)


async def main(remote_url: str) -> None:
    async with A2AAgent(name="LongJob", url=remote_url) as remote:
        token = await A2ARunner.arun(
            remote,
            "Summarise the last 500 papers on retrieval augmentation.",
            background=True,
        )
        logger.info("Submitted; got token: %s", token)

        # Demonstrate JSON round-trip — the token survives serialisation.
        payload = json.dumps(dataclasses.asdict(token))
        logger.info("Token JSON payload (persistable): %s", payload)
        restored = A2AContinuationToken(**json.loads(payload))
        assert restored == token

        # Poll until terminal — bounded by 5 minutes for the example.
        for _ in range(300):
            status = await A2ARunner.poll_task(remote, restored)
            logger.info("status: %s", status.state)
            if status.state == "completed":
                logger.info("Result:\n%s", status.result)
                return
            if status.state in ("failed", "rejected", "cancelled"):
                raise A2ATaskError(
                    task_id=status.task_id,
                    context_id=status.context_id,
                    state=status.state,
                    remote_message=status.message or "",
                )
            await asyncio.sleep(1.0)

        logger.warning("Polling timed out after 5 minutes; cancelling.")
        await A2ARunner.cancel_task(remote, restored)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    asyncio.run(main(url))
