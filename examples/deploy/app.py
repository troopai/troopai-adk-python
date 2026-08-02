"""A deployable agent for the ``troopai deploy`` walkthrough (see README.md).

Exposes ``agent`` so a container can serve it with
``troopai serve --agent app:agent``. There is no ``__main__`` guard on
purpose: the deployment workflow runs this module through the CLI / the
generated image, not by executing the file directly.
"""

from troopai.adk.agents.agent import Agent

agent = Agent(
    name="support",
    system_prompt="You are a concise, helpful support assistant.",
)
