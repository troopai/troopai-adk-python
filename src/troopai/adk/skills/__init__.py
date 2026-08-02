"""Skills module — reusable capability packages for agents.

Skills combine domain instructions, tools, resources, and governance
into composable units that can be attached to agents.
"""

from troopai.adk.skills.activation import SkillActivation
from troopai.adk.skills.discovery import SkillDiscoveryToolset
from troopai.adk.skills.skill import Skill, SkillGovernance, SkillMetadata
from troopai.adk.skills.skill_prompt import (
    RECOMMENDED_SKILL_INSTRUCTIONS,
    prompt_with_skill_instructions,
)
from troopai.adk.skills.skill_set import SkillSet

__all__ = [
    "RECOMMENDED_SKILL_INSTRUCTIONS",
    "Skill",
    "SkillActivation",
    "SkillDiscoveryToolset",
    "SkillGovernance",
    "SkillMetadata",
    "SkillSet",
    "prompt_with_skill_instructions",
]
