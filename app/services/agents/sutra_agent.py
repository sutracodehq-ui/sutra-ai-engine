"""
SutraAgent — Universal AI agent that dynamically loads any skill from YAML.

Instead of 187 separate Python classes (all identical wrappers), this ONE class
handles every skill by loading the appropriate YAML config on demand.

Usage:
    agent = SutraAgent("quiz_generator")   # Loads agent_config/quiz_generator.yaml
    agent = SutraAgent("tax_planner")      # Loads agent_config/tax_planner.yaml
    agent = SutraAgent("brand_analyzer")   # Loads agent_config/brand_analyzer.yaml

Adding a new skill = just create a new YAML file. Zero code changes.
"""

from app.services.agents.base import BaseAgent


class SutraAgent(BaseAgent):
    """
    Universal agent that loads any skill from YAML config.

    This replaces 187 identical wrapper classes with a single dynamic class.
    The skill_id maps directly to a YAML file in agent_config/.
    """

    def __init__(self, skill_id: str, llm=None):
        self._skill_id = skill_id
        super().__init__(llm=llm)

    @property
    def identifier(self) -> str:
        return self._skill_id
