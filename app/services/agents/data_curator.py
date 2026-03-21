"""Data Curator Agent — Dataset cleaning and quality assurance."""
from app.services.agents.base import BaseAgent


class DataCuratorAgent(BaseAgent):
    identifier = "data_curator"
