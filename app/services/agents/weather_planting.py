"""Weather Planting Agent — Weather-based planting advice."""
from app.services.agents.base import BaseAgent


class WeatherPlantingAgent(BaseAgent):
    identifier = "weather_planting"
