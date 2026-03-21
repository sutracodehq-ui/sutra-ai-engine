"""Itinerary Generator Agent — Auto-generates itineraries."""
from app.services.agents.base import BaseAgent


class ItineraryGeneratorAgent(BaseAgent):
    identifier = "itinerary_generator"
