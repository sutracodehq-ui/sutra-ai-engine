"""EventTimelineAgent — wedding_events domain agent."""
from app.services.agents.base import BaseAgent


class EventTimelineAgent(BaseAgent):
    identifier = "event_timeline"
