"""GST Compliance Agent — Indian GST compliance guidance."""
from app.services.agents.base import BaseAgent


class GstComplianceAgent(BaseAgent):
    identifier = "gst_compliance"
