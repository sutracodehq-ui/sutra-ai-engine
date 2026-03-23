"""DocumentOcrExtractorAgent — Indian educational document data extraction agent."""
from app.services.agents.base import BaseAgent


class DocumentOcrExtractorAgent(BaseAgent):
    identifier = "document_ocr_extractor"
