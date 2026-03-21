"""Legal Document Writer Agent — Drafts NDAs, MOUs, notices."""
from app.services.agents.base import BaseAgent


class LegalDocumentWriterAgent(BaseAgent):
    identifier = "legal_document_writer"
