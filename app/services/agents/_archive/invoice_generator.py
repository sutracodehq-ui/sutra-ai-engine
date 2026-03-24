"""Invoice Generator Agent — GST-compliant Indian invoices."""
from app.services.agents.base import BaseAgent


class InvoiceGeneratorAgent(BaseAgent):
    identifier = "invoice_generator"
