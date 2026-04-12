"""
Document Processor — Handles ingestion of Knowledge Base files (PDF, Docx, Text).

RAG-AI: Splits documents into semantically coherent chunks and 
prepares them for vector storage.
"""

import logging
from typing import List, Dict, Any
from io import BytesIO

# Potential libraries (will need pip install if not present, but 
# we'll use robust standard approach or stubs for now)
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service to extract text from various file formats and chunk it."""

    @classmethod
    def extract_text(cls, file_content: bytes, filename: str) -> str:
        """Extract plain text from a file based on its extension."""
        ext = filename.split(".")[-1].lower()
        
        if ext == "pdf":
            return cls._parse_pdf(file_content)
        elif ext in ["docx", "doc"]:
            return cls._parse_docx(file_content)
        elif ext in ["txt", "md", "csv"]:
            return file_content.decode("utf-8", errors="ignore")
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return ""

    @classmethod
    def chunk_text(cls, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for vector search."""
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        
        return chunks

    @classmethod
    def _parse_pdf(cls, content: bytes) -> str:
        if not PyPDF2:
            logger.error("PyPDF2 not installed. Cannot parse PDF.")
            return "[Error: PDF parser missing]"
        
        try:
            reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            return ""

    @classmethod
    def _parse_docx(cls, content: bytes) -> str:
        if not docx:
            logger.error("python-docx not installed. Cannot parse DOCX.")
            return "[Error: DOCX parser missing]"
        
        try:
            doc = docx.Document(BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
            return ""
