"""
RAG API Router — Knowledge Base management.

RAG-AI: Endpoints to upload files, index them into vector space, 
and manage the tenant's private collection.
"""

from typing import Annotated, List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import DbSession, get_current_tenant
from app.models.tenant import Tenant
from app.services.rag.document_processor import DocumentProcessor
from app.services.rag.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/upload")
async def upload_document(
    tenant: Annotated[Tenant, Depends(get_current_tenant)],
    file: UploadFile = File(...),
):
    """Upload a file and index its content into the tenant's vector store."""
    content = await file.read()
    
    # 1. Extract Text
    text = DocumentProcessor.extract_text(content, file.filename)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    # 2. Chunk Text
    chunks = DocumentProcessor.chunk_text(text)
    
    # 3. Index into Knowledge Base
    kb = KnowledgeBaseService()
    
    # Prepare IDs and Metadata
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": file.filename, "chunk": i} for i in range(len(chunks))]
    
    await kb.add_documents(tenant.id, chunks, metadatas, ids)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": len(chunks)
    }


@router.post("/query")
async def query_knowledge_base(
    tenant: Annotated[Tenant, Depends(get_current_tenant)],
    query: str,
    n_results: int = 3
):
    """Directly query the knowledge base for testing or manual search."""
    kb = KnowledgeBaseService()
    results = await kb.query(tenant.id, query, n_results=n_results)
    return {"results": results}
