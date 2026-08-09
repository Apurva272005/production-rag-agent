from typing import List, Optional
from pydantic import BaseModel, Field


class IngestPDFData(BaseModel):
    pdf_path: str = Field(..., description="Absolute path to the temporary PDF file on disk")
    source_id: str = Field(..., description="Unique identifier for the ingested document source")
    replace_existing: bool = Field(default=True, description="Replace existing chunks for this source before ingesting")


class IngestResult(BaseModel):
    status: str
    source_id: str
    chunks_ingested: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question asked by the user")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    source_id: Optional[str] = Field(default=None, description="Optional source ID to restrict retrieval to one document")


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int


class DocumentChunk(BaseModel):
    text: str
    source_id: str
    embedding: Optional[List[float]] = None
