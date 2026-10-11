from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestPDFData(BaseModel):
    pdf_path: str = Field(..., description="Absolute path to the PDF file on disk")
    source_id: str = Field(..., description="Unique identifier for the ingested document source")
    visibility: str = Field(default="public", description="public, student, faculty, or admin")
    department: Optional[str] = None
    replace_existing: bool = Field(default=True, description="Replace existing chunks for this source before ingesting")


class IngestResult(BaseModel):
    status: str
    source_id: str
    chunks_ingested: int
    visibility: str = "public"
    department: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question asked by the user")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    source_id: Optional[str] = Field(default=None, description="Optional source ID to restrict retrieval to one document")


class CitationItem(BaseModel):
    index: int
    source_id: str
    page_number: int = 1
    ocr_used: bool = False
    text: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int
    citations: List[CitationItem] = []


class ExportRequest(BaseModel):
    question: str
    answer: str
    citations: List[Dict[str, Any]] = []
    source_id: Optional[str] = None


class DocumentChunk(BaseModel):
    text: str
    source_id: str
    page_number: int = 1
    ocr_used: bool = False
    visibility: str = "public"
    owner_id: Optional[str] = None
    department: Optional[str] = None
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None
