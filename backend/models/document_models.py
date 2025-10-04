from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentChunk(BaseModel):
    id: Optional[int] = None
    document_id: str
    chunk_index: int
    text: str
    embedding: Optional[list[float]] = None
    created: Optional[datetime] = None

class DocumentUpload(BaseModel):
    text: str
    document_name: str

class SearchRequest(BaseModel):
    query: str
    provider: str = "openai" #does nothing for now
    top_k: int = 100

class SearchResponse(BaseModel):
    chunks: list[tuple[str, float]]

