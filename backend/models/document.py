from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentChunk(BaseModel):
    document_id: str
    chunk_index: int
    text: str

class DocumentUpload(BaseModel):
    text: str
    document_name: str

