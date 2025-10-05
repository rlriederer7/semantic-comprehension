import uuid
from fastapi import APIRouter, HTTPException
from backend.models.document_models import DocumentUpload, SearchRequest, SearchResponse
from backend.services.database_service import db_service
from backend.services.embeddings_service import embedding_service, chunk_text

router = APIRouter()

@router.post("/upload")
async def upload_document(request: DocumentUpload):
    try:
        document_id = str(uuid.uuid4())
        chunks = chunk_text(request.text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Empty Document")

        embeddings = await embedding_service.generate_embedding_batch(chunks)
        chunk_ids = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = await db_service.insert_chunk(
                document_id = document_id,
                chunk_index = idx,
                text=chunk,
                embedding = embedding
            )
            chunk_ids.append(chunk_id)
        return {
            "document_id":document_id,
            "document_name":request.document_name,
            "chunks_created": len(chunks),
            "chunk_ids": chunk_ids,
        }

    except Exception as e:
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_semantic_only", response_model = SearchResponse)
async def search_documents(request: SearchRequest):
    print(1)
    try:
        print(2)
        query_embedding = await embedding_service.generate_embedding(request.query)
        print(3)
        similar_chunks = await db_service.search_similar(
            query_embedding,
            limit=request.top_k
        )

        if not similar_chunks:
            raise HTTPException(
                status_code=404,
                detail="No documents found in database"
            )

        return similar_chunks

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
