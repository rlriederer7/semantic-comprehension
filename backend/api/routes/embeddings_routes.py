import uuid
from fastapi import APIRouter, HTTPException
from backend.models.document_models import DocumentUpload, SearchRequest, SearchResponse, SearchLLMRequest, SearchLLMResponse
from backend.services.database_service import db_service
from backend.services.llm_service import llm_service
from backend.services.embeddings_service import embedding_service, chunk_text

router = APIRouter()

@router.post("/upload")
async def upload_document(request: DocumentUpload):
    try:
        await db_service.create_index()
        document_id = str(uuid.uuid4())
        chunks = chunk_text(request.text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Empty Document")

        embeddings = await embedding_service.generate_embedding_batch(chunks)
        chunk_ids = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = await db_service.insert_chunk(
                document_id = document_id,
                document_name=request.document_name,
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

@router.post("/search_semantic_only", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        query_embedding = await embedding_service.generate_embedding(request.query)
        similar_chunks = await db_service.search_similar(
            query_embedding,
            limit=request.top_k
        )

        if not similar_chunks:
            raise HTTPException(
                status_code=404,
                detail="No documents found in database"
            )

        return SearchResponse(
            chunks = similar_chunks
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_with_llm", response_model=SearchLLMResponse)
async def search_documents_with_llm(request: SearchLLMRequest):
    try:
        query_embedding = await embedding_service.generate_embedding(request.query)
        similar_chunks = await db_service.search_similar(
            query_embedding,
            limit=request.top_k
        )

        similar_truncated_chunks = []

        for i in similar_chunks:
            similar_truncated_chunks.append((i[0],i[2]))

        if not similar_chunks:
            raise HTTPException(
                status_code=404,
                detail="No documents found in database"
            )

        llm_answer = await llm_service.generate(request.query, similar_truncated_chunks)
        #llm_answer = "asdf"
        print(llm_answer)

        return SearchLLMResponse(
            answer=llm_answer,
            chunks=similar_chunks,
            provider_used=request.provider
        )


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))