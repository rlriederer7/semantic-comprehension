import os
from fastapi import HTTPException, APIRouter
from models.document_models import SearchRequest, SearchResponse, SearchLLMResponse, SearchLLMRequest
from services.database_service import db_service
from services.embeddings_service import embedding_service
from services.llm_service import llm_service
if os.getenv("RERANK_TOGGLE") == 'True':
    from services.reranker_service import reranker_service

router = APIRouter()

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
        # If reranking, pull 4x as many chunks that might be relevant b/c reranking makes it cheap
        query_embedding = await embedding_service.generate_embedding(request.query)
        if os.getenv("RERANK_TOGGLE") == 'True':
            reranker_multiplier = 4
        else:
            reranker_multiplier = 1
        similar_chunks = await db_service.search_similar(
            query_embedding,
            limit=request.top_k*reranker_multiplier
        )

        similar_truncated_chunks = []

        for i in similar_chunks:
            similar_truncated_chunks.append((i[0],i[2]))

        if not similar_chunks:
            raise HTTPException(
                status_code=404,
                detail="No documents found in database"
            )
        # If reranking, send rerankable information to the reranker, then to the LLM
        if os.getenv("RERANK_TOGGLE") == 'True':
            reranked_chunks = reranker_service.rerank_chunks(
                request.query,
                similar_truncated_chunks,
                request.top_k
            )

            chunk_texts = [(text, doc_name) for text, doc_name, _ in reranked_chunks]
            chunks_with_scores = reranked_chunks

            llm_answer = await llm_service.generate(request.query, chunk_texts)
        # Otherwise, just send it to the LLm and then get it the order the response expects.
        else:
            llm_answer = await llm_service.generate(request.query, similar_truncated_chunks)
            chunks_with_scores = [(text, doc_name, float(score)) for text, score, doc_name in similar_chunks]
        #llm_answer = "asdf"

        print(llm_answer)
        print(chunks_with_scores)
        print(request.provider)

        return SearchLLMResponse(
            answer=llm_answer,
            chunks=chunks_with_scores,
            provider_used=request.provider
        )


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))