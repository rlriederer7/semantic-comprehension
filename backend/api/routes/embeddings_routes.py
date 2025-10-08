import os
import uuid
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
import PyPDF2
import io
from backend.models.document_models import DocumentUpload, SearchRequest, SearchResponse, SearchLLMRequest, SearchLLMResponse
from backend.services.database_service import db_service
from backend.services.llm_service import llm_service
if os.getenv("RERANK_TOGGLE") == 'True':
    from backend.services.reranker_service import reranker_service
from backend.services.embeddings_service import embedding_service, chunk_text

router = APIRouter()

async def process_and_store_document(text: str, document_name: str):
        await db_service.create_index()
        document_id = str(uuid.uuid4())
        chunks = chunk_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Empty Document")

        embeddings = await embedding_service.generate_embedding_batch(chunks)
        chunk_ids = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = await db_service.insert_chunk(
                document_id = document_id,
                document_name=document_name,
                chunk_index = idx,
                text=chunk,
                embedding = embedding
            )
            chunk_ids.append(chunk_id)
        return {
            "document_id":document_id,
            "document_name":document_name,
            "chunks_created": len(chunks),
            "chunk_ids": chunk_ids,
        }

def extract_pdf_text(pdf_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "/n"

        return text.strip()

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text from PDF: {str(e)}"
        )

@router.post("/upload_file")
async def upload_file(file: UploadFile = File(...), document_name: str = Form(None)):
    try:
        if not document_name:
            document_name = file.filename

        content = await file.read()
        max_file_size = int(os.getenv("MAX_FILE_SIZE"))
        if len(content) > max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large :(, max file size is {max_file_size}"
            )

        if not document_name:
            document_name = file.filename

        if file.filename.endswith('.pdf'):
            text=extract_pdf_text(content)
        elif file.filename.endswith('.txt'):
            text=content.decode('utf-8')
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type"
            )

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Empty document or failed to extract text"
            )

        return await process_and_store_document(
            text=text,
            document_name=document_name
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(request: DocumentUpload):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty Document")

        return await process_and_store_document(
            text = request.text,
            document_name=request.document_name
        )

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