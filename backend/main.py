from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.api.routes import embeddings
from sentence_transformers import SentenceTransformer

#Grab model on startup to prevent 2-3 seconds hang time the first time a request is made
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    yield
    del app.state.embedding_model

app = FastAPI(title='Semantic Search', lifespan=lifespan)

#Routers
app.include_router(embeddings.router, prefix="/backend/embeddings")

#Health checks
@app.get("/")
async def root():
    return {"message": "running!"}

@app.get("/health")
async def health():
    return {"status": "healthy enough for this"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )