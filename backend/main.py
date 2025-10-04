from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.api.routes import embeddings
from backend.services.database import db_service
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_service.connect()
    yield
    await db_service.close()

app = FastAPI(title='Semantic Search', lifespan=lifespan)

#Routers
app.include_router(embeddings.router, prefix="/embeddings")

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