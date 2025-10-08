from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import embeddings_routes
from backend.services.database_service import db_service
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_service.connect()
    yield
    await db_service.close()

app = FastAPI(title='Semantic Search', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Routers
app.include_router(embeddings_routes.router, prefix="/embeddings")

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