from fastapi import FastAPI

app = FastAPI(title='Semantic Search')

@app.get("/")
async def root():
    return {"message": "running!"}

@app.get("/health")
async def health():
    return {"status": "healthy enough for this"}