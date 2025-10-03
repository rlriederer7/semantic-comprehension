from fastapi import APIRouter

router = APIRouter()

@router.post("/encode")
async def encode_text(text: str):
    return {"This is not a real embedding": text}
