from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dimension = 384

    async def generate_embedding(self, text:str) -> List[float]:
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            raise

    async def generate_embedding_batch(self, texts:List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                batch_size = 32,
                show_progress_bar = True
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"Embedding error during batch embedding: {e}")
            raise

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []

    avg_word_length = len(text)/len(words) if words else 5
    words_per_chunk = int(chunk_size // avg_word_length)
    overlap_words = int(overlap // avg_word_length)

    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk = ' '.join(words[i:i + words_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks

embedding_service = EmbeddingService()

