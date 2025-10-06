from sentence_transformers import SentenceTransformer
import re

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
#        self.embedding_dimensions = 384 # really thought that would be useful

    async def generate_embedding(self, text:str) -> list[float]:
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            raise

    async def generate_embedding_batch(self, texts:list[str]) -> list[list[float]]:
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

def normalize_text(text: str) -> str:
    text = text.replace('\r\n', '\n')  # Windows
    text = text.replace('\r', '\n')     # Old Mac
    text = text.replace('\t', ' ')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = text.strip()
    
    return text

# Chunks are ideally equal sizes, but if words make that difficult, we chunk by words at the ends. "numb" and "numbers"
# have very different meanings, despite being almost the same word.
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    text = normalize_text(text)
    words = text.split()
    chunks = []

    avg_word_length = len(text)/len(words) if words else 5
    words_per_chunk = int(chunk_size // avg_word_length)
    overlap_words = int(overlap // avg_word_length)
    # So ultimately we guess how many words to a chunk based, and then add that many to a chunk.
    # It does not matter if we are off by a little.
    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk = ' '.join(words[i:i + words_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks

embedding_service = EmbeddingService()

