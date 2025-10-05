import asyncpg
import os
from typing import List, Tuple

class DatabaseService:
    def __init__(self):
        self.pool = None

    async def connect(self):
        database_url = os.getenv("DATABASE_URL")
        self.pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        await self.init_db()

    async def init_db(self):
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx
                ON document_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists=100)
            
            """)

    async def insert_chunk(self, document_id: str, chunk_index: int, text: str, embedding: List[float]) -> int:
        async with self.pool.acquire() as conn:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            row = await conn.fetchrow("""
                INSERT INTO document_chunks (document_id, chunk_index, text, embedding)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, document_id, chunk_index, text, embedding_str)
            return row['id']

    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[str, float]]:
        async with self.pool.acquire() as conn:
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            rows = await conn.fetch("""
                SELECT text, embedding <=> $1 AS distance
                FROM document_chunks
                ORDER BY distance
                LIMIT $2
            """, embedding_str, limit)

            return [(row['text'], row['distance']) for row in rows]

    async def close(self):
        if self.pool:
            await self.pool.close()

db_service = DatabaseService()