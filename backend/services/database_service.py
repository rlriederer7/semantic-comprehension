import asyncpg
import os

class DatabaseService:
    def __init__(self):
        self.pool = None

    async def connect(self):
        database_url = os.getenv("DATABASE_URL")
        self.pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
        # For debugging/development only, drops index, document_chunks table, vectors on startup
        print(type(os.getenv("DEBUG")))
        if os.getenv("DEBUG") == 'True':
            await self.clear_on_startup()
        await self.init_db()

    async def clear_on_startup(self):
        async with self.pool.acquire() as conn:
            print("dropping")
            await conn.execute("""
                DROP INDEX IF EXISTS embedding_idx;
                DROP TABLE IF EXISTS document_chunks;
                DROP EXTENSION IF EXISTS vector;
            """)
            print("dropped")

    async def init_db(self):
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    document_name VARCHAR(255) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def create_index(self):
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
            # indexes at 40 for debugging purposes, in reality it should wait to index at closer to 1k chunks, a point where
            # data retrieval starts being meaningfully slow. want to wait as long as possible so indexing is based off of
            # as much data as possible
            if os.getenv("DEBUG") == 'True':
                enough_chunks_to_index = 40
            else:
                enough_chunks_to_index = 1000
            if int(count) > enough_chunks_to_index:
                print("checking for index")
                index_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes WHERE indexname = 'embedding_idx'
                    )
                """)
                if not index_exists:
                    print("indexing")
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS embedding_idx
                        ON document_chunks
--                        USING ivfflat (embedding vector_cosine_ops)
                        USING hnsw (embedding vector_cosine_ops)
--                        WITH (lists=100)
                        WITH (m=16, ef_construction=64)
                    """)

    async def insert_chunk(self, document_id: str, document_name: str, chunk_index: int, text: str, embedding: list[float]) -> int:
        async with self.pool.acquire() as conn:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            row = await conn.fetchrow("""
                INSERT INTO document_chunks (document_id, document_name, chunk_index, text, embedding)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, document_id, document_name, chunk_index, text, embedding_str)
            return row['id']

    # Searches similarity by cosine similarity because we don't care if the query and the chunk are similar or different
    # lengths, just their semantic orientation
    async def search_similar(self, query_embedding: list[float], limit: int = 30) -> list[tuple[str, float, str]]:
        async with self.pool.acquire() as conn:
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            rows = await conn.fetch("""
                SELECT text, embedding <=> $1 AS distance, document_name
                FROM document_chunks
                ORDER BY distance, id
                LIMIT $2
            """, embedding_str, limit)

            return [(row['text'], row['distance'], row['document_name']) for row in rows]

    async def close(self):
        if self.pool:
            await self.pool.close()

db_service = DatabaseService()