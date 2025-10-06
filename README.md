To use, you need a .env file in /semantic-comprehension/.
Supply your own ANTHROPIC_API_KEY, the default variables below work for everything else.
The app is not compatible with OpenAI yet.
DEBUG mode means that Docker drops tables, vectors, and indeces on startup. Set it to anything other than True to avoid that.
RERANK_TOGGLE makes startup take much longer, and in my experience has not been particularly worth the slowdown.

DEBUG=True

POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=semantic_db

DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}

OPENAI_API_KEY=

ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514

RERANKER=BAAI/bge-reranker-v2-m3
RERANK_TOGGLE=False
