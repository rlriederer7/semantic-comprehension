import os
if os.getenv("RERANK_TOGGLE") == 'True':
    import os
    from sentence_transformers import CrossEncoder

class RerankerService:
    def __init__(self, model_name: str = os.getenv("RERANKER")):
        self.model = CrossEncoder(model_name)

    def rerank_chunks(self, query: str, chunks: list[tuple[str, str]], top_k=5) -> list[tuple[str, str, float]]:
        if not chunks:
            return []
        # Most of this is just list/tuple manipulation to make sure that when a list of tuples of two strings goes in
        # a correctly structured list of tuples of two strings and the similarity of the first string to a query goes out
        chunk_data = [(text, doc_name) for text, doc_name in chunks]
        chunk_texts = [text for text, _ in chunk_data]
        pairs = [[query, chunk] for chunk in chunk_texts]
        # This is the only part that does prediction, per se.
        scores = self.model.predict(pairs)
        # And this does the reranking.
        reranked = [(text, doc_name, float(score)) for (text, doc_name), score in zip(chunk_data, scores)]
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked[:top_k]

reranker_service = RerankerService()