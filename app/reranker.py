import sys
from typing import List, Tuple

_reranker_model = None

class HuggingFaceReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

    def _get_model(self):
        global _reranker_model
        if _reranker_model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[Reranker] Loading Hugging Face Cross-Encoder model: '{self.model_name}'...")
                _reranker_model = CrossEncoder(self.model_name)
                print("[Reranker] Hugging Face Cross-Encoder loaded successfully.")
            except Exception as e:
                print(f"[Reranker Notice] Could not load Hugging Face CrossEncoder ({e}). Using similarity pass-through fallback.")
                _reranker_model = "FALLBACK"
        return _reranker_model

    def rerank(
        self, query: str, chunks: List[Tuple[str, str, float]], top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Reranks a list of candidate chunks (chunk_text, filename, similarity) using
        a Hugging Face Cross-Encoder model.

        Returns top_k reranked chunks with updated Cross-Encoder scores.
        """
        if not chunks:
            return []

        model = self._get_model()
        if model == "FALLBACK" or model is None:
            # Fallback: Sort by initial vector similarity score
            sorted_chunks = sorted(chunks, key=lambda x: x[2], reverse=True)
            return sorted_chunks[:top_k]

        try:
            pairs = [[query, chunk_text] for chunk_text, _, _ in chunks]
            scores = model.predict(pairs)

            reranked = []
            for (chunk_text, filename, _), score in zip(chunks, scores):
                reranked.append((chunk_text, filename, float(score)))

            reranked.sort(key=lambda x: x[2], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            print(f"[Reranker Exception] Error during reranking ({e}). Returning initial vector rankings.")
            sorted_chunks = sorted(chunks, key=lambda x: x[2], reverse=True)
            return sorted_chunks[:top_k]


# Global instance
reranker = HuggingFaceReranker()
