import numpy as np
from fastembed import TextEmbedding


class VectorMemory:
    """
    Session-scoped vector memory using cosine similarity over numpy vectors.

    This stays request-local (no persistence across queries) to avoid
    cross-topic contamination and to keep memory lifetime short.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSION = 384

    def __init__(self):
        self.model = TextEmbedding(model_name=self.MODEL_NAME)
        self._vecs: list[np.ndarray] = []
        self.memory: list[dict] = []
        self.next_id = 0

    def _embed(self, text: str) -> np.ndarray:
        """Embed one text and return a normalised 1D vector."""
        emb = np.array(list(self.model.embed([text])), dtype="float32")[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _is_duplicate(self, emb: np.ndarray, threshold: float = 0.92) -> bool:
        """Duplicate check against stored vectors via cosine similarity."""
        if not self._vecs:
            return False
        mat = np.stack(self._vecs)
        scores = mat @ emb
        return float(scores.max()) > threshold

    def add_chunks(self, url: str, chunks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """
        Add pre-chunked text to memory.
        Duplicates (cosine similarity > 0.92) are skipped.
        """
        stored = []
        for _chunk_id, chunk_text in chunks:
            emb = self._embed(chunk_text)
            if self._is_duplicate(emb):
                continue
            self._vecs.append(emb)
            self.memory.append({"id": self.next_id, "url": url, "chunk": chunk_text})
            stored.append((self.next_id, chunk_text))
            self.next_id += 1
        return stored

    def add_chunks_batch(self, entries: list[tuple[str, int, str]]) -> int:
        """
        Compatibility helper for existing callers.
        Accepts [(url, chunk_id, text), ...], stores via add_chunks().
        """
        by_url: dict[str, list[tuple[int, str]]] = {}
        for url, chunk_id, text in entries:
            by_url.setdefault(url, []).append((chunk_id, text))
        stored = 0
        for url, chunks in by_url.items():
            stored += len(self.add_chunks(url, chunks))
        return stored

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Semantic search → [{score, url, chunk}] sorted by similarity."""
        if not self._vecs:
            return []
        q = self._embed(query)
        mat = np.stack(self._vecs)
        scores = mat @ q

        k_actual = min(k, len(scores))
        top_idx = np.argpartition(scores, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            m = self.memory[int(idx)]
            results.append({"score": float(scores[idx]), "url": m["url"], "chunk": m["chunk"]})
        return results

    def size(self) -> int:
        return len(self._vecs)
