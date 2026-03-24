import numpy as np
from fastembed import TextEmbedding


# ─────────────────────────────────────────────────────────────
# Global singleton — loaded once at startup, shared across all
# requests. TextEmbedding + ONNX Runtime is stateless and safe
# to share. ~150MB, loaded once, never reloaded.
# ─────────────────────────────────────────────────────────────
_EMBEDDING_MODEL: TextEmbedding | None = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_model() -> TextEmbedding:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = TextEmbedding(model_name=_MODEL_NAME)
    return _EMBEDDING_MODEL


class VectorMemory:
    """
    Session-scoped vector store using pure numpy cosine similarity.

    Replaces FAISS IndexFlatIP with a plain numpy matrix.
    Same mathematical result (normalised dot product = cosine similarity)
    but:
      - No FAISS binary in memory (~50MB saved)
      - Index is a numpy array that gets garbage collected after each request
      - No memory leak between requests

    The embedding model is shared via get_model() singleton — loaded once
    at startup, never reloaded per request.
    """

    DIMENSION = 384

    def __init__(self):
        self.model   = get_model()          # shared singleton — no extra RAM
        self._vecs:  list[np.ndarray] = []  # list of normalised 1D vectors
        self.memory: list[dict]       = []  # [{id, url, chunk}]
        self.next_id = 0

    # ─────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts in one model inference call. Returns (N, 384)."""
        embs = np.array(list(self.model.embed(texts)), dtype="float32")
        # L2 normalise each row so dot product == cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)   # avoid div-by-zero
        return embs / norms

    def _embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns normalised 1D vector (384,)."""
        emb = np.array(list(self.model.embed([text])), dtype="float32")[0]
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _cosine_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Dot product between query_vec (384,) and all stored vectors (N, 384).
        Returns scores array of shape (N,).
        """
        if not self._vecs:
            return np.array([])
        mat = np.stack(self._vecs)          # (N, 384)
        return mat @ query_vec              # (N,)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def add_chunks_batch(self, entries: list[tuple[str, int, str]]) -> int:
        """
        Batch-embed and store chunks.
        Duplicate check runs against the pre-existing corpus only,
        building the matrix once for the whole batch — O(n) not O(n²).
        """
        if not entries:
            return 0

        texts = [e[2] for e in entries]
        embs  = self._embed_batch(texts)    # one model inference call

        # Build existing corpus matrix once — before processing any new chunk
        existing_mat = np.stack(self._vecs) if self._vecs else None  # (N, 384) or None

        stored = 0
        for (url, _chunk_id, chunk_text), emb in zip(entries, embs):
            # Check against existing corpus only (not other chunks in this batch)
            if existing_mat is not None:
                scores = existing_mat @ emb     # (N,) — no matrix rebuild
                if scores.max() > 0.92:
                    continue
            self._vecs.append(emb)
            self.memory.append({"id": self.next_id, "url": url, "chunk": chunk_text})
            self.next_id += 1
            stored += 1

        return stored

    def add_chunks(self, url: str, chunks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Single-document add — uses batch embedding internally."""
        entries = [(url, cid, text) for cid, text in chunks]
        self.add_chunks_batch(entries)
        return [(i, t) for _, i, t in entries]

    def search(self, query: str, k: int = 10) -> list[dict]:
        """
        Return top-k most similar chunks to query.
        Returns list of {score, url, chunk} sorted descending by score.
        """
        if not self._vecs:
            return []

        query_vec = self._embed_one(query)
        scores    = self._cosine_scores(query_vec)     # (N,)

        k_actual  = min(k, len(scores))
        top_idx   = np.argpartition(scores, -k_actual)[-k_actual:]
        top_idx   = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort desc

        results = []
        for idx in top_idx:
            m = self.memory[idx]
            results.append({
                "score": float(scores[idx]),
                "url":   m["url"],
                "chunk": m["chunk"],
            })
        return results

    def size(self) -> int:
        return len(self._vecs)
