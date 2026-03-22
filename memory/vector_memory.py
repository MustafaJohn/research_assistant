import numpy as np
import faiss
from fastembed import TextEmbedding


class VectorMemory:
    """
    Session-scoped vector memory using FAISS IndexFlatIP (cosine similarity).
    Uses batch embedding — all chunks embedded in one model inference call
    instead of one call per chunk. Brings embedding time from ~8-10s to ~1-2s.
    No disk persistence — fresh instance per request.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        self.model    = TextEmbedding(model_name=self.MODEL_NAME)
        self.dimension = 384
        self.index    = faiss.IndexFlatIP(self.dimension)
        self.memory:  list[dict] = []   # [{id, url, chunk}]
        self.next_id  = 0

    # ─────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts in one model inference call."""
        embs = np.array(list(self.model.embed(texts)), dtype="float32")
        faiss.normalize_L2(embs)
        return embs

    def _embed_one(self, text: str) -> np.ndarray:
        """Embed a single text — used for search queries only."""
        emb = np.array(list(self.model.embed([text])), dtype="float32")
        faiss.normalize_L2(emb)
        return emb

    def _is_duplicate(self, emb: np.ndarray, threshold: float = 0.92) -> bool:
        if self.index.ntotal == 0:
            return False
        scores, _ = self.index.search(emb, 1)
        return float(scores[0][0]) > threshold

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def add_chunks_batch(self, entries: list[tuple[str, int, str]]) -> int:
        """
        Add multiple chunks in one batch embedding call.

        Args:
            entries: list of (url, chunk_id, chunk_text)

        Returns:
            Number of chunks actually stored (duplicates skipped)
        """
        if not entries:
            return 0

        texts = [e[2] for e in entries]
        embs  = self._embed_batch(texts)

        stored = 0
        for (url, _chunk_id, chunk_text), emb in zip(entries, embs):
            emb_2d = emb.reshape(1, -1)
            if self._is_duplicate(emb_2d):
                continue
            self.index.add(emb_2d)
            self.memory.append({"id": self.next_id, "url": url, "chunk": chunk_text})
            self.next_id += 1
            stored += 1

        return stored

    def add_chunks(self, url: str, chunks: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """
        Single-document add — kept for compatibility.
        Internally uses batch embedding.
        """
        entries = [(url, cid, text) for cid, text in chunks]
        self.add_chunks_batch(entries)
        return [(i, t) for _, i, t in entries]

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Semantic search — returns [{score, url, chunk}]."""
        if self.index.ntotal == 0:
            return []
        emb      = self._embed_one(query)
        k_actual = min(k, self.index.ntotal)
        scores, ids = self.index.search(emb, k_actual)
        results  = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.memory):
                continue
            m = self.memory[idx]
            results.append({"score": float(score), "url": m["url"], "chunk": m["chunk"]})
        return results

    def size(self) -> int:
        return self.index.ntotal
