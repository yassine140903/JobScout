"""Embedding wrapper for sentence-transformers (multilingual-e5-base)."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-base"

def to_blob(vector: np.ndarray) -> bytes:
    """Serialize a numpy vector to bytes for SQLite BLOB storage."""
    return vector.astype(np.float32).tobytes()

def from_blob(blob: bytes, dim: int = 768) -> np.ndarray:
    """Deserialize a SQLite BLOB back to a numpy vector."""
    return np.frombuffer(blob, dtype=np.float32)


class Embedder:
    """Lazy-loading embedder. Model downloads on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str, is_query: bool = False) -> np.ndarray:
        """Embed a single text. Returns a 1-D float32 array.

        e5 requires prefix: 'query: ' for queries, 'passage: ' for documents.
        Profile facets are queries, job facets are passages.
        """
        prefix = "query: " if is_query else "passage: "
        return self.model.encode(prefix + text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed multiple texts. Returns (N, dim) float32 array."""
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True)