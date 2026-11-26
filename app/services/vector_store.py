"""Embeddings + FAISS index (cosine similarity, persisted to disk)."""
import hashlib
import logging
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from app.config import get_settings

log = logging.getLogger("docucite.vector")

STOPWORDS = set(
    "the a an of to in is are do does how what when where who which for can be i we you it on at "
    "by and or many much per with from must should that this was were there their".split()
)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. No downloads; used for tests/offline."""
    dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name)
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(s.embedding_model)


class VectorStore:
    def __init__(self, embedder: Embedder, path: Path):
        self.embedder = embedder
        self.path = Path(path)
        self._lock = threading.RLock()
        self.index = self._load_or_create()

    def _new_index(self):
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.embedder.dim))

    def _load_or_create(self):
        if self.path.exists():
            try:
                idx = faiss.read_index(str(self.path))
                if idx.d == self.embedder.dim:
                    return idx
                log.warning("Index dim %s != model dim %s; starting empty (run reindex).",
                            idx.d, self.embedder.dim)
            except Exception as e:  # corrupted file
                log.warning("Could not read index (%s); starting empty (run reindex).", e)
        return self._new_index()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp))
        tmp.replace(self.path)

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: list[int], texts: list[str]) -> None:
        if not ids:
            return
        vecs = self.embedder.embed(texts)
        with self._lock:
            self.index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
            self._save()

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))
            self._save()

    def reset(self) -> None:
        with self._lock:
            self.index = self._new_index()
            self._save()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            qv = self.embedder.embed([query])
            scores, ids = self.index.search(qv, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    s.ensure_dirs()
    return VectorStore(build_embedder(), s.faiss_path)
