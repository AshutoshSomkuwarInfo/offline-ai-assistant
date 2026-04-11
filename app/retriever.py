from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional

from app.intent import QueryIntent


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


class ContextRetrievalEngine:
    """
    FAISS-backed retrieval over a general-knowledge corpus and a personal-context corpus
    (emails, files, calendar-style lines — as chunked text lines in this prototype).
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.general_docs: list[str] = []
        self.personal_docs: list[str] = []
        self._general_index: Optional[faiss.IndexFlatL2] = None
        self._personal_index: Optional[faiss.IndexFlatL2] = None

    def load_general_knowledge(self, file_path: str) -> None:
        self.general_docs = _read_lines(file_path)
        if not self.general_docs:
            raise ValueError(f"No documents in general knowledge file: {file_path}")
        emb = np.array(self.model.encode(self.general_docs))
        self._general_index = _build_index(emb)

    def load_personal_context(self, file_path: str) -> None:
        self.personal_docs = _read_lines(file_path)
        if not self.personal_docs:
            raise ValueError(f"No documents in personal context file: {file_path}")
        emb = np.array(self.model.encode(self.personal_docs))
        self._personal_index = _build_index(emb)

    def _search_index(
        self,
        query: str,
        docs: list[str],
        index: Optional[faiss.IndexFlatL2],
        k: int,
    ) -> list[str]:
        if not docs or index is None or index.ntotal == 0:
            return []
        k = min(k, index.ntotal)
        qe = np.array(self.model.encode([query]))
        _, indices = index.search(qe.astype(np.float32), k)
        return [docs[i] for i in indices[0] if 0 <= i < len(docs)]

    def retrieve(self, query: str, intent: QueryIntent, k: int = 3) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"general_knowledge": [], "personal_context": []}

        if intent in (QueryIntent.GENERAL_KNOWLEDGE, QueryIntent.HYBRID):
            out["general_knowledge"] = self._search_index(
                query, self.general_docs, self._general_index, k
            )

        if intent in (QueryIntent.PERSONAL_CONTEXT, QueryIntent.HYBRID):
            out["personal_context"] = self._search_index(
                query, self.personal_docs, self._personal_index, k
            )

        return out
