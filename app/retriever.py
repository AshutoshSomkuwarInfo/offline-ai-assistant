from __future__ import annotations

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional

from app.intent import QueryIntent


def _read_indexable_lines(path: str) -> list[str]:
    """Non-empty lines for embedding; skip # comments (add notes without indexing)."""
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


class ContextRetrievalEngine:
    """
    FAISS-backed retrieval over optional local text (general + personal).

    Offline general Q&A does not require a general corpus: the on-device SLM answers
    from its weights. Indexed ``general`` lines are optional RAG (notes, book excerpts).
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.general_docs: list[str] = []
        self.personal_docs: list[str] = []
        self._general_index: Optional[faiss.IndexFlatL2] = None
        self._personal_index: Optional[faiss.IndexFlatL2] = None

    def load_general_knowledge(self, file_path: str) -> None:
        self.load_general_sources([file_path])

    def load_general_sources(self, file_paths: list[str]) -> None:
        docs: list[str] = []
        for p in file_paths:
            if not p or not os.path.isfile(p):
                continue
            docs.extend(_read_indexable_lines(p))
        self.general_docs = docs
        if not docs:
            self._general_index = None
            return
        emb = np.array(self.model.encode(self.general_docs))
        self._general_index = _build_index(emb)

    def load_personal_context(self, file_path: str) -> None:
        self.load_personal_sources([file_path])

    def load_personal_sources(self, file_paths: list[str]) -> None:
        docs: list[str] = []
        for p in file_paths:
            if not p or not os.path.isfile(p):
                continue
            docs.extend(_read_indexable_lines(p))
        if not docs:
            docs = [
                "(No personal data yet. Add data/personal_context.txt and/or run "
                "python -m app.google_sync to index Gmail, Calendar, and Photos metadata.)"
            ]
        self.personal_docs = docs
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
