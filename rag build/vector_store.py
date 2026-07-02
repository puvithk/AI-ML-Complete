"""
Per-section, in-memory vector store.

Each section owns its own TF-IDF space + chunk matrix, so a query can be
scored against every section independently and in parallel with zero
cross-section interference. TF-IDF is used instead of a neural embedder
so there is no model download/load latency -- fits/searches are
millisecond-scale, which matters for the "run all sections in parallel
with low latency" requirement.
"""
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdf_processor import Section, Chunk


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


class SectionIndex:
    """A single section's searchable chunk index."""

    def __init__(self, section: Section):
        self.section = section
        self.chunks: List[Chunk] = section.chunks
        texts = [c.text for c in self.chunks]
        # Small, fast vectorizer per section -- vocab stays tiny since it's
        # scoped to one section only.
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            RetrievedChunk(chunk_id=self.chunks[i].chunk_id, text=self.chunks[i].text, score=float(sims[i]))
            for i in top_idx
            if sims[i] > 0
        ]


class DocumentStore:
    """Holds all sections/indices for uploaded documents, keyed by doc_id."""

    def __init__(self):
        self._docs: Dict[str, Dict[str, SectionIndex]] = {}
        self._sections_meta: Dict[str, List[Section]] = {}

    def add_document(self, doc_id: str, sections: List[Section]):
        self._docs[doc_id] = {s.section_id: SectionIndex(s) for s in sections}
        self._sections_meta[doc_id] = sections

    def get_sections(self, doc_id: str) -> List[Section]:
        return self._sections_meta.get(doc_id, [])

    def get_section_index(self, doc_id: str, section_id: str) -> SectionIndex:
        return self._docs[doc_id][section_id]

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._docs


# Single process-wide store (fine for a demo/single-instance deployment)
STORE = DocumentStore()
