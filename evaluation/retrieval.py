from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .corpus import Chunk, load_chunks, tokenize


def _bm25_idf(n_docs: int, df: int) -> float:
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    bm25: float
    tfidf: float
    rank: int

    def as_dict(self) -> dict:
        return {
            "id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "text": self.chunk.text,
            "score": round(self.score, 4),
            "bm25": round(self.bm25, 4),
            "tfidf": round(self.tfidf, 4),
            "rank": self.rank,
        }


class HybridRetriever:
    """Lexical BM25 + TF-IDF cosine retriever. Fast, deterministic, zero hallucination."""

    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks or load_chunks()
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(c.text) for c in self.chunks]
        self._doc_len = np.array([len(toks) or 1 for toks in self._tokenized], dtype=float)
        self._avgdl = float(self._doc_len.mean()) if len(self._doc_len) else 1.0
        df: Counter[str] = Counter()
        for toks in self._tokenized:
            df.update(set(toks))
        n = len(self.chunks) or 1
        self._idf = {term: _bm25_idf(n, freq) for term, freq in df.items()}
        self._tfidf = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=lambda x: x,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        texts = [c.text for c in self.chunks] if self.chunks else [""]
        matrix = self._tfidf.fit_transform(texts)
        self._tfidf_matrix = normalize(matrix)

    def _expand_query(self, query: str) -> str:
        expansions = {
            "ceo": "chief executive officer",
            "cto": "chief technology officer",
            "ciso": "chief information security officer",
            "hq": "headquarters portland",
            "ticker": "public exchange ipo",
            "cheapest": "starter",
            "clickstream": "raw events retention",
            "leftovers": "raw events retain",
        }
        extra = [phrase for key, phrase in expansions.items() if re.search(rf"\b{key}\b", query, re.I)]
        return f"{query} {' '.join(extra)}".strip()

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(self._expand_query(query))
        scores = np.zeros(len(self.chunks), dtype=float)
        if not q_terms or not self.chunks:
            return scores
        for i, toks in enumerate(self._tokenized):
            tf = Counter(toks)
            dl = self._doc_len[i]
            s = 0.0
            for term in q_terms:
                idf = self._idf.get(term)
                if not idf:
                    continue
                freq = tf.get(term, 0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
            scores[i] = s
        return scores

    def _tfidf_scores(self, query: str) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=float)
        expanded = self._expand_query(query)
        vec = self._tfidf.transform([expanded])
        vec = normalize(vec)
        sims = self._tfidf_matrix.dot(vec.T).toarray().ravel()
        return sims

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Retrieved]:
        if not self.chunks:
            return []
        bm25 = self._bm25_scores(query)
        tfidf = self._tfidf_scores(query)
        b_max = float(bm25.max()) if len(bm25) and bm25.max() > 0 else 1.0
        t_max = float(tfidf.max()) if len(tfidf) and tfidf.max() > 0 else 1.0
        b_norm = bm25 / b_max
        t_norm = tfidf / t_max

        if mode == "bm25":
            final = b_norm
        elif mode == "tfidf":
            final = t_norm
        else:
            final = 0.55 * b_norm + 0.45 * t_norm

        order = np.argsort(-final)[:k]
        out: list[Retrieved] = []
        for rank, idx in enumerate(order, start=1):
            out.append(
                Retrieved(
                    chunk=self.chunks[idx],
                    score=float(final[idx]),
                    bm25=float(bm25[idx]),
                    tfidf=float(tfidf[idx]),
                    rank=rank,
                )
            )
        return out


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def retrieve(query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
    """Top-level convenience retrieval function returning dictionaries."""
    retriever = get_retriever()
    results = retriever.search(query, k=top_k, mode=mode)
    return [r.as_dict() for r in results]
