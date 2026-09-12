from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Sequence

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Scorer:
    """Lightweight BM25 scorer for recipe chunks without adding a new package dependency."""

    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [_tokenize(doc.page_content) for doc in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.tokenized_docs:
            for term in set(tokens):
                self.doc_freq[term] += 1
        self.num_docs = len(self.documents)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[tuple[int, float]]:
        q_terms = _tokenize(query)
        if not q_terms:
            return [(idx, 0.0) for idx in range(len(self.documents))]

        counts = Counter(q_terms)
        scores: List[tuple[int, float]] = []

        for idx, tokens in enumerate(self.tokenized_docs):
            doc_len = self.doc_lengths[idx]
            score = 0.0
            counter = Counter(tokens)
            for term, qf in counts.items():
                idf = self._idf(term)
                if idf == 0.0:
                    continue
                tf = counter.get(term, 0)
                numerator = idf * tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(1.0, self.avg_doc_length))
                score += numerator / max(1e-9, denominator) * qf
            scores.append((idx, score))

        return sorted(scores, key=lambda item: item[1], reverse=True)


class HybridRecipeRetriever:
    """Combine BM25 lexical retrieval with dense semantic retrieval and rerank with RRF + MMR."""

    def __init__(
        self,
        documents: Sequence[Document],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 5,
        rrf_k: int = 60,
        mmr_lambda: float = 0.6,
    ):
        self.documents = list(documents)
        self.k = k
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [doc.page_content for doc in self.documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self.bm25 = BM25Scorer(self.documents)

    def _semantic_scores(self, query: str) -> List[tuple[int, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        ranked = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked]

    def _rrf_merge(self, lexical_ranked: Sequence[tuple[int, float]], semantic_ranked: Sequence[tuple[int, float]]) -> List[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for rank, (idx, _) in enumerate(lexical_ranked, start=1):
            scores[idx] += 1.0 / (self.rrf_k + rank)
        for rank, (idx, _) in enumerate(semantic_ranked, start=1):
            scores[idx] += 1.0 / (self.rrf_k + rank)

        merged = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(idx, score) for idx, score in merged]

    def _mmr_rerank(self, query: str, ranked_ids: Sequence[int]) -> List[int]:
        if not ranked_ids:
            return []

        if len(ranked_ids) == 1:
            return list(ranked_ids)

        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        chosen: List[int] = []
        unchosen = list(ranked_ids)

        while unchosen:
            best_idx = None
            best_score = -1e9
            for candidate in unchosen:
                doc_emb = self.embeddings[candidate]
                similarity_to_query = float(np.dot(doc_emb, q_emb))
                max_similarity_to_chosen = 0.0
                if chosen:
                    chosen_sims = np.dot(self.embeddings[chosen], doc_emb)
                    max_similarity_to_chosen = float(np.max(chosen_sims))
                mmr_score = similarity_to_query - self.mmr_lambda * max_similarity_to_chosen
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = candidate
            if best_idx is None:
                best_idx = unchosen[0]
            chosen.append(best_idx)
            unchosen.remove(best_idx)
            if len(chosen) >= self.k:
                break

        return chosen

    def retrieve(self, query: str, k: int | None = None) -> List[Document]:
        n = k or self.k

        lexical_ranked = self.bm25.score(query)
        semantic_ranked = self._semantic_scores(query)
        merged = self._rrf_merge(lexical_ranked, semantic_ranked)
        selected_ids = self._mmr_rerank(query, [idx for idx, _ in merged])

        selected_docs: List[Document] = []
        for idx in selected_ids[:n]:
            doc = self.documents[idx]
            selected_docs.append(doc)
        return selected_docs
