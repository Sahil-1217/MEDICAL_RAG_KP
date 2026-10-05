"""Hybrid Retrieval & Supervised ML Re-Ranking Module.

Features:
- BM25Retriever: Sparse keyword search using BM25Okapi over tokenized chunks.
- HybridRetriever: Reciprocal Rank Fusion (RRF, k=60) combining Dense vector similarity
  (ChromaDB / FAISS) with Sparse BM25 retrieval.
- CrossEncoderReranker: Supervised ML Cross-Encoder (ms-marco-MiniLM-L-6-v2) re-scoring
  (query, chunk) pairs with sigmoid calibration and 0.65 relevance threshold filtering.
- Small-to-Big Context Expansion: Maps matched child chunks to broad parent context.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import settings
from src.chunking import TextChunk
from src.embeddings import BaseVectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RetrievedChunk:
    """Standardized retrieval candidate with dense, sparse, and rerank scores."""
    chunk_id: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    source_name: str = ""
    source_type: str = "PDF"
    page_number: int = 1
    section_header: str = ""
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    cluster_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Return the highest-fidelity available score."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.rrf_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "rrf_score": round(self.rrf_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "final_score": round(self.final_score, 4),
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "section_header": self.section_header,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "cluster_label": self.cluster_label,
            "metadata": self.metadata,
        }


# =============================================================================
# BM25 SPARSE RETRIEVER
# =============================================================================

class BM25Retriever:
    """Sparse lexical keyword retriever powered by BM25Okapi."""

    def __init__(self, chunks: Optional[List[TextChunk]] = None) -> None:
        self.chunks: List[TextChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenizer for BM25."""
        import re
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Build BM25 index over a collection of TextChunks."""
        if not chunks:
            return

        self.chunks = list(chunks)
        self.tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built with %d documents.", len(self.chunks))

    def search(self, query: str, top_k: int = settings.BM25_RETRIEVAL_TOP_K) -> List[Tuple[TextChunk, float]]:
        """Query BM25 index and return (chunk, normalized_score) tuples."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        max_score = max(doc_scores) if len(doc_scores) > 0 and max(doc_scores) > 0 else 1.0

        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        results: List[Tuple[TextChunk, float]] = []

        for idx in top_indices:
            score = float(doc_scores[idx])
            if score > 0.0:
                normalized_score = score / max_score
                results.append((self.chunks[idx], normalized_score))

        return results


# =============================================================================
# HYBRID RETRIEVER WITH RECIPROCAL RANK FUSION (RRF)
# =============================================================================

class HybridRetriever:
    """Combines Dense Vector Similarity Search with BM25 Sparse Search using RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = settings.RRF_K_CONSTANT,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        """Index chunks across both Dense Vector Store and BM25 Sparse Index."""
        logger.info("Indexing %d chunks in HybridRetriever (Dense + BM25)...", len(chunks))
        self.vector_store.add_chunks(chunks)
        self.bm25_retriever.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        expand_parent: bool = True,
    ) -> List[RetrievedChunk]:
        """Perform hybrid retrieval with Reciprocal Rank Fusion (RRF).

        Formula:
            RRF(d) = dense_weight / (k + rank_dense(d)) + sparse_weight / (k + rank_sparse(d))
        """
        # 1. Dense retrieval
        dense_hits = self.vector_store.similarity_search(
            query=query,
            top_k=settings.DENSE_RETRIEVAL_TOP_K,
            filter_metadata=filter_metadata,
        )

        # 2. Sparse BM25 retrieval
        sparse_hits = self.bm25_retriever.search(
            query=query,
            top_k=settings.BM25_RETRIEVAL_TOP_K,
        )

        # Apply metadata filter on sparse hits if requested
        if filter_metadata:
            filtered_sparse = []
            for chunk, score in sparse_hits:
                match = True
                for k, v in filter_metadata.items():
                    val = getattr(chunk, k, None) or chunk.metadata.get(k)
                    if val != v:
                        match = False
                        break
                if match:
                    filtered_sparse.append((chunk, score))
            sparse_hits = filtered_sparse

        # 3. Reciprocal Rank Fusion
        fused_candidates: Dict[str, RetrievedChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # Dense ranks
        for rank, hit in enumerate(dense_hits):
            cid = hit.chunk_id
            rrf_contribution = dense_weight / (self.rrf_k + rank + 1)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            if cid not in fused_candidates:
                fused_candidates[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=hit.text,
                    dense_score=hit.score,
                    sparse_score=0.0,
                    source_name=hit.source_name,
                    source_type=str(hit.metadata.get("source_type", "PDF")),
                    page_number=hit.page_number,
                    section_header=str(hit.metadata.get("section_header", "")),
                    parent_id=hit.parent_id,
                    parent_text=hit.parent_text,
                    cluster_label=str(hit.metadata.get("cluster_label", "")),
                    metadata=hit.metadata,
                )
            else:
                fused_candidates[cid].dense_score = hit.score

        # Sparse ranks
        for rank, (chunk, score) in enumerate(sparse_hits):
            cid = chunk.chunk_id
            rrf_contribution = sparse_weight / (self.rrf_k + rank + 1)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            if cid not in fused_candidates:
                fused_candidates[cid] = RetrievedChunk(
                    chunk_id=cid,
                    text=chunk.text,
                    dense_score=0.0,
                    sparse_score=score,
                    source_name=chunk.source_name,
                    source_type=chunk.source_type,
                    page_number=chunk.page_number,
                    section_header=chunk.section_header,
                    parent_id=chunk.parent_id,
                    parent_text=chunk.parent_text,
                    cluster_label=chunk.cluster_label,
                    metadata=chunk.metadata,
                )
            else:
                fused_candidates[cid].sparse_score = score

        # 4. Attach normalized RRF scores and sort
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        ranked_results: List[RetrievedChunk] = []
        for cid in sorted_cids[:top_k]:
            candidate = fused_candidates[cid]
            candidate.rrf_score = rrf_scores[cid] / max_rrf if max_rrf > 0 else rrf_scores[cid]

            # Small-to-Big Context Expansion: use parent_text if available
            if expand_parent and candidate.parent_text:
                candidate.metadata["original_child_text"] = candidate.text
                candidate.text = candidate.parent_text

            ranked_results.append(candidate)

        return ranked_results


# =============================================================================
# SUPERVISED ML CROSS-ENCODER RE-RANKER
# =============================================================================

class CrossEncoderReranker:
    """Supervised Cross-Encoder (ms-marco-MiniLM-L-6-v2) for deep relevance re-scoring."""

    _model_instance: Optional[CrossEncoder] = None

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL_NAME,
        threshold: float = settings.RERANKER_RELEVANCE_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.model = self._load_model()

    def _load_model(self) -> CrossEncoder:
        if CrossEncoderReranker._model_instance is None:
            logger.info("Loading Cross-Encoder model: %s", self.model_name)
            CrossEncoderReranker._model_instance = CrossEncoder(self.model_name)
        return CrossEncoderReranker._model_instance

    def _sigmoid(self, x: float) -> float:
        """Calibrate raw cross-encoder logits into [0.0, 1.0] probability."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = settings.DEFAULT_TOP_K,
        filter_threshold: bool = True,
    ) -> List[RetrievedChunk]:
        """Re-score candidates against user query and filter low-relevance chunks."""
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        raw_scores = self.model.predict(pairs)

        # Calibrate logits with sigmoid
        calibrated_scores = [self._sigmoid(float(s)) for s in raw_scores]

        for candidate, cal_score in zip(candidates, calibrated_scores):
            candidate.rerank_score = cal_score

        # Sort candidates descending by calibrated re-ranking score
        sorted_candidates = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)

        if filter_threshold:
            filtered = [c for c in sorted_candidates if (c.rerank_score or 0.0) >= self.threshold]
            logger.info(
                "CrossEncoder: retained %d of %d candidates >= threshold %.2f (top score: %.3f)",
                len(filtered),
                len(sorted_candidates),
                self.threshold,
                sorted_candidates[0].rerank_score if sorted_candidates else 0.0,
            )
            # If all candidates fell below threshold, keep at least the single best match to avoid empty context
            if not filtered and sorted_candidates:
                logger.warning("All chunks below threshold %.2f; keeping single best candidate.", self.threshold)
                filtered = [sorted_candidates[0]]
            return filtered[:top_k]

        return sorted_candidates[:top_k]
