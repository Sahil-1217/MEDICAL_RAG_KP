"""Multi-Strategy Chunking & Unsupervised Clustering Module.

Features:
- Token-accurate multi-strategy chunking:
    1. Fixed-size chunking (200, 500, 1000 tokens with 10% overlap)
    2. Parent-Child (Small-to-Big) chunking (150-token child -> 600-token parent)
    3. Semantic Boundary chunking (splits along section headers and tables)
- TopicClusterer: Unsupervised K-Means clustering with automated clinical domain labeling.
- QueryAnomalyDetector: IsolationForest anomaly detection for out-of-domain queries.
- Sentence & token-boundary preserving sliding windows with complete provenance metadata.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tiktoken
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ChunkingStrategy, settings
from src.ingestion import IngestedDocument, MedicalNLPProcessor

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TextChunk:
    """Represents a discrete text chunk with complete provenance and ML metadata."""
    chunk_id: str
    text: str
    token_count: int
    strategy: str  # 'fixed', 'parent_child', or 'semantic'
    source_name: str
    source_type: str  # 'PDF' or 'PUBMED'
    page_number: int
    section_header: str = ""
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "strategy": self.strategy,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "section_header": self.section_header,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "metadata": self.metadata,
        }


# =============================================================================
# TOKENIZER UTILITY
# =============================================================================

class Tokenizer:
    """Wrapper around tiktoken with fallback to whitespace tokenizer."""

    def __init__(self, model_name: str = "cl100k_base") -> None:
        try:
            self.encoding = tiktoken.get_encoding(model_name)
        except Exception:
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            except Exception:
                self.encoding = None

    def encode(self, text: str) -> List[int]:
        if self.encoding:
            return self.encoding.encode(text, disallowed_special=())
        return [hash(w) % 100000 for w in text.split()]

    def decode(self, tokens: List[int]) -> str:
        if self.encoding:
            return self.encoding.decode(tokens)
        return " ".join([str(t) for t in tokens])

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding:
            return len(self.encoding.encode(text, disallowed_special=()))
        return len(text.split())


# Global tokenizer instance
_tokenizer = Tokenizer()


# =============================================================================
# MULTI-STRATEGY CHUNKERS
# =============================================================================

class FixedSizeChunker:
    """Fixed-size token chunker supporting 200, 500, and 1000 tokens with 10% overlap."""

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        overlap_percentage: float = settings.CHUNK_OVERLAP_PERCENTAGE,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_tokens = int(chunk_size * overlap_percentage)
        self.step_size = max(1, self.chunk_size - self.overlap_tokens)
        self.tokenizer = tokenizer or _tokenizer

    def chunk_document(self, document: IngestedDocument) -> List[TextChunk]:
        """Chunk all pages of an IngestedDocument using fixed-size token window."""
        chunks: List[TextChunk] = []

        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split into sentence units to avoid splitting words/sentences abruptly
            sentences = re.split(r"(?<=[.!?])\s+", page_text)
            current_tokens: List[int] = []
            current_sentence_parts: List[str] = []
            page_chunk_idx = 0

            for sentence in sentences:
                sent_clean = sentence.strip()
                if not sent_clean:
                    continue

                sent_tokens = self.tokenizer.encode(sent_clean)

                # If single sentence exceeds chunk_size, split by sub-tokens
                if len(sent_tokens) > self.chunk_size:
                    for i in range(0, len(sent_tokens), self.step_size):
                        sub_tokens = sent_tokens[i: i + self.chunk_size]
                        sub_text = self.tokenizer.decode(sub_tokens)
                        chunk_id = f"{document.doc_id}_p{page.page_number}_c{page_chunk_idx}"
                        page_chunk_idx += 1
                        chunks.append(TextChunk(
                            chunk_id=chunk_id,
                            text=sub_text.strip(),
                            token_count=len(sub_tokens),
                            strategy=ChunkingStrategy.FIXED.value,
                            source_name=document.source_name,
                            source_type=document.source_type,
                            page_number=page.page_number,
                            section_header=page.section_headers[0] if page.section_headers else "",
                            metadata=dict(document.metadata),
                        ))
                    continue

                if len(current_tokens) + len(sent_tokens) <= self.chunk_size:
                    current_tokens.extend(sent_tokens)
                    current_sentence_parts.append(sent_clean)
                else:
                    # Flush current window
                    chunk_text = " ".join(current_sentence_parts)
                    chunk_id = f"{document.doc_id}_p{page.page_number}_c{page_chunk_idx}"
                    page_chunk_idx += 1
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text.strip(),
                        token_count=len(current_tokens),
                        strategy=ChunkingStrategy.FIXED.value,
                        source_name=document.source_name,
                        source_type=document.source_type,
                        page_number=page.page_number,
                        section_header=page.section_headers[0] if page.section_headers else "",
                        metadata=dict(document.metadata),
                    ))

                    # Retain overlap sentences for sliding context
                    overlap_tokens_collected = 0
                    overlap_sentences: List[str] = []
                    for s in reversed(current_sentence_parts):
                        s_tok_len = self.tokenizer.count_tokens(s)
                        if overlap_tokens_collected + s_tok_len <= self.overlap_tokens:
                            overlap_sentences.insert(0, s)
                            overlap_tokens_collected += s_tok_len
                        else:
                            break

                    current_sentence_parts = overlap_sentences + [sent_clean]
                    current_tokens = self.tokenizer.encode(" ".join(current_sentence_parts))

            # Flush any remaining tokens
            if current_sentence_parts:
                chunk_text = " ".join(current_sentence_parts)
                chunk_id = f"{document.doc_id}_p{page.page_number}_c{page_chunk_idx}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    token_count=len(current_tokens),
                    strategy=ChunkingStrategy.FIXED.value,
                    source_name=document.source_name,
                    source_type=document.source_type,
                    page_number=page.page_number,
                    section_header=page.section_headers[0] if page.section_headers else "",
                    metadata=dict(document.metadata),
                ))

        return chunks


class ParentChildChunker:
    """Parent-Child (Small-to-Big) chunking strategy.

    Deconstructs documents into 600-token Parent chunks and maps them to
    150-token Child chunks for high-precision retrieval with broad context expansion.
    """

    def __init__(
        self,
        parent_size: int = settings.PARENT_CHUNK_SIZE,
        child_size: int = settings.CHILD_CHUNK_SIZE,
        overlap_percentage: float = settings.PARENT_CHILD_OVERLAP,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap_percentage = overlap_percentage
        self.tokenizer = tokenizer or _tokenizer
        self.parent_chunker = FixedSizeChunker(
            chunk_size=parent_size,
            overlap_percentage=overlap_percentage,
            tokenizer=self.tokenizer,
        )
        self.child_chunker = FixedSizeChunker(
            chunk_size=child_size,
            overlap_percentage=overlap_percentage,
            tokenizer=self.tokenizer,
        )

    def chunk_document(self, document: IngestedDocument) -> List[TextChunk]:
        """Generate Parent chunks and linked Child chunks."""
        all_chunks: List[TextChunk] = []
        parent_chunks = self.parent_chunker.chunk_document(document)

        for p_idx, parent_chunk in enumerate(parent_chunks):
            parent_id = f"{parent_chunk.chunk_id}_PARENT"
            parent_chunk.chunk_id = parent_id
            parent_chunk.strategy = ChunkingStrategy.PARENT_CHILD.value

            # Create dummy document for the parent chunk to partition into children
            from src.ingestion import PageContent
            dummy_page = PageContent(
                page_number=parent_chunk.page_number,
                text=parent_chunk.text,
                section_headers=[parent_chunk.section_header] if parent_chunk.section_header else [],
            )
            dummy_doc = IngestedDocument(
                doc_id=parent_id,
                source_name=parent_chunk.source_name,
                source_type=parent_chunk.source_type,
                pages=[dummy_page],
                full_text=parent_chunk.text,
                metadata=parent_chunk.metadata,
            )

            child_chunks = self.child_chunker.chunk_document(dummy_doc)
            for c_idx, child in enumerate(child_chunks):
                child.chunk_id = f"{parent_id}_c{c_idx}"
                child.parent_id = parent_id
                child.parent_text = parent_chunk.text
                child.strategy = ChunkingStrategy.PARENT_CHILD.value
                child.section_header = parent_chunk.section_header
                child.metadata = dict(parent_chunk.metadata)
                child.metadata["parent_id"] = parent_id
                child.metadata["parent_text_length"] = len(parent_chunk.text)
                all_chunks.append(child)

        return all_chunks


class SemanticBoundaryChunker:
    """Semantic Boundary chunker: splits by section headers, paragraphs, and tables."""

    def __init__(
        self,
        max_tokens: int = 600,
        min_tokens: int = 50,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokenizer = tokenizer or _tokenizer

    def chunk_document(self, document: IngestedDocument) -> List[TextChunk]:
        """Split document into chunks based on natural document layout boundaries."""
        chunks: List[TextChunk] = []

        for page in document.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Split on double line breaks or section boundaries
            raw_blocks = re.split(r"\n{2,}", page_text)
            current_header = page.section_headers[0] if page.section_headers else ""
            current_block_parts: List[str] = []
            chunk_idx = 0

            for block in raw_blocks:
                b_str = block.strip()
                if not b_str:
                    continue

                # Check if this block is a header
                if any(h.lower() in b_str.lower() for h in page.section_headers) and len(b_str) < 80:
                    current_header = b_str

                tokens_in_block = self.tokenizer.count_tokens(b_str)

                # Check accumulator size
                accum_tokens = self.tokenizer.count_tokens(" ".join(current_block_parts))
                if accum_tokens + tokens_in_block <= self.max_tokens:
                    current_block_parts.append(b_str)
                else:
                    if current_block_parts:
                        c_text = "\n\n".join(current_block_parts)
                        chunk_id = f"{document.doc_id}_p{page.page_number}_sem{chunk_idx}"
                        chunk_idx += 1
                        chunks.append(TextChunk(
                            chunk_id=chunk_id,
                            text=c_text,
                            token_count=self.tokenizer.count_tokens(c_text),
                            strategy=ChunkingStrategy.SEMANTIC.value,
                            source_name=document.source_name,
                            source_type=document.source_type,
                            page_number=page.page_number,
                            section_header=current_header,
                            metadata=dict(document.metadata),
                        ))
                    current_block_parts = [b_str]

            if current_block_parts:
                c_text = "\n\n".join(current_block_parts)
                chunk_id = f"{document.doc_id}_p{page.page_number}_sem{chunk_idx}"
                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    text=c_text,
                    token_count=self.tokenizer.count_tokens(c_text),
                    strategy=ChunkingStrategy.SEMANTIC.value,
                    source_name=document.source_name,
                    source_type=document.source_type,
                    page_number=page.page_number,
                    section_header=current_header,
                    metadata=dict(document.metadata),
                ))

        return chunks


# =============================================================================
# UNSUPERVISED MACHINE LEARNING: TOPIC CLUSTERER & ANOMALY DETECTOR
# =============================================================================

class TopicClusterer:
    """Unsupervised K-Means clustering to discover and tag medical themes across chunks."""

    # Reference medical domain keywords for semantic label alignment
    DOMAIN_VOCABULARY: Dict[str, List[str]] = {
        "Pharmacology & Therapeutics": [
            "metformin", "lisinopril", "dosage", "drug", "medication", "mg", "oral",
            "ceftriaxone", "azithromycin", "dexamethasone", "remdesivir", "therapy",
            "antibiotic", "prescription", "regimen", "pharmacological", "daily",
        ],
        "Clinical Diagnosis & Symptoms": [
            "diagnosis", "symptoms", "hypertension", "diabetes", "pneumonia", "fever",
            "hypoxemia", "blood", "pressure", "glucose", "hba1c", "screening", "signs",
        ],
        "Treatment Protocols & Guidelines": [
            "guideline", "protocol", "recommendation", "management", "first-line",
            "who", "cdc", "criteria", "triage", "intervention", "procedure",
        ],
        "Epidemiology & Prevention": [
            "epidemiology", "risk", "prevention", "lifestyle", "population", "worldwide",
            "mortality", "morbidity", "smoking", "sodium", "exercise", "physical activity",
        ],
        "Laboratory & Procedures": [
            "imaging", "ct scan", "mri", "ecg", "electrocardiogram", "ultrasound",
            "intubation", "ventilation", "icu", "culture", "laboratory", "biopsy",
        ],
    }

    def __init__(self, n_clusters: int = settings.KMEANS_NUM_CLUSTERS) -> None:
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=500,
            ngram_range=(1, 2),
        )
        self.model: Optional[KMeans] = None
        self.cluster_labels_map: Dict[int, str] = {}

    def fit_and_assign(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Train K-Means on chunk representations and attach topic labels to chunk metadata."""
        if not chunks:
            return chunks

        corpus = [c.text for c in chunks]
        k = min(self.n_clusters, len(chunks))

        if k <= 1:
            for c in chunks:
                c.cluster_id = 0
                c.cluster_label = "General Clinical Medicine"
                c.metadata["cluster_id"] = 0
                c.metadata["cluster_label"] = "General Clinical Medicine"
            return chunks

        try:
            X = self.vectorizer.fit_transform(corpus)
            self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_assignments = self.model.fit_predict(X)

            # Discover representative keywords per cluster
            order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
            terms = self.vectorizer.get_feature_names_out()

            self.cluster_labels_map = {}
            for i in range(k):
                top_terms = [terms[ind] for ind in order_centroids[i, :8]]
                assigned_domain = self._match_domain_label(top_terms)
                self.cluster_labels_map[i] = assigned_domain
                logger.info("Cluster %d ('%s') top terms: %s", i, assigned_domain, ", ".join(top_terms[:4]))

            # Update chunks
            for idx, c in enumerate(chunks):
                cid = int(cluster_assignments[idx])
                label = self.cluster_labels_map.get(cid, "General Medicine")
                c.cluster_id = cid
                c.cluster_label = label
                c.metadata["cluster_id"] = cid
                c.metadata["cluster_label"] = label

        except Exception as e:
            logger.error("Error during topic clustering: %s", e)
            for c in chunks:
                c.cluster_id = 0
                c.cluster_label = "General Medicine"
                c.metadata["cluster_id"] = 0
                c.metadata["cluster_label"] = "General Medicine"

        return chunks

    def _match_domain_label(self, terms: List[str]) -> str:
        """Map cluster terms to the best matching clinical topic."""
        scores: Dict[str, int] = {domain: 0 for domain in self.DOMAIN_VOCABULARY}
        for term in terms:
            term_clean = term.lower()
            for domain, keywords in self.DOMAIN_VOCABULARY.items():
                if any(kw in term_clean for kw in keywords):
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain
        return "Clinical Management"


class QueryAnomalyDetector:
    """Isolation Forest anomaly detector for filtering out-of-domain or malicious queries."""

    def __init__(self, contamination: float = settings.ISOLATION_FOREST_CONTAMINATION) -> None:
        self.contamination = contamination
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
        self.model: Optional[IsolationForest] = None
        self.is_fitted: bool = False

    def fit(self, reference_texts: List[str]) -> None:
        """Fit Isolation Forest on authentic medical corpus texts."""
        if not reference_texts or len(reference_texts) < 3:
            logger.warning("Insufficient reference texts to fit Anomaly Detector.")
            return

        try:
            X = self.vectorizer.fit_transform(reference_texts).toarray()
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
            )
            self.model.fit(X)
            self.is_fitted = True
            logger.info("Fitted Query Anomaly Detector on %d reference texts.", len(reference_texts))
        except Exception as e:
            logger.error("Error fitting Anomaly Detector: %s", e)

    def is_anomalous(self, query: str) -> Tuple[bool, float]:
        """Evaluate if incoming user query is an anomaly (out-of-domain or nonsensical).

        Returns:
            Tuple of (is_anomalous: bool, anomaly_score: float).
            score < 0 indicates anomaly; score >= 0 indicates normal medical query.
        """
        if not self.is_fitted or self.model is None:
            # Fallback heuristic: check word count and alphanumeric ratio
            if len(query.strip()) < 3 or len(re.findall(r"[a-zA-Z]", query)) < 3:
                return True, -1.0
            return False, 0.5

        try:
            X = self.vectorizer.transform([query]).toarray()
            # If query has zero overlap with vocabulary
            if X.sum() == 0:
                return True, -0.8

            prediction = self.model.predict(X)[0]  # -1 for anomaly, 1 for normal
            score = float(self.model.decision_function(X)[0])
            is_anomaly = (prediction == -1)
            return is_anomaly, score
        except Exception as e:
            logger.debug("Anomaly detection evaluation error: %s", e)
            return False, 0.0


# =============================================================================
# MASTER CHUNKING PIPELINE
# =============================================================================

class ChunkingPipeline:
    """Master orchestrator converting IngestedDocuments into enriched, clustered TextChunks."""

    def __init__(
        self,
        strategy: Union[ChunkingStrategy, str] = ChunkingStrategy.FIXED,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        nlp_processor: Optional[MedicalNLPProcessor] = None,
    ) -> None:
        self.strategy = ChunkingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.chunk_size = chunk_size
        self.nlp = nlp_processor or MedicalNLPProcessor()
        self.clusterer = TopicClusterer()
        self.anomaly_detector = QueryAnomalyDetector()

    def process_documents(self, documents: List[IngestedDocument]) -> List[TextChunk]:
        """Execute chunking, entity extraction on chunks, and unsupervised clustering."""
        logger.info("Executing chunking pipeline with strategy: %s (chunk_size=%d)", self.strategy.value, self.chunk_size)
        raw_chunks: List[TextChunk] = []

        # 1. Select chunker
        if self.strategy == ChunkingStrategy.FIXED:
            chunker = FixedSizeChunker(chunk_size=self.chunk_size)
            for doc in documents:
                raw_chunks.extend(chunker.chunk_document(doc))

        elif self.strategy == ChunkingStrategy.PARENT_CHILD:
            chunker = ParentChildChunker()
            for doc in documents:
                raw_chunks.extend(chunker.chunk_document(doc))

        elif self.strategy == ChunkingStrategy.SEMANTIC:
            chunker = SemanticBoundaryChunker(max_tokens=self.chunk_size)
            for doc in documents:
                raw_chunks.extend(chunker.chunk_document(doc))

        logger.info("Generated %d raw text chunks across %d documents.", len(raw_chunks), len(documents))

        # 2. Enrich each chunk with entity metadata
        for chunk in raw_chunks:
            enriched_meta = self.nlp.enrich_metadata(chunk.text, chunk.metadata)
            chunk.metadata = enriched_meta

        # 3. Apply unsupervised clustering
        clustered_chunks = self.clusterer.fit_and_assign(raw_chunks)

        # 4. Fit Anomaly Detector on chunk representations
        chunk_texts = [c.text for c in clustered_chunks]
        if chunk_texts:
            self.anomaly_detector.fit(chunk_texts)

        logger.info("Chunking & Clustering pipeline complete. Total chunks: %d", len(clustered_chunks))
        return clustered_chunks
