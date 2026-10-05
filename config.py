"""Configuration Module for Medical Research Assistant (Agentic RAG System).

Provides centralized management for environment variables, hyperparameters,
model selections, chunking strategies, vector database configurations,
and pipeline thresholds.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    FIXED = "fixed"
    PARENT_CHILD = "parent_child"
    SEMANTIC = "semantic"


class EmbeddingModelType(str, Enum):
    """Supported embedding models across OpenAI and Biomedical HuggingFace models."""
    OPENAI_SMALL = "text-embedding-3-small"
    PUBMED_BERT = "pritamdeka/S-PubMedBert-MS-MARCO"
    BIO_BERT = "dmis-lab/biobert-v1.1"
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"


class VectorDBType(str, Enum):
    """Supported vector database backends."""
    CHROMADB = "chromadb"
    FAISS = "faiss"


class LLMModelType(str, Enum):
    """Supported LLM generation backends."""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_35_TURBO = "gpt-3.5-turbo"


class Settings(BaseSettings):
    """System-wide application settings and hyperparameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Base Project Paths
    # -------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_PDF_DIR: Path = DATA_DIR / "raw_pdfs"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_PERSIST_DIR: Path = PROCESSED_DATA_DIR / "chroma_db"
    FAISS_INDEX_DIR: Path = PROCESSED_DATA_DIR / "faiss_index"

    # -------------------------------------------------------------------------
    # API Credentials
    # -------------------------------------------------------------------------
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    NCBI_EMAIL: str = Field(
        default="medical_researcher@example.com",
        description="User email required for NCBI PubMed Entrez API",
    )
    NCBI_API_KEY: Optional[str] = Field(
        default=None,
        description="Optional NCBI API key to increase Entrez rate limit to 10 req/s",
    )

    # -------------------------------------------------------------------------
    # Chunking Hyperparameters
    # -------------------------------------------------------------------------
    DEFAULT_CHUNK_SIZE: int = 500
    CHUNK_SIZES: List[int] = [200, 500, 1000]
    CHUNK_OVERLAP_PERCENTAGE: float = 0.10  # 10% token overlap

    # Parent-Child (Small-to-Big) Chunking Specs
    PARENT_CHUNK_SIZE: int = 600
    CHILD_CHUNK_SIZE: int = 150
    PARENT_CHILD_OVERLAP: float = 0.10

    # -------------------------------------------------------------------------
    # Unsupervised Clustering & Anomaly Detection
    # -------------------------------------------------------------------------
    KMEANS_NUM_CLUSTERS: int = 5
    TOPIC_CATEGORIES: List[str] = [
        "Pharmacology & Therapeutics",
        "Clinical Diagnosis & Symptoms",
        "Treatment Protocols & Guidelines",
        "Epidemiology & Prevention",
        "Laboratory & Procedures",
    ]
    ISOLATION_FOREST_CONTAMINATION: float = 0.05  # Query anomaly detection threshold

    # -------------------------------------------------------------------------
    # Embedding & Vector Database Settings
    # -------------------------------------------------------------------------
    DEFAULT_EMBEDDING_MODEL: EmbeddingModelType = EmbeddingModelType.OPENAI_SMALL
    DEFAULT_VECTOR_DB: VectorDBType = VectorDBType.CHROMADB
    CHROMA_COLLECTION_NAME: str = "medical_knowledge_base"

    # Embedding model mapping for local / huggingface vs openai
    EMBEDDING_MODEL_MAP: Dict[str, str] = {
        "text-embedding-3-small": "text-embedding-3-small",
        "PubMedBERT": "pritamdeka/S-PubMedBert-MS-MARCO",
        "BioBERT": "dmis-lab/biobert-v1.1",
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    }

    # -------------------------------------------------------------------------
    # Retrieval & Re-Ranking Settings
    # -------------------------------------------------------------------------
    TOP_K_OPTIONS: List[int] = [3, 5, 10]
    DEFAULT_TOP_K: int = 5
    DENSE_RETRIEVAL_TOP_K: int = 20
    BM25_RETRIEVAL_TOP_K: int = 20
    RRF_K_CONSTANT: int = 60  # Reciprocal Rank Fusion smoothing constant

    # Supervised Cross-Encoder Re-Ranker
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_RELEVANCE_THRESHOLD: float = 0.65

    # -------------------------------------------------------------------------
    # LLM Generation & Fact-Checking Agents
    # -------------------------------------------------------------------------
    DEFAULT_LLM_MODEL: LLMModelType = LLMModelType.GPT_4O_MINI
    LLM_TEMPERATURE: float = 0.0  # Deterministic generation for medical rigor
    LLM_MAX_TOKENS: int = 1200

    # NLI Grounding Verifier
    NLI_VERIFIER_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    NLI_ENTAILMENT_THRESHOLD: float = 0.70
    NLI_CONTRADICTION_THRESHOLD: float = 0.40

    # Biomedical NER Spacy Model
    SCISPACY_MODEL_NAME: str = "en_core_sci_sm"
    NER_LABELS: List[str] = ["DISEASE", "DRUG", "DOSAGE", "PROCEDURE"]

    # -------------------------------------------------------------------------
    # Evaluation & Benchmarking
    # -------------------------------------------------------------------------
    RAGAS_METRICS: List[str] = ["faithfulness", "answer_relevancy", "context_recall"]
    EVALUATION_OUTPUT_DIR: Path = DATA_DIR / "processed" / "evaluation_results"

    def ensure_directories(self) -> None:
        """Create necessary project directories if they do not exist."""
        for path in [
            self.RAW_PDF_DIR,
            self.PROCESSED_DATA_DIR,
            self.CHROMA_PERSIST_DIR,
            self.FAISS_INDEX_DIR,
            self.EVALUATION_OUTPUT_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
