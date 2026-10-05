"""Unit tests for STEP 1: Core Configuration."""

import os
from pathlib import Path
import pytest
from config import Settings, ChunkingStrategy, EmbeddingModelType, VectorDBType, LLMModelType, settings


def test_settings_initialization():
    """Verify settings loads with valid default attributes."""
    assert settings.DEFAULT_CHUNK_SIZE == 500
    assert 200 in settings.CHUNK_SIZES
    assert 500 in settings.CHUNK_SIZES
    assert 1000 in settings.CHUNK_SIZES
    assert settings.CHUNK_OVERLAP_PERCENTAGE == 0.10


def test_directory_creation():
    """Verify directories are created upon initialization."""
    settings.ensure_directories()
    assert settings.RAW_PDF_DIR.exists()
    assert settings.PROCESSED_DATA_DIR.exists()
    assert settings.CHROMA_PERSIST_DIR.exists()
    assert settings.FAISS_INDEX_DIR.exists()
    assert settings.EVALUATION_OUTPUT_DIR.exists()


def test_enum_members():
    """Verify supported enum types are correct."""
    assert ChunkingStrategy.FIXED == "fixed"
    assert ChunkingStrategy.PARENT_CHILD == "parent_child"
    assert ChunkingStrategy.SEMANTIC == "semantic"

    assert VectorDBType.CHROMADB == "chromadb"
    assert VectorDBType.FAISS == "faiss"

    assert EmbeddingModelType.OPENAI_SMALL == "text-embedding-3-small"
    assert EmbeddingModelType.PUBMED_BERT == "pritamdeka/S-PubMedBert-MS-MARCO"


def test_thresholds_and_hyperparameters():
    """Verify pipeline thresholds and hyperparameters."""
    assert settings.PARENT_CHUNK_SIZE == 600
    assert settings.CHILD_CHUNK_SIZE == 150
    assert settings.RERANKER_RELEVANCE_THRESHOLD == 0.65
    assert settings.ISOLATION_FOREST_CONTAMINATION == 0.05
    assert settings.RRF_K_CONSTANT == 60
    assert settings.NLI_ENTAILMENT_THRESHOLD == 0.70
    assert len(settings.TOPIC_CATEGORIES) == 5
