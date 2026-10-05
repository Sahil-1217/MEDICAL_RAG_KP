"""Unit and integration tests for STEP 4: Neural Embeddings & Vector Database Store."""

import shutil
from pathlib import Path
import numpy as np
import pytest
from config import EmbeddingModelType, VectorDBType
from src.chunking import TextChunk
from src.embeddings import (
    EmbeddingEngine,
    ChromaVectorStore,
    FAISSVectorStore,
    VectorStoreManager,
    VectorSearchResult,
)


@pytest.fixture
def embedding_engine():
    """Fixture providing initialized local MiniLM EmbeddingEngine."""
    return EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def sample_chunks():
    """Fixture providing test chunks with rich clinical metadata."""
    c1 = TextChunk(
        chunk_id="chunk_diabetes_01",
        text="Metformin 500mg daily is the first-line pharmacotherapy for type 2 diabetes mellitus.",
        token_count=16,
        strategy="fixed",
        source_name="WHO_Guidelines.pdf",
        source_type="PDF",
        page_number=1,
        section_header="Pharmacological Protocols",
        cluster_id=1,
        cluster_label="Pharmacology & Therapeutics",
        metadata={"drug": "metformin", "source_type": "PDF"},
    )
    c2 = TextChunk(
        chunk_id="chunk_pneumonia_01",
        text="Severe community-acquired pneumonia requires Ceftriaxone 2g IV daily and Azithromycin 500mg daily.",
        token_count=17,
        strategy="fixed",
        source_name="CDC_Pneumonia.pdf",
        source_type="PDF",
        page_number=2,
        section_header="Antimicrobial Protocols",
        cluster_id=2,
        cluster_label="Treatment Protocols & Guidelines",
        metadata={"drug": "ceftriaxone", "source_type": "PDF"},
    )
    c3 = TextChunk(
        chunk_id="chunk_pubmed_01",
        text="A randomized clinical trial evaluating SGLT2 inhibitors and metformin in glycemic control.",
        token_count=15,
        strategy="fixed",
        source_name="PubMed: Glycemic Control",
        source_type="PUBMED",
        page_number=1,
        section_header="Abstract",
        cluster_id=1,
        cluster_label="Pharmacology & Therapeutics",
        metadata={"pmid": "12345678", "source_type": "PUBMED"},
    )
    return [c1, c2, c3]


def test_embedding_engine_properties(embedding_engine):
    """Verify embedding dimensions and L2 normalization."""
    assert embedding_engine.dimension == 384

    texts = ["Diabetes management", "Pneumonia treatment"]
    embeddings = embedding_engine.embed_documents(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384

    # Verify L2 normalization (norm == 1.0)
    norm = np.linalg.norm(embeddings[0])
    assert pytest.approx(norm, abs=1e-3) == 1.0

    query_emb = embedding_engine.embed_query("Blood glucose")
    assert len(query_emb) == 384


def test_embedding_semantic_similarity(embedding_engine):
    """Verify semantic search relevance: diabetes query matches diabetes text closer than unrelated text."""
    emb_query = np.array(embedding_engine.embed_query("metformin oral dosage for blood sugar"))
    emb_diabetes = np.array(embedding_engine.embed_query("Metformin 500mg daily for type 2 diabetes"))
    emb_unrelated = np.array(embedding_engine.embed_query("planetary orbits in astrophysics"))

    sim_diabetes = np.dot(emb_query, emb_diabetes)
    sim_unrelated = np.dot(emb_query, emb_unrelated)
    assert sim_diabetes > sim_unrelated


def test_chroma_vector_store(embedding_engine, sample_chunks, tmp_path):
    """Verify ChromaDB indexing, metadata filtering, and retrieval."""
    chroma_dir = tmp_path / "test_chroma"
    store = ChromaVectorStore(
        embedding_engine=embedding_engine,
        persist_dir=chroma_dir,
        collection_name="test_collection",
    )
    store.clear()
    store.add_chunks(sample_chunks)
    assert store.count() == 3

    # Search without filter
    results = store.similarity_search("first-line treatment for diabetes", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert isinstance(top_hit, VectorSearchResult)
    assert "metformin" in top_hit.text.lower()
    assert top_hit.score > 0.0

    # Search with metadata filter
    filtered_results = store.similarity_search(
        "diabetes medication",
        top_k=2,
        filter_metadata={"source_type": "PUBMED"},
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].chunk_id == "chunk_pubmed_01"


def test_faiss_vector_store(embedding_engine, sample_chunks, tmp_path):
    """Verify FAISS vector store indexing, persistence, and search."""
    faiss_dir = tmp_path / "test_faiss"
    store = FAISSVectorStore(
        embedding_engine=embedding_engine,
        index_dir=faiss_dir,
    )
    store.clear()
    store.add_chunks(sample_chunks)
    assert store.count() == 3

    # Search query
    results = store.similarity_search("antibiotics for severe pneumonia", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "pneumonia" in top_hit.text.lower()

    # Test persistence: initialize second instance from same directory
    store2 = FAISSVectorStore(
        embedding_engine=embedding_engine,
        index_dir=faiss_dir,
    )
    assert store2.count() == 3
    results2 = store2.similarity_search("antibiotics for severe pneumonia", top_k=1)
    assert len(results2) == 1
    assert results2[0].chunk_id == top_hit.chunk_id


def test_vector_store_manager_facade(sample_chunks, tmp_path):
    """Verify VectorStoreManager switches backends cleanly."""
    manager_chroma = VectorStoreManager(
        backend=VectorDBType.CHROMADB,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    manager_chroma.clear()
    manager_chroma.add_chunks(sample_chunks)
    assert manager_chroma.count() >= 3

    manager_faiss = VectorStoreManager(
        backend=VectorDBType.FAISS,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    manager_faiss.clear()
    manager_faiss.add_chunks(sample_chunks)
    assert manager_faiss.count() >= 3

    res_chroma = manager_chroma.similarity_search("ceftriaxone", top_k=1)
    res_faiss = manager_faiss.similarity_search("ceftriaxone", top_k=1)
    assert len(res_chroma) == 1
    assert len(res_faiss) == 1
    assert "ceftriaxone" in res_chroma[0].text.lower()
    assert "ceftriaxone" in res_faiss[0].text.lower()
