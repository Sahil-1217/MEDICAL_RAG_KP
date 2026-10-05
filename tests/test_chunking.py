"""Unit and integration tests for STEP 3: Multi-Strategy Chunking & Unsupervised Clustering."""

from pathlib import Path
import pytest
from config import ChunkingStrategy
from src.ingestion import PDFIngestor, IngestedDocument, PageContent
from src.chunking import (
    FixedSizeChunker,
    ParentChildChunker,
    SemanticBoundaryChunker,
    TopicClusterer,
    QueryAnomalyDetector,
    ChunkingPipeline,
    Tokenizer,
)


@pytest.fixture
def sample_document():
    """Create a multi-page test IngestedDocument."""
    p1 = PageContent(
        page_number=1,
        text=(
            "1. Introduction and Epidemiology\n"
            "Hypertension and Type 2 Diabetes Mellitus are prevalent chronic conditions. "
            "Patients require routine monitoring of blood pressure and blood glucose levels. "
            "First-line pharmacological management for Type 2 Diabetes is Metformin 500mg orally daily. "
            "For hypertension, Lisinopril 10mg orally daily is recommended. Amlodipine 5mg can be added."
        ),
        section_headers=["1. Introduction and Epidemiology"],
    )
    p2 = PageContent(
        page_number=2,
        text=(
            "2. Diagnostic Imaging and Sepsis Management\n"
            "Patients with severe pneumonia require chest CT scans and arterial blood gas evaluation. "
            "Antimicrobial therapy consists of Ceftriaxone 2g IV daily and Azithromycin 500mg IV daily. "
            "If severe hypoxemic respiratory failure ensues, endotracheal intubation and mechanical ventilation "
            "in the intensive care unit (ICU) are initiated promptly."
        ),
        section_headers=["2. Diagnostic Imaging and Sepsis Management"],
    )
    return IngestedDocument(
        doc_id="test_doc_01",
        source_name="Clinical_Guideline_Test.pdf",
        source_type="PDF",
        pages=[p1, p2],
        full_text=f"{p1.text}\n\n{p2.text}",
        metadata={"author": "WHO / CDC Expert Panel"},
    )


def test_tokenizer():
    """Verify tokenizer encodes, decodes, and accurately counts tokens."""
    tok = Tokenizer()
    text = "Clinical management of severe diabetes mellitus with metformin."
    tokens = tok.encode(text)
    assert len(tokens) > 0
    assert tok.count_tokens(text) == len(tokens)
    decoded = tok.decode(tokens)
    assert "diabetes" in decoded.lower()


def test_fixed_size_chunker_sizes(sample_document):
    """Verify FixedSizeChunker across 200, 500, and 1000 token specifications."""
    for size in [200, 500, 1000]:
        chunker = FixedSizeChunker(chunk_size=size, overlap_percentage=0.10)
        chunks = chunker.chunk_document(sample_document)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count <= size + 10  # allow sentence boundary tolerance
            assert chunk.strategy == "fixed"
            assert chunk.page_number in [1, 2]
            assert chunk.source_name == "Clinical_Guideline_Test.pdf"


def test_parent_child_chunker(sample_document):
    """Verify Parent-Child chunking maps child chunks to parent chunks with metadata."""
    chunker = ParentChildChunker(parent_size=600, child_size=150)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 0

    for child in chunks:
        assert child.strategy == "parent_child"
        assert child.parent_id is not None
        assert "_PARENT" in child.parent_id
        assert child.parent_text is not None
        assert len(child.parent_text) >= len(child.text)
        assert "parent_id" in child.metadata


def test_semantic_boundary_chunker(sample_document):
    """Verify SemanticBoundaryChunker splits along logical paragraphs and preserves headers."""
    chunker = SemanticBoundaryChunker(max_tokens=300)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.strategy == "semantic"
        assert chunk.page_number in [1, 2]


def test_topic_clusterer(sample_document):
    """Verify unsupervised K-Means assigns clinical topics to chunks."""
    chunker = FixedSizeChunker(chunk_size=100)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2

    clusterer = TopicClusterer(n_clusters=2)
    clustered = clusterer.fit_and_assign(chunks)
    assert len(clustered) == len(chunks)

    for c in clustered:
        assert c.cluster_id is not None
        assert c.cluster_label is not None
        assert "cluster_label" in c.metadata


def test_query_anomaly_detector(sample_document):
    """Verify IsolationForest distinguishes valid medical queries from anomalous queries."""
    detector = QueryAnomalyDetector(contamination=0.10)
    corpus = [
        sample_document.pages[0].text,
        sample_document.pages[1].text,
        "Metformin dosage in elderly diabetes patients.",
        "Hypertension treatment with ACE inhibitors and ARBs.",
        "Community-acquired pneumonia antibiotic coverage with ceftriaxone.",
    ]
    detector.fit(corpus)

    # Valid in-domain medical query
    is_anom, score = detector.is_anomalous("What is the first-line medication for type 2 diabetes?")
    assert not is_anom, "Medical query should not be flagged as anomaly"

    # Out-of-domain random query
    is_anom_rand, _ = detector.is_anomalous("xyzabc quantum astrophysics bitcoin cryptocurrency price")
    assert is_anom_rand, "Random query should be detected as anomaly"


def test_chunking_pipeline_end_to_end(sample_document):
    """Verify complete master pipeline execution with NER metadata enrichment."""
    pipeline = ChunkingPipeline(
        strategy=ChunkingStrategy.FIXED,
        chunk_size=200,
    )
    processed_chunks = pipeline.process_documents([sample_document])
    assert len(processed_chunks) > 0

    first_chunk = processed_chunks[0]
    assert first_chunk.token_count > 0
    assert first_chunk.cluster_label is not None
    assert "entity_summary" in first_chunk.metadata
    assert first_chunk.metadata["entity_summary"]["total_entities"] >= 1
