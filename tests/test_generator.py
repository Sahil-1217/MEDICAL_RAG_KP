"""Unit and integration tests for STEP 6: Grounded LLM Generation and Master Pipeline."""

from pathlib import Path
import pytest
from src.chunking import ChunkingPipeline, TextChunk
from src.embeddings import EmbeddingEngine, FAISSVectorStore, VectorStoreManager
from src.generator import GroundedLLMGenerator, MedicalRAGPipeline, GroundedAnswer
from src.ingestion import PDFIngestor
from src.retrieval import HybridRetriever, CrossEncoderReranker


@pytest.fixture
def populated_rag_pipeline(tmp_path):
    """Fixture providing a ready-to-query MedicalRAGPipeline with real indexed PDFs."""
    # 1. Parse sample WHO and CDC PDFs
    who_path = Path("data/raw_pdfs/WHO_Guideline_Hypertension_and_Diabetes.pdf")
    cdc_path = Path("data/raw_pdfs/CDC_Clinical_Procedure_Pneumonia_and_Sepsis.pdf")

    ingestor = PDFIngestor()
    docs = [ingestor.parse_pdf(who_path), ingestor.parse_pdf(cdc_path)]

    # 2. Chunk documents
    chunk_pipeline = ChunkingPipeline(chunk_size=300)
    chunks = chunk_pipeline.process_documents(docs)

    # 3. Index into FAISS store
    emb_engine = EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v_store = FAISSVectorStore(embedding_engine=emb_engine, index_dir=tmp_path / "faiss_pipe")
    v_store.clear()

    retriever = HybridRetriever(vector_store=v_store)
    retriever.index_chunks(chunks)

    reranker = CrossEncoderReranker()
    generator = GroundedLLMGenerator()

    pipeline = MedicalRAGPipeline(
        hybrid_retriever=retriever,
        reranker=reranker,
        generator=generator,
        anomaly_detector=chunk_pipeline.anomaly_detector,
    )
    return pipeline


def test_grounded_generator_citations():
    """Verify GroundedLLMGenerator attaches bracketed citations and extracts them."""
    generator = GroundedLLMGenerator()
    test_chunk = [
        TextChunk(
            chunk_id="test_01",
            text="Metformin 500mg daily is the initial pharmacotherapy for Type 2 Diabetes Mellitus.",
            token_count=14,
            strategy="fixed",
            source_name="WHO_Guidelines.pdf",
            source_type="PDF",
            page_number=1,
            section_header="Pharmacology",
        )
    ]
    from src.retrieval import RetrievedChunk
    retrieved = [
        RetrievedChunk(
            chunk_id=c.chunk_id,
            text=c.text,
            source_name=c.source_name,
            source_type=c.source_type,
            page_number=c.page_number,
        )
        for c in test_chunk
    ]

    answer = generator.generate("What is the first-line medication for diabetes?", retrieved)
    assert len(answer) > 0
    assert "[WHO_Guidelines.pdf, Page 1]" in answer or "WHO" in answer

    citations = generator.extract_citations(answer, retrieved)
    assert len(citations) >= 1
    assert citations[0].page_number == 1


def test_grounded_generator_empty_context_fallback():
    """Verify generator returns fallback when context is empty."""
    generator = GroundedLLMGenerator()
    answer = generator.generate("What is the treatment for hypertension?", [])
    assert "do not contain sufficient evidence" in answer.lower()


def test_end_to_end_pipeline_diabetes_query(populated_rag_pipeline):
    """Verify complete end-to-end RAG query flow on real WHO clinical document."""
    query = "What is the starting dose of Metformin for Type 2 Diabetes?"
    res = populated_rag_pipeline.query(user_query=query, top_k=3)

    assert isinstance(res, GroundedAnswer)
    assert not res.fallback_triggered
    assert len(res.retrieved_chunks) > 0
    assert "metformin" in res.answer.lower()
    assert "500mg" in res.answer.lower() or "500" in res.answer
    assert len(res.citations) >= 1
    assert res.confidence_score > 0.0
    assert res.verification_report is not None


def test_end_to_end_pipeline_pneumonia_query(populated_rag_pipeline):
    """Verify complete end-to-end RAG query flow on real CDC clinical procedure document."""
    query = "What antibiotics are recommended for severe pneumonia?"
    res = populated_rag_pipeline.query(user_query=query, top_k=3)

    assert isinstance(res, GroundedAnswer)
    assert "ceftriaxone" in res.answer.lower() or "azithromycin" in res.answer.lower()
    assert len(res.citations) >= 1
    assert any("CDC" in c.source_name for c in res.citations)


def test_end_to_end_pipeline_anomaly_query(populated_rag_pipeline):
    """Verify pipeline rejects nonsensical query with anomaly fallback response."""
    anomalous_query = "xyz987 quantum cosmic bitcoin encryption"
    res = populated_rag_pipeline.query(user_query=anomalous_query)

    assert isinstance(res, GroundedAnswer)
    assert res.fallback_triggered or "out-of-domain" in res.answer.lower() or "sufficient evidence" in res.answer.lower()
