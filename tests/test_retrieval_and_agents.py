"""Unit and integration tests for STEP 5: Hybrid Retrieval, Cross-Encoder, and Multi-Agent System."""

import pytest
from src.chunking import TextChunk
from src.embeddings import EmbeddingEngine, FAISSVectorStore
from src.retrieval import BM25Retriever, HybridRetriever, CrossEncoderReranker, RetrievedChunk
from src.agents import QueryRouterAgent, QueryIntent, VerifierAgent


@pytest.fixture
def test_chunks():
    """Fixture providing sample clinical chunks."""
    c1 = TextChunk(
        chunk_id="chunk_metformin",
        text="Metformin is the first-line oral antidiabetic drug prescribed at 500mg daily for Type 2 Diabetes.",
        token_count=18,
        strategy="parent_child",
        source_name="WHO_Guideline_Hypertension_and_Diabetes.pdf",
        source_type="PDF",
        page_number=1,
        section_header="Pharmacological Treatment Protocols",
        parent_id="chunk_metformin_PARENT",
        parent_text="1. Introduction and Treatment Protocols. Metformin is the first-line oral antidiabetic drug prescribed at 500mg daily for Type 2 Diabetes. Titrate up to 1000mg twice daily with meals.",
        cluster_id=1,
        cluster_label="Pharmacology & Therapeutics",
        metadata={"source_type": "PDF"},
    )
    c2 = TextChunk(
        chunk_id="chunk_lisinopril",
        text="Lisinopril 10mg daily is an ACE inhibitor used for blood pressure control in hypertension.",
        token_count=16,
        strategy="fixed",
        source_name="WHO_Guideline_Hypertension_and_Diabetes.pdf",
        source_type="PDF",
        page_number=1,
        section_header="Pharmacological Treatment Protocols",
        cluster_id=1,
        cluster_label="Pharmacology & Therapeutics",
        metadata={"source_type": "PDF"},
    )
    c3 = TextChunk(
        chunk_id="chunk_pneumonia",
        text="Severe pneumonia management requires Ceftriaxone 2g IV daily and Azithromycin 500mg daily.",
        token_count=15,
        strategy="fixed",
        source_name="CDC_Clinical_Procedure_Pneumonia_and_Sepsis.pdf",
        source_type="PDF",
        page_number=1,
        section_header="Antimicrobial Regimens",
        cluster_id=2,
        cluster_label="Treatment Protocols & Guidelines",
        metadata={"source_type": "PDF"},
    )
    return [c1, c2, c3]


def test_bm25_retriever(test_chunks):
    """Verify BM25 sparse search ranks exact lexical matches at rank 1."""
    bm25 = BM25Retriever(test_chunks)
    results = bm25.search("metformin 500mg dosage", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.chunk_id == "chunk_metformin"
    assert score > 0.0


def test_hybrid_retriever_rrf(test_chunks, tmp_path):
    """Verify Reciprocal Rank Fusion combines dense and sparse hits."""
    engine = EmbeddingEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")
    v_store = FAISSVectorStore(embedding_engine=engine, index_dir=tmp_path / "faiss_test")
    v_store.clear()

    retriever = HybridRetriever(vector_store=v_store)
    retriever.index_chunks(test_chunks)

    # Search with RRF and Small-to-Big context expansion
    results = retriever.retrieve("first-line treatment for diabetes mellitus", top_k=2, expand_parent=True)
    assert len(results) > 0
    top_hit = results[0]
    assert top_hit.chunk_id == "chunk_metformin"
    assert top_hit.rrf_score > 0.0
    # Context expansion verification
    assert "Titrate up to 1000mg" in top_hit.text


def test_cross_encoder_reranker(test_chunks):
    """Verify Supervised Cross-Encoder re-ranks and filters low-confidence candidates."""
    reranker = CrossEncoderReranker(threshold=0.65)
    candidates = [
        RetrievedChunk(chunk_id=c.chunk_id, text=c.text, source_name=c.source_name, page_number=c.page_number)
        for c in test_chunks
    ]

    reranked = reranker.rerank("What is the first-line medication for diabetes?", candidates, top_k=2)
    assert len(reranked) > 0
    top = reranked[0]
    assert top.chunk_id == "chunk_metformin"
    assert top.rerank_score is not None
    assert top.rerank_score >= 0.65


def test_query_router_agent():
    """Verify QueryRouterAgent detects intents and builds metadata filters."""
    router = QueryRouterAgent()

    # Guideline Intent
    r1 = router.route_query("According to WHO guidelines, what is the management for hypertension?")
    assert r1.intent in [QueryIntent.CLINICAL_GUIDELINE, QueryIntent.PHARMACOLOGY_DOSAGE]
    assert not r1.should_fetch_pubmed
    assert len(r1.sub_queries) >= 1

    # PubMed Trigger Intent
    r2 = router.route_query("Recent 2024 clinical trial studies on SGLT2 inhibitors from pubmed")
    assert r2.intent == QueryIntent.RECENT_LITERATURE
    assert r2.should_fetch_pubmed
    assert r2.metadata_filters.get("source_type") == "PUBMED"

    # Acronym Expansion in query
    r3 = router.route_query("Management of T2DM in elderly patients")
    assert "Diabetes" in r3.optimized_query


def test_verifier_agent_grounded_vs_hallucination(test_chunks):
    """Verify VerifierAgent NLI identifies grounded claims and catches hallucinations."""
    verifier = VerifierAgent()
    context = [
        RetrievedChunk(
            chunk_id=test_chunks[0].chunk_id,
            text=test_chunks[0].text,
            source_name=test_chunks[0].source_name,
            page_number=test_chunks[0].page_number,
        )
    ]

    # Grounded answer
    grounded_answer = "Metformin is the primary oral drug for type 2 diabetes initiated at 500mg daily."
    report_grounded = verifier.verify_answer(grounded_answer, context)
    assert report_grounded.total_claims >= 1
    assert report_grounded.faithfulness_score >= 0.70
    assert report_grounded.is_faithful

    # Hallucinated answer
    hallucinated_answer = "Patients with diabetes must take 500mg morphine daily to stimulate bone growth."
    report_hallucinated = verifier.verify_answer(hallucinated_answer, context)
    assert not report_hallucinated.is_faithful
    assert len(report_hallucinated.detected_hallucinations) > 0
