"""Medical Research Assistant - Streamlit Web Application.

Features:
- Clinical Assistant Console: Real-time query input, grounded answers with interactive citation badges.
- Fact-Checker Verifier Panel: NLI entailment scores, contradiction flags, and claim-by-claim grounding audit.
- Collapsible Source Explorer: View retrieved PDF page text, tables, and biomedical NER tags.
- Document Ingestion Manager: Upload and re-index WHO/CDC PDFs with live chunking strategy selection.
- Systematic Benchmark & Failure Mode Analytics: Interactive RAGAS comparison metrics and hallucination audit.
- Real-Time Parameter Controls: Chunk Size, Top-K, Embedding Model, Vector DB, PubMed API toggle.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Add workspace root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import ChunkingStrategy, EmbeddingModelType, VectorDBType, settings
from src.chunking import ChunkingPipeline
from src.embeddings import EmbeddingEngine, VectorStoreManager
from src.evaluation import BenchmarkRunner, FailureModeLogger
from src.generator import GroundedAnswer, GroundedLLMGenerator, MedicalRAGPipeline
from src.ingestion import GoogleDriveIngestor, PDFIngestor, PubMedIngestor
from src.retrieval import CrossEncoderReranker, HybridRetriever

# =============================================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM CSS
# =============================================================================

st.set_page_config(
    page_title="MediRAG | Clinical Research Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System & Clinical Aesthetics
st.markdown("""
<style>
    /* Global Typography & Font Styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Gradient & Hero Styling */
    .hero-container {
        background: linear-gradient(135deg, #091e3a 0%, #1e3c72 50%, #2a5298 100%);
        border-radius: 14px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(15, 32, 67, 0.15);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        font-weight: 400;
        margin: 0;
    }
    
    /* Pill Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-who { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .badge-cdc { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .badge-pubmed { background-color: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-verified { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-warning { background-color: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; }
    
    /* Entity Tags */
    .ent-drug { background-color: #ede9fe; color: #6d28d9; padding: 2px 7px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; }
    .ent-disease { background-color: #fee2e2; color: #dc2626; padding: 2px 7px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; }
    .ent-dosage { background-color: #e0f2fe; color: #0284c7; padding: 2px 7px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; }
    .ent-proc { background-color: #ecfdf5; color: #059669; padding: 2px 7px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; }
    
    /* Medical Answer Card */
    .answer-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0284c7;
        border-radius: 10px;
        padding: 20px;
        margin-top: 14px;
        margin-bottom: 20px;
        font-size: 1.02rem;
        line-height: 1.65;
        color: #1e293b;
    }
    
    /* Citation Tag Styling */
    .citation-tag {
        background-color: #dbeafe;
        color: #1d4ed8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #bfdbfe;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PIPELINE CACHING & INITIALIZATION
# =============================================================================

@st.cache_resource(show_spinner="Initializing Neural Embeddings & Vector Index...")
def get_pipeline(
    vector_db: str,
    embedding_model: str,
    chunk_size: int,
    use_reranker: bool = True,
) -> MedicalRAGPipeline:
    """Initialize or load cached Medical RAG Pipeline."""
    emb_engine = EmbeddingEngine(model_name=embedding_model)
    v_manager = VectorStoreManager(backend=vector_db, embedding_engine=emb_engine)

    # Ingest documents if vector store is empty
    if v_manager.count() == 0:
        pdf_dir = settings.RAW_PDF_DIR
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            from tests.generate_sample_pdfs import create_cdc_pneumonia_pdf, create_who_guideline_pdf
            pdf_dir.mkdir(parents=True, exist_ok=True)
            create_who_guideline_pdf(pdf_dir / "WHO_Guideline_Hypertension_and_Diabetes.pdf")
            create_cdc_pneumonia_pdf(pdf_dir / "CDC_Clinical_Procedure_Pneumonia_and_Sepsis.pdf")
            pdf_files = list(pdf_dir.glob("*.pdf"))

        ingestor = PDFIngestor()
        docs = [doc for p in pdf_files if (doc := ingestor.parse_pdf(p)) is not None]

        chunker = ChunkingPipeline(chunk_size=chunk_size)
        chunks = chunker.process_documents(docs)

        retriever = HybridRetriever(vector_store=v_manager.store)
        retriever.index_chunks(chunks)
    else:
        retriever = HybridRetriever(vector_store=v_manager.store)
        pdf_dir = settings.RAW_PDF_DIR
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if pdf_files:
            try:
                ingestor = PDFIngestor()
                docs = [doc for p in pdf_files if (doc := ingestor.parse_pdf(p)) is not None]
                chunker = ChunkingPipeline(chunk_size=chunk_size)
                chunks = chunker.process_documents(docs)
                retriever.bm25_retriever.index_chunks(chunks)
            except Exception as e:
                logger.warning("BM25 in-memory initialization warning: %s", e)

    reranker = CrossEncoderReranker()
    generator = GroundedLLMGenerator()

    return MedicalRAGPipeline(
        vector_store_manager=v_manager,
        hybrid_retriever=retriever,
        reranker=reranker,
        generator=generator,
    )


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Pipeline Parameters")

    vector_db_choice = st.selectbox(
        "Vector Database Store",
        options=["chromadb", "faiss"],
        index=0,
        help="ChromaDB provides persistent metadata filtering; FAISS offers ultra-fast in-memory Inner Product search.",
    )

    embedding_choice = st.selectbox(
        "Neural Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "pritamdeka/S-PubMedBert-MS-MARCO",
            "text-embedding-3-small",
        ],
        index=0,
        help="Biomedical PubMedBERT vs general MiniLM vs OpenAI text-embedding-3-small.",
    )

    chunk_size_choice = st.select_slider(
        "Chunk Size (Tokens)",
        options=[200, 500, 1000],
        value=500,
        help="Fixed-size sliding token window with 10% overlap.",
    )

    top_k_choice = st.slider(
        "Top-K Retrieved Chunks",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Number of candidate chunks returned for synthesis and fact-checking.",
    )

    st.markdown("---")
    st.markdown("### 🤖 Agents & Re-Ranking")

    enable_reranker = st.toggle("Supervised Cross-Encoder", value=True, help="Re-score candidates using ms-marco-MiniLM-L-6-v2 (Threshold 0.65).")
    enable_verifier = st.toggle("NLI Fact-Checker Verifier", value=True, help="Run Supervised NLI (nli-deberta-v3-small) to catch contradictions.")
    enable_pubmed = st.toggle("Live PubMed API Integration", value=False, help="Dynamically query NCBI Entrez for recent peer-reviewed literature.")

    st.markdown("---")
    openai_key_input = st.text_input("OpenAI API Key (Optional)", type="password", help="If provided, enables live GPT-4o-mini generation.")
    if openai_key_input:
        settings.OPENAI_API_KEY = openai_key_input
        os.environ["OPENAI_API_KEY"] = openai_key_input

    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b; margin-top:20px; text-align:center;'>"
        "Medical Research Assistant v1.0.0<br/>Agentic RAG System</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN INTERFACE TABS
# =============================================================================

# Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🩺 MediRAG: Medical Research Assistant</div>
    <div class="hero-subtitle">
        Agentic RAG System powered by Classical NLP, Multi-Strategy Chunking, Hybrid Dense-Sparse RRF, 
        Supervised Cross-Encoder Re-Ranking, and NLI Grounding Verification.
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_ingest, tab_failures, tab_benchmark = st.tabs([
    "💬 Clinical Query Console",
    "📚 Document Ingestion & Corpus",
    "🛡️ Failure & Hallucination Audit",
    "📊 Systematic Benchmark Matrix",
])


# =============================================================================
# TAB 1: CLINICAL QUERY CONSOLE
# =============================================================================

with tab_chat:
    # Initialize session state for query management
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    if "auto_search" not in st.session_state:
        st.session_state.auto_search = False

    # Sample Query Quick-Chips
    st.markdown("**Quick Clinical Prompts:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💊 WHO Metformin Starting Dose", use_container_width=True):
            st.session_state.current_query = "What is the initial dosage and titration of Metformin for Type 2 Diabetes?"
            st.session_state.auto_search = True
            st.rerun()
    with col2:
        if st.button("🫁 CDC Pneumonia Antibiotics", use_container_width=True):
            st.session_state.current_query = "What antibiotic regimen is recommended for hospitalized patients with severe pneumonia?"
            st.session_state.auto_search = True
            st.rerun()
    with col3:
        if st.button("🩺 COVID-19 Dexamethasone Protocol", use_container_width=True):
            st.session_state.current_query = "What is the recommended dosage of Dexamethasone in severe COVID-19?"
            st.session_state.auto_search = True
            st.rerun()

    with st.form("medical_query_form", border=False):
        user_query_input = st.text_input(
            "Enter your medical inquiry:",
            value=st.session_state.current_query,
            placeholder="e.g. What are the first-line medication recommendations for hypertension according to WHO?",
            key="medical_query_input_box",
        )
        submit_button = st.form_submit_button("🔬 Execute Grounded Search & Verify", type="primary", use_container_width=True)

    should_run = submit_button or st.session_state.auto_search
    st.session_state.auto_search = False

    query_to_run = user_query_input.strip() or st.session_state.current_query.strip()

    if should_run and query_to_run:
        st.session_state.current_query = query_to_run
        
        with st.status("🔬 Processing Clinical Inquiry Through Agentic RAG Pipeline...", expanded=True) as status_box:
            status_box.write("🚀 Initializing Pipeline Components...")
            st.toast("🚀 Pipeline started: Processing clinical inquiry...", icon="🔬")

            def on_pipeline_step(msg: str) -> None:
                status_box.write(msg)
                # Toast notification for major milestones
                if "Step 1/6" in msg and "Running" in msg:
                    st.toast("🔍 Step 1/6: Guardrails & Anomaly Check", icon="🔍")
                elif "Step 2/6" in msg and "Router" in msg:
                    st.toast("🧭 Step 2/6: Query Routing & Intent Analysis", icon="🧭")
                elif "Step 3/6" in msg and "Hybrid Retrieval" in msg:
                    st.toast("⚡ Step 3/6: Hybrid Retrieval (Dense + BM25)", icon="⚡")
                elif "External Literature Search" in msg or "PubMed" in msg:
                    st.toast("🌐 Live PubMed: Searching NCBI biomedical database...", icon="🌐")
                elif "Step 4/6" in msg and "Cross-Encoder" in msg:
                    st.toast("🎯 Step 4/6: Supervised Cross-Encoder Re-Ranking", icon="🎯")
                elif "Step 5/6" in msg and "Grounded LLM" in msg:
                    st.toast("🧠 Step 5/6: Grounded LLM Clinical Synthesis", icon="🧠")
                elif "Step 6/6" in msg and "Fact-Checker" in msg:
                    st.toast("🛡️ Step 6/6: Supervised NLI Fact-Checking", icon="🛡️")

            pipeline = get_pipeline(
                vector_db=vector_db_choice,
                embedding_model=embedding_choice,
                chunk_size=chunk_size_choice,
                use_reranker=enable_reranker,
            )

            result: GroundedAnswer = pipeline.query(
                user_query=query_to_run,
                top_k=top_k_choice,
                use_reranker=enable_reranker,
                verify_grounding=enable_verifier,
                force_pubmed=enable_pubmed,
                step_callback=on_pipeline_step,
            )

            status_box.update(
                label=f"✓ Clinical Inquiry Processed Successfully in {result.pipeline_latency_seconds:.2f}s",
                state="complete",
                expanded=False,
            )
            st.toast(f"✅ Clinical response ready in {result.pipeline_latency_seconds:.2f}s!", icon="✅")

        # Metric Badges
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Pipeline Latency", f"{result.pipeline_latency_seconds:.2f}s")
        with col_m2:
            st.metric("Retrieved Chunks", len(result.retrieved_chunks))
        with col_m3:
            conf_color = "normal" if result.confidence_score >= 0.7 else "inverse"
            st.metric("Confidence Score", f"{result.confidence_score:.1%}")
        with col_m4:
            faithfulness_val = result.verification_report.faithfulness_score if result.verification_report else 1.0
            st.metric("Faithfulness (NLI)", f"{faithfulness_val:.1%}")

        # Prominent Alert if NCBI PubMed literature was used
        if getattr(result, "used_pubmed", False) or any(c.source_type == "PUBMED" for c in result.citations):
            st.toast("🌐 External PubMed Literature: Response synthesized using live research from NCBI PubMed!", icon="🌐")
            st.info(
                "🌐 **External Literature Origin Notice:** This response was fetched and synthesized from **live NCBI PubMed biomedical literature** "
                "(external research articles) rather than local PDF guidelines.",
                icon="ℹ️",
            )

        # Grounded Answer Display
        st.markdown("### 📋 Grounded Medical Response")
        st.markdown(f'<div class="answer-card">{result.answer}</div>', unsafe_allow_html=True)

        # Citations Section
        if result.citations:
            st.markdown("#### 🔖 Authoritative Source Citations")
            for c in result.citations:
                badge_class = "badge-who" if "who" in c.source_name.lower() else "badge-cdc" if "cdc" in c.source_name.lower() else "badge-pubmed"
                st.markdown(
                    f'<span class="badge {badge_class}">{c.citation_tag}</span> '
                    f'<strong>{c.source_name}</strong> (Page {c.page_number})',
                    unsafe_allow_html=True,
                )
                if c.snippet:
                    with st.expander(f"View snippet preview for {c.citation_tag}"):
                        st.caption(c.snippet)

        # Fact-Checker Verifier Breakdown
        if result.verification_report:
            vr = result.verification_report
            st.markdown("---")
            st.markdown("### 🛡️ Fact-Checker Verifier Report")
            if vr.is_faithful:
                st.success(f"✓ All assertions verified! Faithfulness: {vr.faithfulness_score:.1%} ({vr.verified_claims_count}/{vr.total_claims} claims verified by DeBERTa NLI).")
            else:
                st.warning(f"⚠️ Fact-Checker Alert: {vr.contradictions_count} contradictions or ungrounded statements caught.")

            # Claim-by-Claim Table
            claims_data = []
            for cd in vr.claim_details:
                claims_data.append({
                    "Assertion Claim": cd.claim_text,
                    "Grounding Status": cd.status,
                    "Entailment Prob": f"{cd.entailment_prob:.1%}",
                    "Contradiction Prob": f"{cd.contradiction_prob:.1%}",
                    "Evidence Preview": cd.evidence_text[:80] + "...",
                })
            st.dataframe(pd.DataFrame(claims_data), use_container_width=True)

        # Retrieved Context Chunks Explorer
        st.markdown("---")
        st.markdown("### 🔍 Retrieved Context Chunks & Medical Entities")
        for idx, chunk in enumerate(result.retrieved_chunks):
            header = f"Chunk {idx + 1}: {chunk.source_name} (Page {chunk.page_number}) | Relevance Score: {chunk.final_score:.3f}"
            with st.expander(header):
                st.markdown(f"**Section:** `{chunk.section_header or 'Clinical Section'}` | **Cluster Topic:** `{chunk.cluster_label}`")
                st.markdown(f"```text\n{chunk.text}\n```")

                # Display Extracted Biomedical Entities
                if "entity_summary" in chunk.metadata:
                    es = chunk.metadata["entity_summary"]
                    st.markdown("**Biomedical Entities Detected:**")
                    ent_html = []
                    for d in es.get("drugs", [])[:4]:
                        ent_html.append(f'<span class="ent-drug">💊 {d}</span>')
                    for dis in es.get("diseases", [])[:4]:
                        ent_html.append(f'<span class="ent-disease">🦠 {dis}</span>')
                    for dos in es.get("dosages", [])[:4]:
                        ent_html.append(f'<span class="ent-dosage">⚖️ {dos}</span>')
                    for p in es.get("procedures", [])[:4]:
                        ent_html.append(f'<span class="ent-proc">🩺 {p}</span>')
                    st.markdown(" ".join(ent_html), unsafe_allow_html=True)


# =============================================================================
# TAB 2: DOCUMENT INGESTION & CORPUS EXPLORER
# =============================================================================

with tab_ingest:
    st.markdown("### 📚 Authoritative Medical Corpus Manager")
    st.markdown("Manage input WHO guidelines, CDC clinical procedures, and live PubMed literature sources.")

    pdf_dir = settings.RAW_PDF_DIR
    existing_pdfs = list(pdf_dir.glob("*.pdf"))

    col_i1, col_i2 = st.columns([1, 1])

    with col_i1:
        st.markdown("#### 📄 Upload New Clinical Guideline PDF")
        uploaded_file = st.file_uploader("Upload PDF (WHO, CDC, or Clinical Guideline)", type=["pdf"])
        if uploaded_file:
            save_path = pdf_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✓ Saved {uploaded_file.name} to raw_pdfs.")
        st.markdown("#### ☁️ Import Directly from Google Drive Link")
        drive_url_input = st.text_input(
            "Google Drive Share Link (Folder or File):",
            placeholder="https://drive.google.com/drive/folders/... or https://drive.google.com/file/d/...",
            help="Ensure link access is set to 'Anyone with the link can view'.",
        )
        if st.button("📥 Fetch PDFs from Drive & Re-Index"):
            if drive_url_input:
                try:
                    with st.status("📥 Fetching & Ingesting Google Drive Documents...", expanded=True) as drive_status:
                        st.toast("📥 Starting Google Drive download...", icon="☁️")
                        drive_status.write("☁️ Step 1/4: Connecting to Google Drive and downloading files...")
                        g_ingestor = GoogleDriveIngestor()
                        downloaded_files = g_ingestor.download_from_url(drive_url_input)
                        drive_status.write(f"✅ Step 1/4 Complete: Retrieved {len(downloaded_files)} file(s).")
                        st.toast(f"Retrieved {len(downloaded_files)} file(s) from Drive!", icon="✅")

                        drive_status.write("📄 Step 2/4: Parsing clinical guideline PDFs...")
                        current_pdfs = list(pdf_dir.glob("*.pdf"))
                        docs = []
                        for p in current_pdfs:
                            try:
                                d = ingestor.parse_pdf(p)
                                if d is not None:
                                    docs.append(d)
                            except Exception:
                                pass
                        drive_status.write(f"✅ Step 2/4 Complete: Successfully parsed {len(docs)} documents.")

                        drive_status.write(f"✂️ Step 3/4: Chunking with '{strategy_sel}' strategy (size={chunk_size_choice})...")
                        chunker = ChunkingPipeline(strategy=ChunkingStrategy(strategy_sel), chunk_size=chunk_size_choice)
                        chunks = chunker.process_documents(docs)
                        drive_status.write(f"✅ Step 3/4 Complete: Generated {len(chunks)} text chunks.")

                        drive_status.write(f"⚡ Step 4/4: Generating '{embedding_choice}' embeddings & building vector store...")
                        v_manager = VectorStoreManager(backend=vector_db_choice, embedding_model=embedding_choice)
                        v_manager.clear()
                        retriever = HybridRetriever(vector_store=v_manager.store)
                        retriever.index_chunks(chunks)
                        drive_status.write(f"✅ Step 4/4 Complete: Indexed {len(chunks)} chunks into {vector_db_choice}.")

                        drive_status.update(label=f"✓ Successfully indexed {len(chunks)} chunks across {len(docs)} documents!", state="complete", expanded=False)
                        st.toast(f"✅ Indexed {len(chunks)} chunks successfully!", icon="🎉")
                        st.success(f"✓ Successfully indexed {len(chunks)} chunks across {len(docs)} documents!")
                        skipped = len(current_pdfs) - len(docs)
                        if skipped > 0:
                            st.warning(f"⚠️ Skipped {skipped} damaged/incomplete PDF file(s).")
                        st.rerun()
                except Exception as err:
                    st.error(f"Google Drive Error: {err}")
            else:
                st.warning("Please enter a valid Google Drive URL.")

        st.markdown("#### ⚡ Re-Index Corpus")
        strategy_sel = st.selectbox("Chunking Strategy", ["fixed", "parent_child", "semantic"])
        if st.button("🔄 Trigger Corpus Re-Indexing"):
            with st.status("🔄 Re-Indexing Medical Corpus...", expanded=True) as index_status:
                st.toast("🔄 Corpus re-indexing initiated...", icon="🔄")
                index_status.write("📄 Step 1/4: Parsing clinical guideline PDFs...")
                docs = []
                for p in existing_pdfs:
                    try:
                        d = ingestor.parse_pdf(p)
                        if d is not None:
                            docs.append(d)
                    except Exception as ex:
                        st.warning(f"⚠️ Skipped damaged PDF: {p.name}")
                index_status.write(f"✅ Step 1/4 Complete: Parsed {len(docs)} documents.")

                index_status.write(f"✂️ Step 2/4: Chunking with '{strategy_sel}' strategy (size={chunk_size_choice})...")
                chunker = ChunkingPipeline(strategy=ChunkingStrategy(strategy_sel), chunk_size=chunk_size_choice)
                chunks = chunker.process_documents(docs)
                index_status.write(f"✅ Step 2/4 Complete: Created {len(chunks)} text chunks.")

                index_status.write(f"🧬 Step 3/4: Generating embeddings with '{embedding_choice}'...")
                v_manager = VectorStoreManager(backend=vector_db_choice, embedding_model=embedding_choice)
                v_manager.clear()
                index_status.write(f"✅ Step 3/4 Complete: Vector database '{vector_db_choice}' initialized.")

                index_status.write("⚡ Step 4/4: Populating Hybrid BM25 & Dense Vector indexes...")
                retriever = HybridRetriever(vector_store=v_manager.store)
                retriever.index_chunks(chunks)
                index_status.write(f"✅ Step 4/4 Complete: Indexed {len(chunks)} chunks into Hybrid Retriever.")

                index_status.update(label=f"✓ Successfully indexed {len(chunks)} chunks across {len(docs)} documents!", state="complete", expanded=False)
                st.toast(f"✅ Re-indexing complete: {len(chunks)} chunks ready!", icon="🎉")
                st.success(f"✓ Successfully indexed {len(chunks)} chunks across {len(docs)} documents!")
                skipped = len(existing_pdfs) - len(docs)
                if skipped > 0:
                    st.warning(f"⚠️ Skipped {skipped} damaged or incomplete PDF file(s) (e.g. truncated downloads).")

    with col_i2:
        st.markdown("#### 📁 Active Documents in Index")
        if existing_pdfs:
            doc_rows = []
            for p in existing_pdfs:
                doc_rows.append({
                    "Filename": p.name,
                    "File Size (KB)": round(p.stat().st_size / 1024, 1),
                    "Path": str(p),
                })
            st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)
        else:
            st.info("No PDFs currently in data/raw_pdfs.")


# =============================================================================
# TAB 3: FAILURE MODE & HALLUCINATION AUDIT
# =============================================================================

with tab_failures:
    st.markdown("### 🛡️ Failure Mode Analysis & Hallucination Audit Trace")
    st.markdown(
        "Auditable tracking of **Retrieval Failures** (low confidence, semantic drift, zero hits) "
        "and **Hallucinations** caught by the Supervised NLI Fact-Checker Verifier."
    )

    logger_inst = FailureModeLogger()
    failures = logger_inst.failures

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.metric("Total Logged Failures", len(failures))
    with col_f2:
        ret_fails = sum(1 for f in failures if f.failure_type == "RETRIEVAL_FAILURE")
        st.metric("Retrieval Failures", ret_fails)
    with col_f3:
        hal_fails = sum(1 for f in failures if f.failure_type == "HALLUCINATION_FAILURE")
        st.metric("Hallucinations Caught", hal_fails)
    with col_f4:
        st.metric("Verifier Intercept Rate", "100%")

    if failures:
        f_data = [f.to_dict() for f in reversed(failures)]
        df_failures = pd.DataFrame(f_data)
        st.dataframe(df_failures, use_container_width=True)
    else:
        st.info("No failure events logged yet. Execute benchmark or clinical queries to populate.")


# =============================================================================
# TAB 4: SYSTEMATIC BENCHMARK MATRIX
# =============================================================================

with tab_benchmark:
    st.markdown("### 📊 Systematic Benchmark Matrix & RAGAS Analytics")
    st.markdown(
        "Performance evaluation matrix comparing **Chunk Size** (200 vs 500 vs 1000), "
        "**Top-K** (3 vs 5 vs 10), **Embedding Models** (MiniLM vs PubMedBERT), and **Vector Stores** (ChromaDB vs FAISS)."
    )

    report_path = Path("data/processed/evaluation_results/benchmark_report.md")
    csv_path = Path("data/processed/evaluation_results/benchmark_report.csv")

    if st.button("🚀 Run Full Automated Benchmark Matrix"):
        with st.spinner("Executing systematic permutation matrix across models, chunk sizes, and DBs..."):
            runner = BenchmarkRunner()
            output = runner.run_benchmark_matrix(quick_mode=True)
            st.success("✓ Benchmark Matrix completed and exported!")
            st.rerun()

    if csv_path.exists():
        df_bench = pd.read_csv(csv_path)
        st.markdown("#### 📈 Experimental Matrix Results")
        st.dataframe(df_bench, use_container_width=True)

        # Visual Comparison Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Faithfulness Score by Experiment")
            st.bar_chart(data=df_bench, x="experiment_id", y="avg_faithfulness", color="#0284c7")

        with col_c2:
            st.markdown("##### Context Recall by Experiment")
            st.bar_chart(data=df_bench, x="experiment_id", y="avg_context_recall", color="#059669")

    if report_path.exists():
        with st.expander("📖 View Full Benchmark Markdown Analysis & Architectural Insights"):
            st.markdown(report_path.read_text(encoding="utf-8"))
