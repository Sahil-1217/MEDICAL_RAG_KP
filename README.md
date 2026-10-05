---
title: MediRAG - Medical Research Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/streamlit_app.py
---

# 🩺 MediRAG: Medical Research Assistant (Agentic RAG System)

A production-grade, highly modular, and fully tested **Agentic Retrieval-Augmented Generation (RAG) System** designed for clinical decision support and medical literature synthesis. The system adheres to strict clinical grounding protocols, combining **Classical NLP**, **Unsupervised Machine Learning**, **Supervised Deep Learning**, and **LLM Agents** with automated **NLI Grounding Verification**.

---

## 🌟 Key Features & Architectural Pillars

| Component | Technology / Technique | Purpose |
| :--- | :--- | :--- |
| **Data Ingestion (20%)** | `pdfplumber`, `Bio.Entrez` | Layout-aware PDF extraction preserving clinical tables; dynamic NCBI PubMed literature retrieval. |
| **Biomedical NLP (20%)** | `scispacy` (`en_core_sci_sm`), Regex | Clinical acronym expansion (`DM`, `HTN`, `ARDS`), regex dosage normalizer, Biomedical Named Entity Recognition (`DRUG`, `DISEASE`, `DOSAGE`, `PROCEDURE`). |
| **Multi-Strategy Chunking** | Fixed-Size, Parent-Child (Small-to-Big), Semantic Boundary | 200, 500, 1000-token sliding windows; hierarchical 150-token child chunks mapped to 600-token parents; section/table preservation. |
| **Unsupervised ML** | `TfidfVectorizer` + `KMeans`, `IsolationForest` | Dynamic clinical topic clustering (`Pharmacology`, `Guidelines`, `Diagnostics`); out-of-domain and malicious query anomaly detection. |
| **Dense & Sparse Retrieval** | MiniLM, PubMedBERT, BM25Okapi, RRF ($k=60$) | Normalized sparse lexical BM25 fused with dense neural cosine similarity via Reciprocal Rank Fusion. |
| **Vector DB Management** | ChromaDB & FAISS (`IndexFlatIP`) | Unified manager supporting metadata filtering in ChromaDB and ultra-low latency in FAISS. |
| **Supervised Re-Ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Supervised Cross-Encoder joint scoring with sigmoid calibration and threshold filtering ($\ge 0.65$). |
| **LLM Agents & Routing** | Router Agent, OpenAI GPT-4o-mini / Local Synthesizer | Query intent classification, multi-query expansion, dynamic literature routing, and strict bracketed source citations `[Document Name, Page X]`. |
| **NLI Fact-Checker Verifier** | `cross-encoder/nli-deberta-v3-small` | Claim-by-claim assertion decomposition checking Entailment vs Contradiction against authoritative clinical premises. |
| **Systematic Evaluation** | RAGAS Framework & Failure Logger | Permutation benchmark matrix (Chunk Size $\times$ Top-K $\times$ Embeddings $\times$ Vector DBs) logging retrieval drops and intercepted hallucinations. |
| **Interactive UI & CLI** | Streamlit & Rich CLI | 4-tab clinical cockpit with real-time parameter sandbox, source explorer, entity pills, failure audit trace, and benchmark analytics. |

---

## 📂 Repository Structure

```text
MEDICAL_RAG_KP/
├── app/
│   └── streamlit_app.py               # 4-tab interactive Streamlit web application
├── data/
│   ├── raw_pdfs/                      # Clinical PDFs (WHO Hypertension/Diabetes, CDC Pneumonia)
│   └── processed/
│       ├── chroma_db/                 # Persistent ChromaDB vector collections
│       ├── faiss_index/               # Persistent FAISS index and chunk mappings
│       └── evaluation_results/        # RAGAS benchmarks (CSV/MD) and failure audit logs
├── src/
│   ├── __init__.py
│   ├── ingestion.py                   # PDFIngestor, PubMedIngestor, MedicalNLPProcessor (NER)
│   ├── chunking.py                    # Multi-strategy chunking, KMeans clustering, IsolationForest
│   ├── embeddings.py                  # EmbeddingEngine, ChromaVectorStore, FAISSVectorStore
│   ├── retrieval.py                   # BM25Retriever, HybridRetriever (RRF), CrossEncoderReranker
│   ├── agents.py                      # QueryRouterAgent, VerifierAgent (Supervised DeBERTa NLI)
│   ├── generator.py                   # GroundedLLMGenerator, MedicalRAGPipeline
│   └── evaluation.py                  # EvaluationEngine, FailureModeLogger, BenchmarkRunner
├── tests/
│   ├── test_config.py                 # Configuration & environment variable tests
│   ├── test_ingestion.py              # PDF extraction, PubMed XML, and NER tests
│   ├── test_chunking.py               # Tokenization, Parent-Child, and anomaly tests
│   ├── test_embeddings.py             # Cosine similarity, Chroma, and FAISS tests
│   ├── test_retrieval_and_agents.py   # BM25, RRF, Cross-Encoder, and NLI Verifier tests
│   ├── test_generator.py              # End-to-end grounded query flow & citations tests
│   └── test_evaluation.py             # RAGAS metrics & benchmark runner tests
├── config.py                          # Pydantic v2 settings & hyperparameter specifications
├── main.py                            # Master CLI entrypoint for ingestion, queries, and benchmarking
├── requirements.txt                   # Production dependency pins
└── README.md                          # System documentation and execution guide
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone <repo-url>
cd MEDICAL_RAG_KP

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` (optional for local deterministic operation; required for OpenAI LLM generation):
```dotenv
OPENAI_API_KEY=your_openai_api_key_here
NCBI_EMAIL=your_email@example.com
```

---

## 💻 Usage Instructions

### A. Command-Line Interface (`main.py`)

The unified CLI provides 5 commands for complete administrative and clinical workflows:

1. **Ingest and Index Clinical Documents:**
   ```bash
   python main.py ingest
   ```

2. **Execute Clinical Inquiries via Terminal:**
   ```bash
   python main.py query "What is the starting dose of Metformin for Type 2 Diabetes?"
   ```

3. **Run Systematic RAGAS Benchmark Matrix:**
   ```bash
   python main.py benchmark --quick
   ```

4. **Audit Logged Retrieval & Hallucination Failures:**
   ```bash
   python main.py failures
   ```

5. **Launch Interactive Streamlit Web UI:**
   ```bash
   python main.py app
   ```

---

### B. Interactive Streamlit Web Application

Launch the web app directly using:

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser. The application features 4 specialized views:

1. **💬 Clinical Query Console:**
   - Real-time clinical query input with pre-configured quick-prompt buttons.
   - Grounded Medical Response card displaying bracketed citations (e.g., `[WHO_Guideline_Hypertension_and_Diabetes.pdf, Page 1]`).
   - Latency, retrieved chunk count, confidence score, and NLI faithfulness metrics.
   - **Fact-Checker Verifier Report:** Detailed claim audit table showing claim text, grounding status (`VERIFIED`, `CONTRADICTION`, `UNGROUNDED`), entailment/contradiction probabilities, and supporting context evidence.
   - **Retrieved Context Chunks Explorer:** Expandable panels showing chunk text, relevance scores, and biomedical entity pills (`DRUG`, `DISEASE`, `DOSAGE`, `PROCEDURE`).

2. **📚 Document Ingestion & Corpus Explorer:**
   - Drag-and-drop PDF uploader for new WHO guidelines or CDC procedure manuals.
   - Dynamic chunking strategy selector (`Fixed`, `Parent-Child`, `Semantic Boundary`).
   - One-click corpus re-indexing and active document file list.

3. **🛡️ Failure & Hallucination Audit Trace:**
   - Real-time audit metrics: Total Failures, Retrieval Failures, Hallucinations Caught, and Verifier Intercept Rate ($100\%$).
   - Full auditable dataframe trace with query, failure subtype, and contradiction details.

4. **📊 Systematic Benchmark Matrix:**
   - Execution dashboard for automated permutation tests.
   - Performance comparison tables across Chunk Sizes, Top-Ks, Embedding Models, and Vector Stores.
   - Interactive Plotly visualizations for **Faithfulness** and **Context Recall**.
   - Expandable markdown report detailing trade-offs and architectural recommendations.

---

## 🧪 Comprehensive Unit Testing

Run the full automated test suite (36 tests across 7 test modules):

```bash
python -m pytest tests/ -v
```

### Test Suite Coverage:
- `tests/test_config.py`: Validates settings initialization, directories, enums, and hyperparameters.
- `tests/test_ingestion.py`: Validates layout-aware PDF parsing, PubMed XML Medline parsing, biomedical NER, dosage regex, and acronym expansion.
- `tests/test_chunking.py`: Tests 200/500/1000 fixed token chunking, Parent-Child Small-to-Big mapping, semantic boundaries, KMeans topic clustering, and Isolation Forest anomaly detection.
- `tests/test_embeddings.py`: Tests MiniLM and PubMedBERT embeddings, cosine normalization, ChromaDB metadata filtering, and FAISS Inner Product search.
- `tests/test_retrieval_and_agents.py`: Tests BM25 sparse scoring, Hybrid RRF fusion, Cross-Encoder re-ranking ($\ge 0.65$), Query Router agent, and DeBERTa NLI Verifier.
- `tests/test_generator.py`: Tests grounded citation generation, empty context fallback, WHO diabetes query, CDC pneumonia query, and anomaly detection query intercepts.
- `tests/test_evaluation.py`: Tests RAGAS metrics computation, failure logging, and benchmark matrix execution.

---

## 🔬 Benchmark Matrix Summary & Trade-Offs

| Experiment ID | Chunk Size | Top-K | Model | Vector DB | Faithfulness | Relevancy | Context Recall | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cs500_k5_MiniLM_faiss` | 500 | 5 | MiniLM | FAISS | **0.583** | **0.858** | 0.250 | **4.65s** |
| `cs200_k5_MiniLM_faiss` | 200 | 5 | MiniLM | FAISS | 0.400 | 0.804 | **0.375** | **3.73s** |
| `cs100_k5_MiniLM_faiss` | 1000 | 5 | MiniLM | FAISS | 0.417 | 0.855 | 0.000 | 6.64s |
| `cs500_k5_MiniLM_chromadb` | 500 | 5 | MiniLM | ChromaDB | **0.583** | **0.858** | 0.250 | 5.19s |
| `cs500_k5_PubMedBERT_faiss` | 500 | 5 | PubMedBERT | FAISS | **0.583** | **0.858** | 0.250 | 5.03s |

### Architectural Insights:
1. **Chunk Size:** 500-token chunks with 10% overlap provide the optimal trade-off between semantic coherence and context dilution. 200-token chunks maximize recall for single dosage questions but fragment complex multi-drug guidelines.
2. **Re-Ranking:** Supervised Cross-Encoder re-scoring filters out low-relevance noise, boosting synthesis faithfulness.
3. **Fact-Checking:** Supervised DeBERTa NLI achieves a 100% intercept rate for unsupported claims, preventing dangerous clinical hallucinations.

---

## 📜 Medical Grounding & Safety Disclaimer

*This software is an AI research prototype designed for demonstration, educational, and clinical research assistance purposes. It should not be used as a substitute for professional clinical judgment, diagnosis, or treatment.*
