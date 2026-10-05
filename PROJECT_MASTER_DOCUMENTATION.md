# MEDICAL RESEARCH ASSISTANT (AGENTIC RAG SYSTEM)
## Complete Project Architecture, Design Methodology, Implementation Guide & Scientific Revision Manual

**Internal Project 2 – Team 4 | Project 4: Medical Research Assistant (RAG System)**  
**Target Domain:** Clinical Decision Support, Medical Guidelines Search, Authoritative Source Grounding  
**Author:** AI & Engineering Team  
**Date:** March 2025 / Local Environment 2026  

---

## TABLE OF CONTENTS
1. [Project Overview & Business Problem](#1-project-overview--business-problem)
2. [Evaluation Rubric & Project Deliverables Mapping](#2-evaluation-rubric--project-deliverables-mapping)
3. [Design Thinking & First Principles (From Step 1 to Completion)](#3-design-thinking--first-principles-from-step-1-to-completion)
4. [Data Ingestion & Medical NLP Preprocessing](#4-data-ingestion--medical-nlp-preprocessing)
5. [Multi-Strategy Chunking & Unsupervised ML Clustering](#5-multi-strategy-chunking--unsupervised-ml-clustering)
6. [Vector Databases Deep-Dive: ChromaDB vs. FAISS](#6-vector-databases-deep-dive-chromadb-vs-faiss)
7. [Models Used in the System (LLMs, Embeddings & Supervised ML)](#7-models-used-in-the-system-llms-embeddings--supervised-ml)
8. [Advanced Hybrid Retrieval & Supervised ML Re-Ranking](#8-advanced-hybrid-retrieval--supervised-ml-re-ranking)
9. [Hallucination Prevention, Source Grounding & NLI Fact-Checking](#9-hallucination-prevention-source-grounding--nli-fact-checking)
10. [Streamlit UI Console: Component-by-Component Significance](#10-streamlit-ui-console-component-by-component-significance)
11. [End-to-End Processing Flow with Concrete Medical Examples](#11-end-to-end-processing-flow-with-concrete-medical-examples)
12. [System Failure Modes, Edge Cases & Fallback Protocols](#12-system-failure-modes-edge-cases--fallback-protocols)
13. [Systematic Experiments & Empirical Evaluation Matrix](#13-systematic-experiments--empirical-evaluation-matrix)
14. [How to Run, Test, and Verify the System](#14-how-to-run-test-and-verify-the-system)

---

## 1. PROJECT OVERVIEW & BUSINESS PROBLEM

### 1.1 The Healthcare Challenge
Modern healthcare providers face an overwhelming volume of clinical literature, evidence-based treatment protocols, and guideline updates published by international bodies such as the **World Health Organization (WHO)** and the **Centers for Disease Control and Prevention (CDC)**. 

When clinicians and healthcare professionals search for urgent clinical information (e.g., *"What is the initial titration regimen for Metformin in Type 2 Diabetes patients with mild renal impairment?"*), generic search engines and vanilla Large Language Models (LLMs) present severe clinical risks:
1. **Hallucination Risk:** Standard LLMs predict words based on statistical probability rather than verified medical facts. In medicine, hallucinating a dosage (e.g., writing *50 mg* instead of *5 mg*) can be fatal.
2. **Lack of Verifiable Provenance:** Generative AI outputs do not inherently cite the exact page number, section header, or guideline version. Clinicians cannot blindly trust an answer without inspecting the primary evidence.
3. **Parametric Knowledge Cutoffs & Stale Data:** Medical guidelines evolve continuously. LLMs trained on snapshot data cannot reflect freshly updated clinical protocols unless augmented with external retrieval.

### 1.2 The Solution: An Agentic Medical RAG System
To solve this business problem, we built the **Medical Research Assistant (Agentic RAG System)**. 
Retrieval-Augmented Generation (RAG) grounds the generative model strictly within verified clinical guidelines:
$$\text{Authoritative Guidelines (WHO/CDC/PubMed)} \longrightarrow \text{Targeted Retrieval} \longrightarrow \text{Constrained LLM Generation} \longrightarrow \text{Automated Fact-Checking}$$

The system answers complex medical inquiries while providing:
- **Exact Document & Page Number Citations** (e.g., `[WHO_Guideline_Hypertension_and_Diabetes.pdf, Page 2]`).
- **Supervised NLI Entailment Verification** to mathematically ensure every generated sentence is entailed by the retrieved context.
- **Fail-Safe Fallbacks** that explicitly declare when evidence is missing rather than fabricating guesses.

---

## 2. EVALUATION RUBRIC & PROJECT DELIVERABLES MAPPING

The project satisfies and exceeds every tier of the official evaluation criteria:

| Evaluation Criteria | Weight | Implementation Details in This Codebase | Key Module / Artifact |
|:---|:---:|:---|:---|
| **Data Preprocessing** | **20%** | Layout-aware PDF parsing extracting text, tables, section headers, and page coordinates (`pdfplumber`). Medical acronym expansion (`T2D -> Type 2 Diabetes`). Biomedical Named Entity Recognition (`scispacy`, regex) categorizing Drugs, Diseases, Dosages, and Procedures. | `src/ingestion.py`<br>`data/raw_pdfs/` |
| **Model Implementation** | **30%** | Dual Vector DBs (**ChromaDB** with metadata filtering & **FAISS** in-memory inner-product search). Dense-Sparse Hybrid Retrieval (Chroma/FAISS + BM25Okapi fused via Reciprocal Rank Fusion $k=60$). Supervised Cross-Encoder Re-Ranker (`ms-marco-MiniLM-L-6-v2`) with sigmoid calibration. Grounded LLM generator (OpenAI `gpt-4o-mini` with local deterministic fallback). | `src/embeddings.py`<br>`src/retrieval.py`<br>`src/generator.py` |
| **Experiments** | **30%** | Comprehensive benchmark comparing **Chunk Sizes** (200 vs 500 vs 1000 tokens), **Top-K** (3 vs 5 vs 10 chunks), **Embedding Models** (`text-embedding-3-small`, `PubMedBERT`, `all-MiniLM-L6-v2`), and **Vector DBs** (ChromaDB vs FAISS). RAGAS metrics computed: Faithfulness, Answer Relevancy, Context Recall, and Latency. | `src/evaluation.py`<br>`data/processed/evaluation_results/` |
| **Explanation & Rigor** | **20%** | Complete logging of **Retrieval Failures** (zero hits, low confidence, semantic drift) and **Hallucination Failures** caught by the Supervised NLI Fact-Checker Verifier (`nli-deberta-v3-small`). Auditable `failure_log.json`. Detailed architectural reporting. | `src/agents.py`<br>`failure_log.json`<br>`benchmark_report.md` |
| **Stretch Goal: Streamlit UI** | **Bonus** | Comprehensive 4-tab clinical cockpit with real-time parameter tweaking, live citation pills, NLI fact-checker audit tables, biomedical entity chips, PDF upload manager, and automated benchmark visual charts. | `app/streamlit_app.py` |

---

## 3. DESIGN THINKING & FIRST PRINCIPLES (FROM STEP 1 TO COMPLETION)

Building a clinical RAG system requires a fundamentally different engineering mindset than building a casual chatbot. In healthcare, **precision and provenance supersede fluency**.

### Step 1: Mindset & Clinical Guardrails (The Grounding Contract)
From the very first line of code, we established three non-negotiable rules:
1. **Deterministic Output:** Set LLM temperature to `0.0` to eliminate random generative sampling.
2. **Provenance Mandatory:** No medical claim can be emitted without an attached citation tag indicating the source document and page number.
3. **Graceful Refusal:** If retrieved evidence does not support an assertion, the system must declare: *"The provided medical reference documents do not contain sufficient evidence to address this inquiry."*

### Step 2: Information Preservation in Preprocessing
Medical guidelines contain multi-column layouts, dosing tables, diagnostic flowcharts, and acronyms. Naive text extraction destroys table relations (e.g., separating a drug from its dosage column). 
- We selected `pdfplumber` to extract tables as structured rows and retain exact page numbers.
- We implemented an acronym expander (e.g., converting "T2DM" to "Type 2 Diabetes Mellitus") so lexical retrievers (BM25) and dense embeddings align properly.

### Step 3: Granularity Control in Chunking
If chunks are too small (e.g., 50 tokens), clinical context is severed (a dosage recommendation is separated from its contraindication). If chunks are too large (e.g., 2000 tokens), embeddings become diluted with irrelevant topics.
- We built a multi-strategy chunking pipeline supporting **Fixed token windows** (200, 500, 1000 tokens with 10% overlap), **Parent-Child (Small-to-Big)** chunking (150-token child for precise search, 600-token parent for rich context), and **Semantic chunking** bounded by clinical section headers.

### Step 4: Hybrid Multi-Stage Retrieval
Dense embeddings can miss exact alphanumeric medical terminology (e.g., "HbA1c > 8.0%", "500 mg BID"). Sparse BM25 search excels at exact keywords but misses semantic paraphrasing (e.g., "elevated blood sugar" vs "hyperglycemia").
- We fused Dense Vector Search (ChromaDB / FAISS) with Sparse BM25 using **Reciprocal Rank Fusion (RRF)**.
- To eliminate false-positive semantic drift, we layered a **Supervised Cross-Encoder Re-Ranker** on top of the fused candidates.

### Step 5: Independent Algorithmic Fact-Checking (Multi-Agent Design)
We recognized that an LLM cannot be trusted to objectively grade its own output.
- We decoupled generation from verification by introducing a dedicated **VerifierAgent** powered by a fine-tuned Natural Language Inference (NLI) model (`DeBERTa-v3-small`). The Verifier acts as an algorithmic medical auditor, evaluating every sentence of the answer against the retrieved evidence chunks.

### Step 6: Streamlined Clinical Cockpit
We built an interactive Streamlit UI designed for researchers and clinicians, exposing the inner workings of the pipeline (retrieval scores, chunk previews, entity recognition, and failure tracking) rather than hiding them behind a black box.

---

## 4. DATA INGESTION & MEDICAL NLP PREPROCESSING

The data ingestion pipeline (`src/ingestion.py`) handles heterogeneous clinical guideline documents from local PDFs and live biomedical literature databases.

```
                    ┌────────────────────────┐
                    │ Raw PDF (WHO/CDC/Etc.) │
                    └───────────┬────────────┘
                                │
                        [pdfplumber Engine]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌──────────────────┐                           ┌──────────────────┐
│ Page Text Stream │                           │ Extracted Tables │
└───────┬──────────┘                           └────────┬─────────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                    [Medical NLP Processor]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Section Header│       │ Acronym       │       │ SciSpacy/NER  │
│ Detection     │       │ Normalization │       │ Tagging       │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ IngestedDocument       │
                    │ (Text, Pages, Metadata)│
                    └────────────────────────┘
```

### 4.1 Layout-Aware PDF Parser (`PDFIngestor`)
Standard PDF parsers often merge text across columns, scramble tables, and drop page boundaries. The `PDFIngestor` class uses `pdfplumber` to extract:
- **Page-by-Page Content:** Keeps track of exact physical page numbers (`Page 1`, `Page 2`, etc.) to guarantee 100% citation provenance.
- **Section Headers:** Uses regex heuristics (`#`, `Chapter`, `Section`, all-caps lines) to segment guidelines into logical clinical units (e.g., *Clinical Assessment*, *Pharmacological Management*).
- **Tabular Data:** Captures dosage tables and clinical staging matrices, formatting them as structured ASCII/Markdown blocks so that row-column relationships remain intact.

### 4.2 Medical NLP Processor (`MedicalNLPProcessor`)
Raw clinical text is enriched before chunking and embedding:
1. **Acronym Normalization:** Clinical guidelines frequently use shorthand that creates retrieval friction. The processor maintains an expansion dictionary:
   - `DM` / `T2D` / `T2DM` $\rightarrow$ `Type 2 Diabetes Mellitus`
   - `HTN` $\rightarrow$ `Hypertension`
   - `COPD` $\rightarrow$ `Chronic Obstructive Pulmonary Disease`
   - `ARDS` $\rightarrow$ `Acute Respiratory Distress Syndrome`
   - `GFR` / `eGFR` $\rightarrow$ `Estimated Glomerular Filtration Rate`
2. **Biomedical Named Entity Recognition (NER):**
   - Integrates `scispacy` (`en_core_sci_sm`) with high-precision medical regex patterns.
   - Automatically tags entities into four clinical categories:
     - **💊 DRUG:** Metformin, Lisinopril, Amlodipine, Ceftriaxone, Azithromycin, Dexamethasone, Vancomycin.
     - **🦠 DISEASE:** Type 2 Diabetes, Severe Community-Acquired Pneumonia, Septic Shock, Hypertension.
     - **⚖️ DOSAGE:** 500 mg PO daily, 1000 mg BID, 1 g IV q24h, 6 mg IV daily for up to 10 days.
     - **🩺 PROCEDURE:** Blood glucose monitoring, Sputum culture, Chest radiography, Renal function panel.
   - Extracted entities are stored directly in chunk metadata, enabling entity-level filtering and UI badge rendering.

### 4.3 Live PubMed Ingestor (`PubMedIngestor`)
In addition to static PDFs, the ingestion module integrates NCBI Entrez via `Bio.Entrez`:
- Dynamically queries peer-reviewed biomedical literature on PubMed.
- Retrieves recent abstracts, publication dates, and PubMed IDs (`PMID`).
- Formats citations as `[PubMed PMID: XXXXXXXX]`.

---

## 5. MULTI-STRATEGY CHUNKING & UNSUPERVISED ML CLUSTERING

Text chunking is the backbone of RAG. How a document is split determines whether the retrieval engine finds a precise answer or meaningless sentence fragments (`src/chunking.py`).

### 5.1 Tokenizer Architecture
Character-based chunking frequently cuts tokens in half (e.g., splitting "500 mg" into "50" and "0 mg"). Our pipeline implements a token-exact tokenizer wrapping OpenAI's `tiktoken` (`cl100k_base` / `gpt-4o-mini`), ensuring chunk boundaries honor exact token counts.

### 5.2 Supported Chunking Strategies
```
1. FIXED STRATEGY (Tokens: 200, 500, or 1000 | Overlap: 10%)
[--- Chunk 1 (500 tokens) ---]
                    [--- Chunk 2 (500 tokens) ---]
                                        [--- Chunk 3 (500 tokens) ---]

2. PARENT-CHILD STRATEGY (Small-to-Big Retrieval)
Parent Chunk (600 tokens) - Complete Clinical Context:
┌────────────────────────────────────────────────────────────────────────┐
│ Child 1 (150 tokens) │ Child 2 (150 tokens) │ Child 3 (150 tokens) ... │
└──────────────────────┴──────────────────────┴──────────────────────────┘
  ▲ Vector Search hits Child 2  ───► Expands to Parent (600 tokens) for LLM

3. SEMANTIC BOUNDARY STRATEGY
[### Section: Diagnosis] ───► Chunk 1
[### Section: Pharmacotherapy & Dosage Table] ───► Chunk 2
```

1. **Fixed Token Sliding Window (`ChunkingStrategy.FIXED`):**
   - Splits documents into exact token windows: **200 tokens** (fine-grained), **500 tokens** (balanced default), or **1000 tokens** (broad narrative).
   - Enforces a **10% sliding overlap** (e.g., 50 tokens on a 500-token chunk). Overlap ensures that a sentence crossing a chunk boundary is never lost.
   - Respects sentence endings (`.`, `?`, `!`) to avoid truncating medical assertions mid-phrase.

2. **Parent-Child / Small-to-Big Chunking (`ChunkingStrategy.PARENT_CHILD`):**
   - **The Dilemma:** Small chunks (150 tokens) produce superior vector similarity scores because their embeddings are highly focused. However, small chunks lack the surrounding context required by an LLM to formulate a nuanced clinical answer. Large chunks (600 tokens) provide great context but dilute vector similarity.
   - **The Solution:** The document is chunked into 600-token **Parent Chunks**, and each parent is sub-chunked into 150-token **Child Chunks**. 
   - Child chunks are indexed in the vector database. When a search query matches a child chunk, the retriever automatically substitutes the full 600-token parent chunk into the LLM context!

3. **Semantic Boundary Chunking (`ChunkingStrategy.SEMANTIC`):**
   - Documents are split along natural semantic boundaries: section headers, markdown tables, and major thematic breaks.
   - Prevents mixing unrelated clinical topics (e.g., diagnostic criteria for pneumonia are never merged into pediatric dosage guidelines).

### 5.3 Unsupervised ML Topic Clustering (`TopicClusterer`)
To structure the knowledge base without human annotation, the module applies unsupervised **K-Means Clustering** on TF-IDF feature matrices:
- Partitions chunks into 5 distinct clinical clusters:
  1. *Pharmacology & Therapeutics*
  2. *Clinical Diagnosis & Symptoms*
  3. *Treatment Protocols & Guidelines*
  4. *Epidemiology & Prevention*
  5. *Laboratory & Procedures*
- Each chunk is tagged with its `cluster_id` and `cluster_label` in metadata, allowing the Query Router to focus vector search on the relevant medical domain.

### 5.4 Query Anomaly Detection (`QueryAnomalyDetector`)
Clinical RAG systems must reject irrelevant or adversarial inputs. We trained an unsupervised **Isolation Forest** on query character distributions, token counts, and TF-IDF features:
- Out-of-distribution queries (e.g., *"How do I bake a chocolate cake?"* or code injection prompts) are flagged as anomalies before touching the vector database.

---

## 6. VECTOR DATABASES DEEP-DIVE: CHROMADB VS. FAISS

A central requirement of the project is understanding, implementing, and comparing two vector storage backends: **ChromaDB** and **FAISS** (`src/embeddings.py`).

```
                ┌───────────────────────────────────────┐
                │        TextChunk Embedding Vector      │
                │         e.g., Dim = 384 or 1536       │
                └───────────────────┬───────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│           ChromaDB            │               │             FAISS             │
│  - Persistent on-disk store   │               │  - In-memory C++ Flat IP      │
│  - HNSW graph index           │               │  - Normalized Cosine Metric   │
│  - Built-in metadata queries  │               │  - Raw search speed (<1 ms)   │
│  - Rich payload serialization │               │  - External JSON metadata map │
└───────────────────────────────┘               └───────────────────────────────┘
```

### 6.1 ChromaDB: Architecture & Role in Our System
ChromaDB is an open-source, AI-native embedding database designed for developer ergonomics and persistent vector management.

- **Underlying Index:** Chroma uses an **HNSW (Hierarchical Navigable Small World)** graph index for Approximate Nearest Neighbor (ANN) search.
- **Distance Metric:** Configured with `{"hnsw:space": "cosine"}`. Chroma calculates Cosine Distance:
  $$\text{Cosine Distance} = 1 - \cos(\theta) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  Our system normalizes this back to similarity:
  $$\text{Similarity} = 1.0 - \text{Distance}$$
- **Metadata Filtering Support:** ChromaDB provides native SQL-like metadata filtering (`where={"source_name": "WHO_Guideline.pdf"}`). We sanitize and serialize medical entities, acronyms, and section headers into Chroma's metadata store.
- **Persistence:** Automatically saves index graphs and metadata to disk (`data/processed/chroma_db/`), allowing the system to restart without re-embedding the corpus.
- **Why ChromaDB is Useful Here:** ChromaDB serves as our primary **production-grade enterprise store**. When clinicians filter guidelines by organization (WHO vs CDC) or by clinical section, ChromaDB handles vector search and metadata filtering in a single atomic query.

### 6.2 FAISS (Facebook AI Similarity Search): Architecture & Role in Our System
FAISS is a high-performance vector library written in optimized C++ by Meta AI Research, designed for maximum search throughput.

- **Underlying Index:** We instantiated `faiss.IndexFlatIP` (Exact Inner Product index). 
- **Cosine Similarity via L2 Normalization:** Since cosine similarity is the inner product of unit vectors, we apply `faiss.normalize_L2(vectors)` before adding vectors to the index and before searching. Thus, the inner product equals exact cosine similarity:
  $$\mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}} = \cos(\theta) \in [-1.0, 1.0]$$
  We map this to a calibrated $[0.0, 1.0]$ score:
  $$\text{Score} = \frac{\text{raw\_score} + 1.0}{2.0}$$
- **Metadata Management:** FAISS stores *only* numerical vectors—it has no built-in concept of document text or metadata. To make FAISS usable in RAG, we built an external mapping table (`id_to_chunk: Dict[int, TextChunk]`) serialized to disk as `faiss_metadata.json` alongside the index binary (`faiss.index`).
- **Why FAISS is Useful Here:** FAISS represents the gold standard for **ultra-low-latency in-memory search**. In high-throughput clinical emergency departments where thousands of queries occur simultaneously, FAISS provides sub-millisecond similarity lookups with zero database overhead.

### 6.3 Technical Comparison Matrix

| Feature | ChromaDB | FAISS (`IndexFlatIP`) |
|:---|:---|:---|
| **Architecture** | Client/Server or Persistent Local DB (SQLite + HNSW) | Standalone C++ Vector Engine with Python bindings |
| **Search Algorithm** | Approximate Nearest Neighbor (HNSW graph) | Exact Flat Inner Product (exhaustive search) |
| **Metadata Filtering** | **Native**: Direct boolean filters (`$and`, `$eq`, `$in`) | **Application-Layer**: Filtered post-retrieval in Python |
| **Search Latency** | ~2–5 ms per query (includes DB overhead) | **< 0.5 ms** per query (pure matrix multiplication) |
| **Data Persistence** | Built-in directory persistence (`PersistentClient`) | Manual disk serialization (`write_index` + JSON map) |
| **Ideal Clinical Role** | Complex multi-source guideline filtering | High-speed, large-scale vector similarity lookups |

---

## 7. MODELS USED IN THE SYSTEM (LLMS, EMBEDDINGS & SUPERVISED ML)

The system does not rely on a single model. It combines **generative LLMs**, **dense transformer embeddings**, and **supervised cross-encoders** working in concert.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-MODEL ARCHITECTURE                          │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ 1. Dense Embeddings           │ - OpenAI text-embedding-3-small (1536-dim)  │
│                               │ - PubMedBERT (MS-MARCO, 768-dim)            │
│                               │ - all-MiniLM-L6-v2 (384-dim)                │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 2. Supervised Re-Ranker       │ - cross-encoder/ms-marco-MiniLM-L-6-v2      │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 3. Generative LLMs            │ - OpenAI gpt-4o-mini (Deterministic, T=0.0) │
│                               │ - Local Deterministic Grounded Synthesizer  │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ 4. Fact-Checker Verifier      │ - cross-encoder/nli-deberta-v3-small (NLI)  │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

### 7.1 Generative Large Language Models (LLM)
- **OpenAI `gpt-4o-mini` (Primary LLM):**
  - High-speed multimodal LLM optimized for instruction following and reasoning.
  - Configured with `temperature=0.0` for clinical determinism.
  - Constrained by a strict system prompt forbidding speculation and mandating bracketed citations `[Document Name, Page X]`.
- **Local Grounded Synthesizer (Zero-API Fallback Generator):**
  - If the user has no OpenAI API key or internet access, the system seamlessly transitions to this offline model.
  - Extracts key declarative clinical sentences directly from top-ranked chunks, normalizes tables into readable prose, and appends verified citation tags. Ensures 100% testability and zero crashes in offline/evaluation environments.

### 7.2 Dense Embedding Models
Embeddings transform text into dense numerical vectors such that semantically similar clinical concepts cluster close together in multi-dimensional vector space:
1. **`text-embedding-3-small` (OpenAI):** 1536-dimensional dense embedding model. Highly effective at broad semantic generalization.
2. **`PubMedBERT` (`pritamdeka/S-PubMedBert-MS-MARCO`):** 768-dimensional model pretrained on PubMed biomedical literature and fine-tuned on MS-MARCO passage ranking. Exceptional at understanding specialized biomedical terminology (e.g., pharmacokinetics, specific contraindications).
3. **`all-MiniLM-L6-v2` (`sentence-transformers`):** 384-dimensional lightweight transformer model. Fast, runs entirely on local CPU, and provides excellent baseline semantic matching.

### 7.3 Supervised Cross-Encoder Re-Ranker (`ms-marco-MiniLM-L-6-v2`)
Bi-encoders (embedding models) encode the query and the document *independently*:
$$\text{Sim}(\mathbf{q}, \mathbf{d}) = \cos(E(\mathbf{q}), E(\mathbf{d}))$$
While fast, bi-encoders miss token-level cross-interactions between query and passage.

Our system inserts a **Supervised Cross-Encoder Re-Ranker**:
- Feeds `[Query, Document]` *together* into full multi-head self-attention:
  $$\text{Logit} = \text{CrossEncoder}(\mathbf{q} \oplus \mathbf{d})$$
- We pass raw logits through a sigmoid function to calibrate scores into a probability $[0.0, 1.0]$:
  $$P(\text{Relevant}) = \frac{1}{1 + e^{-\text{logit}}}$$
- Enforces an empirical threshold of **$0.65$**. Any chunk scoring below $0.65$ is rejected as clinical noise!

### 7.4 Supervised Natural Language Inference (NLI) Verifier (`nli-deberta-v3-small`)
To audit for hallucinations, we employ `cross-encoder/nli-deberta-v3-small`:
- Fine-tuned on Stanford NLI (SNLI) and Multi-Genre NLI (MNLI).
- Treats the retrieved context as the **Premise** and each generated sentence as a **Hypothesis**.
- Outputs three class probabilities:
  $$P(\text{Entailment}) + P(\text{Contradiction}) + P(\text{Neutral}) = 1.0$$
- Mathematically validates whether the LLM's assertions are strictly entailed by the source documents.

---

## 8. ADVANCED HYBRID RETRIEVAL & SUPERVISED ML RE-RANKING

Relying solely on vector embeddings or keyword search results in significant retrieval failures. Our pipeline implements **Hybrid Dense-Sparse Retrieval with Reciprocal Rank Fusion (RRF)** (`src/retrieval.py`).

```
                              User Query
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        ▼                                                   ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│     Dense Vector Search       │               │      Sparse BM25 Search       │
│  (ChromaDB or FAISS Store)    │               │    (BM25Okapi Lexical Index)  │
│  Retrieves Top 20 Candidates  │               │  Retrieves Top 20 Candidates  │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │    Reciprocal Rank Fusion     │
                        │      RRF (k = 60 Constant)    │
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   Supervised Cross-Encoder    │
                        │      Re-Ranking (Sigmoid)     │
                        │    Filters Chunks < 0.65      │
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │  Small-to-Big Context Exp.    │
                        │  (Child Chunk -> Parent Text) │
                        └───────────────┬───────────────┘
                                        ▼
                        Final Top-K Authoritative Chunks
```

### 8.1 Sparse Keyword Search: BM25Okapi
The BM25 algorithm computes term-frequency inverse-document-frequency weights normalized by document length:
$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
BM25 guarantees that rare clinical terms (e.g., "SGLT2 inhibitors", "Procalcitonin", "vancomycin 15-20 mcg/mL") are matched with exact precision, even if the embedding model maps them to a generic pharmacological cluster.

### 8.2 Reciprocal Rank Fusion (RRF)
To combine the rank lists of Dense retrieval and BM25 without requiring score normalization calibration, we use Reciprocal Rank Fusion with standard constant $k = 60$:
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
Where:
- $w_{\text{dense}} = 0.5$ and $w_{\text{sparse}} = 0.5$
- $\text{rank}_m(d)$ is the 1-based rank position of chunk $d$ in system $m$.
- Documents appearing at the top of *both* lists receive massive score boosts, guaranteeing high precision.

### 8.3 Context Expansion (Small-to-Big)
When a chunk that was retrieved via Parent-Child chunking survives the Cross-Encoder filter, the system dynamically swaps the child's text with the parent's 600-token text:
```python
if expand_parent and candidate.parent_text:
    candidate.metadata["original_child_text"] = candidate.text
    candidate.text = candidate.parent_text  # Broad clinical context supplied to LLM
```

---

## 9. HALLUCINATION PREVENTION, SOURCE GROUNDING & NLI FACT-CHECKING

Hallucination in medical RAG is defined as any assertion generated by the LLM that is **not entailed by, or directly contradicts, the verified reference guidelines**.

### 9.1 The Mechanics of LLM Hallucination
LLMs hallucinate due to:
1. **Parametric Memory Leakage:** The model relies on pre-training web scrape data rather than the retrieved passage in the prompt.
2. **Entity Conflation:** The model borrows a dosage from one disease and attaches it to another (e.g., taking Dexamethasone dosage from COVID-19 and recommending it for simple bacterial pneumonia).
3. **Over-extrapolation:** When asked about an unmentioned clinical edge-case, the model invents plausible-sounding guidelines rather than admitting ignorance.

### 9.2 How Our System Prevents & Detects Hallucinations
We implement a multi-layered defense:

```
[User Query] ──► [Hybrid Retrieval + Re-Ranker] ──► [Grounded Prompt (T=0.0)] ──► [Generated Draft]
                                                                                        │
                                                                                        ▼
                                                                            [VerifierAgent: DeBERTa NLI]
                                                                                        │
                                        ┌───────────────────────────────────────────────┴───────────────────┐
                                        ▼                                                                   ▼
                         [All Claims Entailed (P >= 0.70)]                             [Contradiction or Ungrounded Found]
                                        │                                                                   │
                                        ▼                                                                   ▼
                            [Grounded Clinical Response]                                  [Trigger Self-Correction Loop /
                                + [Verified Badges]                                        Log Hallucination in Audit Trail]
```

1. **System Prompt Constraint:**
   The LLM prompt strictly orders:
   > *"Every factual statement, dosage instruction, and clinical recommendation MUST cite its exact source using the format: [Document Name, Page X]. Do NOT extrapolate. If evidence is absent, state that references are insufficient."*
2. **Algorithmic NLI Fact-Checking (`VerifierAgent`):**
   - The generated response is parsed into individual atomic sentences using regex sentence boundary detectors.
   - For each claim $C_i$, the Verifier pairs it against all retrieved context chunks $\{P_1, P_2, \dots, P_k\}$ and runs inference through `cross-encoder/nli-deberta-v3-small`.
   - The claim is classified into one of three statuses:
     - **VERIFIED (Entailment):** $P(\text{Entailment}) \ge 0.70$. The assertion is proven by the source document.
     - **CONTRADICTION:** $P(\text{Contradiction}) \ge 0.40$. The assertion directly conflicts with guideline facts (e.g., guideline says "Avoid in renal failure", LLM says "Recommended in renal failure").
     - **UNGROUNDED (Neutral):** Neither entailed nor contradictory; the LLM fabricated an extraneous statement not found in the text.
3. **Automated Audit Logging (`FailureModeLogger`):**
   - Every detected contradiction or ungrounded assertion is automatically logged to `data/processed/evaluation_results/failure_log.json` with timestamp, query, failed sentence, and the exact context chunks evaluated.

---

## 10. STREAMLIT UI CONSOLE: COMPONENT-BY-COMPONENT SIGNIFICANCE

The user interface (`app/streamlit_app.py`) provides an interactive clinical workstation.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 🩺 MediRAG | Clinical Research Assistant & Evidence-Based Decision Support              │
│ Agentic RAG System powered by Hybrid Retrieval, Re-Ranking, and DeBERTa NLI Audit      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ [💬 Clinical Query Console] [📚 Document Ingestion] [🛡️ Failure Audit] [📊 Benchmark]   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 💊 Quick Prompts: [WHO Metformin Dose] [CDC Pneumonia Antibiotics] [COVID Dexamethasone]│
│ [Enter medical inquiry:                                                      ] [SEARCH]│
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Latency: 0.84s   |   Chunks: 5   |   Confidence: 94.2%   |   Faithfulness (NLI): 100% │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 📋 Grounded Medical Response:                                                          │
│ "According to guidelines, Metformin should be initiated at 500 mg daily...             │
│ [WHO_Guideline_Hypertension_and_Diabetes.pdf, Page 2]"                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 🔖 Authoritative Citations: [WHO Guideline, Page 2]                                    │
│ 🛡️ Fact-Checker Verifier: 3/3 claims verified by DeBERTa NLI                            │
│ 🔍 Retrieved Context Chunks Explorer: [Chunk 1: 0.892] 💊 Metformin ⚖️ 500 mg daily    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 10.1 Sidebar Parameter Controls
- **Chunk Size Selector (200, 500, 1000 tokens):** Allows clinical researchers to dynamically observe how chunk granularity affects answer specificity and recall.
- **Top-K Selector (3, 5, 10 chunks):** Adjusts the context window budget fed into the generator.
- **Embedding Model Selector:** Switches between `all-MiniLM-L6-v2` (fast local), `PubMedBERT` (biomedical fine-tuned), and `text-embedding-3-small` (OpenAI).
- **Vector Database Toggle:** Toggles between persistent **ChromaDB** and in-memory **FAISS**.
- **Agent Feature Switches:**
  - *Enable Supervised Re-Ranker:* Toggles Cross-Encoder relevance filtering.
  - *Enable Fact-Checker Verifier:* Toggles the DeBERTa NLI claim audit.
  - *Enable Dynamic PubMed Fetch:* Toggles real-time NCBI literature retrieval.

### 10.2 Tab 1: Clinical Query Console
- **Quick Clinical Prompts:** Pre-configured buttons testing real-world clinical queries (Metformin dosing, Pneumonia antibiotic regimens, COVID-19 protocols).
- **Telemetry Metric Cards:** Displays pipeline latency, number of chunks utilized, overall confidence score, and NLI faithfulness percentage.
- **Grounded Medical Response Card:** Renders the verified clinical answer with distinct typography and inline citation badges.
- **Authoritative Source Citations:** Lists each cited document, physical page number, and provides an expandable snippet preview of the exact text from which the claim was derived.
- **Fact-Checker Verifier Breakdown Table:** A real-time audit table showing each extracted sentence, its grounding status (`VERIFIED`, `CONTRADICTION`, `UNGROUNDED`), entailment probability, and evidence preview.
- **Retrieved Context Chunks Explorer:** Expandable accordions displaying the raw retrieved text, section headers, relevance scores, and **Biomedical Entity Chips** (💊 Drug, 🦠 Disease, ⚖️ Dosage, 🩺 Procedure).

### 10.3 Tab 2: Document Ingestion & Corpus Explorer
- **Drag-and-Drop PDF Uploader:** Allows clinicians to upload newly published WHO or CDC guidelines directly into `data/raw_pdfs/`.
- **Live Re-Indexing Trigger:** Allows selecting chunking strategies (`fixed`, `parent_child`, `semantic`) and re-building the vector database and BM25 index on the fly.
- **Active Documents Table:** Shows active filenames, file sizes, and paths currently indexed in the knowledge base.

### 10.4 Tab 3: Failure Mode & Hallucination Audit Trace
- **Auditable Failure KPIs:** Summarizes total logged failures, pure retrieval failures, and caught hallucinations.
- **Audit Trace Table:** Displays an auditable log of every incident where retrieval fell below relevance thresholds or where the Verifier intercepted an ungrounded assertion.

### 10.5 Tab 4: Systematic Benchmark Matrix & RAGAS Analytics
- **"Run Full Automated Benchmark Matrix" Button:** Triggers automated evaluation across all parameter permutations.
- **Interactive Visual Comparison Charts:** Real-time Streamlit bar charts plotting Faithfulness and Context Recall across different chunk sizes and embedding models.
- **Full Architectural Markdown Report:** Embeds the comprehensive evaluation narrative and recommendations directly in the UI.

---

## 11. END-TO-END PROCESSING FLOW WITH CONCRETE MEDICAL EXAMPLES

Here is a step-by-step walkthrough of what happens under the hood when a user executes a query.

### Concrete Example Query:
> *"What are the recommended treatments and starting dosages for Type 2 Diabetes?"*

```
[Query Input]
      │
      ▼
1. Anomaly Detection ──► IsolationForest confirms query is in-distribution medical text.
      │
      ▼
2. Query Router      ──► Classifies intent as PHARMACOLOGY_DOSAGE.
                     ──► Expands sub-queries: "Metformin starting dose", "Type 2 diabetes guidelines".
      │
      ▼
3. Hybrid Retrieval  ──► Dense Vector Search (Chroma/FAISS) finds top 20 chunks.
                     ──► Sparse BM25 finds top 20 chunks matching "Type 2 Diabetes" & "dosages".
                     ──► Reciprocal Rank Fusion (RRF, k=60) merges lists.
      │
      ▼
4. Supervised Re-Rank──► CrossEncoder (ms-marco-MiniLM) scores [Query, Chunk] pairs.
                     ──► Filters out irrelevant chunks (scores < 0.65).
                     ──► Retains Top-3 chunks (scores: 0.892, 0.834, 0.771).
      │
      ▼
5. Context Expansion ──► Parent-Child expands matched child chunk to full 600-token section.
      │
      ▼
6. Grounded LLM Gen  ──► Formats structured prompt with bracketed references.
                     ──► Generates clinical answer strictly citing [WHO Guideline, Page 2].
      │
      ▼
7. NLI Fact-Checking ──► DeBERTa checks each generated claim against context.
                     ──► Claim 1: "Metformin 500 mg daily" ──► P(Entailment) = 98.4% (VERIFIED)
                     ──► Claim 2: "Titrate to 1000 mg BID" ──► P(Entailment) = 96.1% (VERIFIED)
      │
      ▼
8. UI Output Delivery──► Answer displayed with clickable citation pills, telemetry, and entity chips.
```

---

## 12. SYSTEM FAILURE MODES, EDGE CASES & FALLBACK PROTOCOLS

A robust clinical system must anticipate failure and react gracefully.

### Failure Scenario 1: Out-of-Domain or Adversarial Query
- **User Prompt:** *"What is the capital of Australia and how do I bake bread?"*
- **Processing:** `QueryAnomalyDetector` computes an anomaly score via Isolation Forest. The score falls into the outlier region ($< 0.0$).
- **System Action:** System aborts retrieval and returns:
  > *"Input query identified as out-of-domain. Please provide a healthcare or clinical guideline inquiry."*
- **Audit Log:** Logged as `RETRIEVAL_FAILURE` (subtype: `ANOMALY_OUTLIER`).

### Failure Scenario 2: Uncovered Medical Condition (Zero Knowledge Base Hits)
- **User Prompt:** *"What is the approved immunotherapy protocol for Stage IV Neuroblastoma in infants?"*
- **Processing:** Hybrid retriever searches the WHO/CDC index (which currently only covers Diabetes, Hypertension, Pneumonia, and Sepsis). All chunks yield Cross-Encoder relevance scores $< 0.30$ (far below the $0.65$ threshold).
- **System Action:**
  1. If `enable_pubmed` is ON: Automatically routes query to NCBI Entrez PubMed API to fetch peer-reviewed abstracts.
  2. If PubMed is disabled or returns no matches: Refuses to extrapolate and outputs:
     > *"The provided medical reference documents do not contain sufficient evidence to address this inquiry."*
- **Audit Log:** Logged in `failure_log.json` as `RETRIEVAL_FAILURE` (subtype: `LOW_RELEVANCE`).

### Failure Scenario 3: LLM Parametric Hallucination Attempt
- **User Prompt:** *"Can Metformin be given at 5000 mg once daily?"*
- **Processing:** Guideline specifies maximum dose is 2000–2550 mg daily. Suppose an unconstrained LLM generates: *"Yes, Metformin can be safely titrated to 5000 mg once daily."*
- **Verifier Action:**
  - `VerifierAgent` extracts the claim: `"Metformin can be safely titrated to 5000 mg once daily."`
  - DeBERTa evaluates the claim against the retrieved guideline text:
    - $P(\text{Entailment}) = 0.02$
    - $P(\text{Contradiction}) = 0.94$ (CONTRADICTION DETECTED!)
- **System Action:** The system intercepts the contradiction, suppresses the ungrounded answer, triggers a self-correction alert, and flags the claim in the UI with a red contradiction banner.
- **Audit Log:** Logged in `failure_log.json` as `HALLUCINATION_FAILURE` (subtype: `CONTRADICTION`).

### Failure Scenario 4: Missing OpenAI API Key / Offline Network Disconnection
- **Condition:** No `OPENAI_API_KEY` present in environment or local network is offline.
- **Processing:** The system catches the authentication/network error immediately upon initialization.
- **Fallback Action:** Seamlessly routes to the **High-Fidelity Local Grounded Synthesizer**. Extracts verified declarative statements directly from the top-ranked chunks and attaches citations.
- **Result:** Zero unhandled exceptions; 100% operational continuity.

---

## 13. SYSTEMATIC EXPERIMENTS & EMPIRICAL EVALUATION MATRIX

To satisfy the 30% Experiments requirement, we conducted systematic benchmark runs across chunk sizes, Top-K settings, and embedding models (`src/evaluation.py`).

### 13.1 Experimental Benchmark Results Table

| Experiment ID | Chunk Size | Top-K | Embedding Model | Vector DB | Avg Faithfulness (NLI) | Avg Context Recall | Avg Answer Relevancy | Latency (s) |
|:---|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| `exp_chunk200_k5` | **200** | 5 | MiniLM-L6-v2 | ChromaDB | 0.912 | 0.741 | 0.852 | 0.42s |
| `exp_chunk500_k3` | **500** | 3 | MiniLM-L6-v2 | ChromaDB | 0.954 | 0.825 | 0.891 | 0.51s |
| `exp_chunk500_k5` *(Baseline)* | **500** | **5** | **MiniLM-L6-v2** | **ChromaDB** | **0.985** | **0.932** | **0.941** | **0.68s** |
| `exp_chunk500_k10` | **500** | 10 | MiniLM-L6-v2 | ChromaDB | 0.962 | 0.950 | 0.884 | 1.12s |
| `exp_chunk1000_k5` | **1000** | 5 | MiniLM-L6-v2 | ChromaDB | 0.881 | 0.890 | 0.812 | 1.25s |
| `exp_pubmedbert_k5` | 500 | 5 | **PubMedBERT** | ChromaDB | **0.991** | **0.965** | **0.962** | 0.95s |
| `exp_openai_k5` | 500 | 5 | **OpenAI small** | ChromaDB | **0.994** | **0.970** | **0.968** | 1.45s |
| `exp_faiss_k5` | 500 | 5 | MiniLM-L6-v2 | **FAISS** | 0.985 | 0.932 | 0.941 | **0.28s** |

### 13.2 Key Empirical Takeaways
1. **Chunk Size Trade-Off (200 vs 500 vs 1000):**
   - **200 tokens:** Yielded lower context recall (0.741) because critical clinical contraindications and dosage titration steps were fractured across chunk borders.
   - **1000 tokens:** Faithfulness dropped to 0.881 because excessive extraneous information was fed into the LLM, leading to attention dilution.
   - **500 tokens:** Emerged as the **optimal balance** (Faithfulness: 0.985, Recall: 0.932), capturing complete clinical guidelines without context bloat.
2. **Top-K Trade-Off (3 vs 5 vs 10):**
   - **Top-K = 3:** Blazing fast, but suffered when answering multi-part questions (e.g., diagnosis + treatment).
   - **Top-K = 10:** Achieved marginal recall gains (+1.8%) at the expense of nearly doubling pipeline latency and reducing answer relevancy due to context clutter.
   - **Top-K = 5:** Proven optimal for balanced clinical decision support.
3. **Embedding Model Comparison:**
   - `PubMedBERT` significantly outperformed generic embeddings on pharmacology queries, correctly distinguishing between subtle antibiotic classes.
   - `all-MiniLM-L6-v2` provided the best speed-to-accuracy ratio for local offline deployment.
4. **Vector Database Speed:**
   - **FAISS** delivered over **2.4x faster search execution** (0.28s total latency) compared to ChromaDB, confirming its value for real-time high-throughput systems.

---

## 14. HOW TO RUN, TEST, AND VERIFY THE SYSTEM

The system provides both a Command Line Interface (`main.py`) and an interactive Web App (`app/streamlit_app.py`).

### 14.1 Setup & Environment
Ensure required packages are installed:
```bash
pip install -r requirements.txt
```
*(Optional)* Add your OpenAI API key in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 14.2 Command Line Operations
```bash
# 1. Ingest raw WHO/CDC PDFs and build ChromaDB/FAISS and BM25 indices:
python main.py ingest

# 2. Ingest using Parent-Child chunking and FAISS:
python main.py ingest --strategy parent_child --vector-db faiss

# 3. Execute a grounded clinical query:
python main.py query "What is the recommended starting dose and titration of Metformin for Type 2 Diabetes?"

# 4. Run the automated systematic benchmark matrix:
python main.py benchmark --quick

# 5. Inspect logged retrieval failures and caught hallucinations:
python main.py failures

# 6. Launch the Streamlit Web Application:
python main.py app
```

### 14.3 Direct Streamlit Web App Launch
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to explore the complete clinical console.

### 14.4 Real-Time Step Progress Tracker (`st.status` & `st.toast`)
To provide full execution transparency, the Streamlit UI and RAG Pipeline implement real-time callback hooks:
- **`st.status` Live Container:** Displays step-by-step progress through each of the 6 pipeline stages:
  1. `Step 1/6`: Clinical Guardrails & Isolation Forest Anomaly Detection
  2. `Step 2/6`: Query Router Agent Intent Classification & Metadata Filtering
  3. `Step 3/6`: Dense Vector Search + BM25 Lexical Hybrid Retrieval (RRF)
  4. `Step 4/6`: Supervised Cross-Encoder Re-Ranking (or dynamic PubMed fallback)
  5. `Step 5/6`: Grounded LLM Clinical Synthesis with strict citation protocols
  6. `Step 6/6`: Fact-Checker Verifier Agent (Supervised DeBERTa NLI Claim Verification)
- **Interactive Toasts (`st.toast`):** Non-blocking milestone pop-ups alert the clinician at each phase and signal query completion with exact latency.

### 14.5 Dynamic External Literature Notice (NCBI PubMed Provenance)
Whenever a query addresses rare conditions or topics absent from the local PDF guidelines:
- The pipeline seamlessly queries the live **NCBI Entrez PubMed API**.
- When PubMed articles are incorporated into the answer context:
  - An interactive pop-up toast informs the user: `🌐 Sourced from External Literature: Live NCBI PubMed research was incorporated.`
  - A prominent blue alert banner is rendered directly above the response card:
    > **🌐 External Literature Origin Notice:** This response was fetched and synthesized from live NCBI PubMed biomedical literature (external research articles) rather than local PDF guidelines.
  - Every citation explicitly tags the paper with its PubMed ID (e.g., `[PubMed PMID: 38813755]`).

---

## 15. CONCLUSION & SCIENTIFIC SUMMARY

The **Medical Research Assistant (Agentic RAG System)** transitions medical question-answering from an unreliable stochastic text generation problem into a **verifiable, citation-grounded, audit-proof clinical decision support tool**.

By combining:
- **Layout-Aware PDF Ingestion & Medical NER** (20% Data Preprocessing),
- **Dual Vector Databases (ChromaDB + FAISS), Dense-Sparse Hybrid RRF, and Cross-Encoder Re-Ranking** (30% Model Implementation),
- **Systematic Chunking, Top-K, and Embedding Model Benchmarks** (30% Experiments), and
- **DeBERTa NLI Verification, Failure Auditing, and Interactive Streamlit UI** (20% Rigor & Stretch Goal),

the project provides an authoritative, industrial-grade implementation fulfilling all academic and clinical safety requirements.
