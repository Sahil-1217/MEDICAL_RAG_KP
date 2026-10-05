"""Master CLI Entry Point for Medical Research Assistant (Agentic RAG System).

Commands:
    python main.py query "<medical inquiry>"    # Run end-to-end grounded query with citations & NLI verifier
    python main.py ingest                       # Ingest raw WHO/CDC PDFs and build vector stores
    python main.py benchmark                    # Run automated benchmark matrix across chunk sizes, Top-Ks & models
    python main.py failures                     # Display logged retrieval failures and caught hallucinations
    python main.py app                          # Launch interactive Streamlit web dashboard
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import ChunkingStrategy, EmbeddingModelType, VectorDBType, settings
from src.chunking import ChunkingPipeline
from src.embeddings import EmbeddingEngine, VectorStoreManager
from src.evaluation import BenchmarkRunner, FailureModeLogger
from src.generator import GroundedLLMGenerator, MedicalRAGPipeline
from src.ingestion import PDFIngestor
from src.retrieval import CrossEncoderReranker, HybridRetriever

console = Console(highlight=False)


def build_pipeline(
    vector_db: str = "chromadb",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
) -> MedicalRAGPipeline:
    """Initialize and assemble the complete Medical RAG Pipeline."""
    emb_engine = EmbeddingEngine(model_name=embedding_model)
    v_manager = VectorStoreManager(backend=vector_db, embedding_engine=emb_engine)
    retriever = HybridRetriever(vector_store=v_manager.store)
    reranker = CrossEncoderReranker()
    generator = GroundedLLMGenerator()

    return MedicalRAGPipeline(
        vector_store_manager=v_manager,
        hybrid_retriever=retriever,
        reranker=reranker,
        generator=generator,
    )


def handle_ingest(args: argparse.Namespace) -> None:
    """Ingest raw PDFs from data/raw_pdfs and populate vector store."""
    console.print(Panel.fit("[bold blue]Data Ingestion & Medical Indexing[/bold blue]", border_style="blue"))
    pdf_dir = settings.RAW_PDF_DIR

    # Download from Google Drive if URL provided
    if getattr(args, "drive_url", None):
        try:
            from src.ingestion import GoogleDriveIngestor
            console.print(f"[cyan]Fetching research papers from Google Drive URL:[/cyan] {args.drive_url}")
            g_ingestor = GoogleDriveIngestor()
            downloaded = g_ingestor.download_from_url(args.drive_url)
            console.print(f"[green]Successfully retrieved {len(downloaded)} PDF(s) from Google Drive.[/green]")
        except Exception as e:
            console.print(f"[bold red]Failed to download from Google Drive:[/bold red] {e}")

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[yellow]No PDFs found in {pdf_dir}. Generating sample documents first...[/yellow]")
        from tests.generate_sample_pdfs import create_who_guideline_pdf, create_cdc_pneumonia_pdf
        pdf_dir.mkdir(parents=True, exist_ok=True)
        create_who_guideline_pdf(pdf_dir / "WHO_Guideline_Hypertension_and_Diabetes.pdf")
        create_cdc_pneumonia_pdf(pdf_dir / "CDC_Clinical_Procedure_Pneumonia_and_Sepsis.pdf")
        pdf_files = list(pdf_dir.glob("*.pdf"))

    console.print(f"[cyan]Found {len(pdf_files)} PDF document(s) to process:[/cyan]")
    for p in pdf_files:
        console.print(f"  - {p.name}")

    # 1. Parse PDFs
    ingestor = PDFIngestor()
    docs = []
    with console.status("[bold green]Parsing PDFs and extracting medical entities..."):
        for p in pdf_files:
            doc = ingestor.parse_pdf(p)
            if doc is not None:
                docs.append(doc)
            else:
                console.print(f"[yellow]Skipping corrupted/unreadable PDF: {p.name}[/yellow]")

    # 2. Chunk documents
    strategy = ChunkingStrategy(args.strategy)
    chunker = ChunkingPipeline(strategy=strategy, chunk_size=args.chunk_size)
    with console.status(f"[bold green]Chunking documents (strategy={strategy.value}, size={args.chunk_size})..."):
        chunks = chunker.process_documents(docs)

    # 3. Index into vector store
    v_manager = VectorStoreManager(backend=args.vector_db, embedding_model=args.embedding)
    retriever = HybridRetriever(vector_store=v_manager.store)
    with console.status("[bold green]Generating neural embeddings & building BM25 index..."):
        v_manager.clear()
        retriever.index_chunks(chunks)

    console.print(Panel(
        f"[bold green][OK] Ingestion and Indexing Complete![/bold green]\n"
        f"- Total Processed Documents: {len(docs)}\n"
        f"- Total Chunks Generated: {len(chunks)}\n"
        f"- Vector Database: {args.vector_db.upper()}\n"
        f"- Embedding Model: {args.embedding}",
        border_style="green",
    ))


def handle_query(args: argparse.Namespace) -> None:
    """Execute end-to-end query and print grounded answer with provenance."""
    user_query = args.query_text
    console.print(Panel.fit(f"[bold cyan]Clinical Inquiry:[/bold cyan] {user_query}", border_style="cyan"))

    pipeline = build_pipeline(
        vector_db=args.vector_db,
        embedding_model=args.embedding,
    )

    with console.status("[bold green]Executing Agentic RAG Pipeline (Router -> Hybrid -> Re-Rank -> NLI Verifier)..."):
        t0 = time.time()
        answer = pipeline.query(
            user_query=user_query,
            top_k=args.top_k,
            force_pubmed=args.pubmed,
        )
        elapsed = time.time() - t0

    # Display Answer
    console.print("\n[bold green]Grounded Medical Response:[/bold green]")
    console.print(Panel(answer.answer, border_style="green"))

    # Citations Table
    if answer.citations:
        c_table = Table(title="[bold yellow]Authoritative Citations[/bold yellow]", border_style="yellow")
        c_table.add_column("Tag", style="cyan")
        c_table.add_column("Source Document", style="white")
        c_table.add_column("Page", style="magenta")
        c_table.add_column("Snippet Preview", style="dim")
        for c in answer.citations:
            c_table.add_row(c.citation_tag, c.source_name, str(c.page_number), c.snippet[:80] + "...")
        console.print(c_table)

    # Verification Report
    if answer.verification_report:
        vr = answer.verification_report
        v_style = "green" if vr.is_faithful else "red"
        console.print(Panel(
            f"[bold {v_style}]Fact-Checker Verifier Report:[/bold {v_style}]\n"
            f"- Faithfulness Score: {vr.faithfulness_score:.1%} ({vr.verified_claims_count}/{vr.total_claims} claims verified)\n"
            f"- Contradictions Caught: {vr.contradictions_count}\n"
            f"- Ungrounded Claims: {vr.ungrounded_claims_count}\n"
            f"- Overall Status: {'VERIFIED GROUNDED' if vr.is_faithful else 'UNGROUNDED / FLAGGED'}",
            border_style=v_style,
        ))

    console.print(f"[dim]Execution Latency: {elapsed:.2f}s | Confidence Score: {answer.confidence_score:.2f}[/dim]\n")


def handle_benchmark(args: argparse.Namespace) -> None:
    """Run systematic benchmark matrix experiments."""
    console.print(Panel.fit("[bold magenta]Automated RAGAS Benchmark Matrix Runner[/bold magenta]", border_style="magenta"))
    runner = BenchmarkRunner()

    with console.status("[bold green]Running experimental permutations (Chunk Size, Top-K, Models, DBs)..."):
        benchmark_output = runner.run_benchmark_matrix(quick_mode=not args.full)

    results = benchmark_output["results"]
    b_table = Table(title="[bold magenta]Experimental Matrix Results[/bold magenta]", border_style="magenta")
    b_table.add_column("Experiment ID", style="cyan")
    b_table.add_column("Chunk", style="white")
    b_table.add_column("Top-K", style="magenta")
    b_table.add_column("Model", style="yellow")
    b_table.add_column("DB", style="blue")
    b_table.add_column("Faithfulness", style="green")
    b_table.add_column("Context Recall", style="green")
    b_table.add_column("Latency (s)", style="dim")

    for r in results:
        b_table.add_row(
            r["experiment_id"],
            str(r["chunk_size"]),
            str(r["top_k"]),
            r["embedding_model"],
            r["vector_db"],
            f"{r['avg_faithfulness']:.2f}",
            f"{r['avg_context_recall']:.2f}",
            f"{r['avg_latency_sec']:.2f}s",
        )
    console.print(b_table)
    console.print(f"\n[green][OK] Benchmark Report saved to: {benchmark_output['markdown_report_path']}[/green]")


def handle_failures(args: argparse.Namespace) -> None:
    """Display logged retrieval failures and caught hallucinations."""
    console.print(Panel.fit("[bold red]Auditable Failure Modes & Hallucination Trace[/bold red]", border_style="red"))
    logger_inst = FailureModeLogger()
    console.print(logger_inst.export_markdown_summary())


def handle_app(args: argparse.Namespace) -> None:
    """Launch Streamlit frontend."""
    console.print("[bold green]Launching Streamlit Medical Assistant App...[/bold green]")
    app_path = Path("app/streamlit_app.py").resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


def main() -> None:
    """Main CLI command dispatcher."""
    parser = argparse.ArgumentParser(
        description="Medical Research Assistant - Agentic RAG System Master CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: Ingest
    ingest_p = subparsers.add_parser("ingest", help="Ingest raw clinical PDFs and build vector databases")
    ingest_p.add_argument("--strategy", default="fixed", choices=["fixed", "parent_child", "semantic"])
    ingest_p.add_argument("--chunk-size", type=int, default=500, choices=[200, 500, 1000])
    ingest_p.add_argument("--vector-db", default="chromadb", choices=["chromadb", "faiss"])
    ingest_p.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    ingest_p.add_argument("--drive-url", type=str, default=None, help="Google Drive folder or file share URL to download and ingest")

    # Command: Query
    query_p = subparsers.add_parser("query", help="Ask a medical inquiry with grounding and citations")
    query_p.add_argument("query_text", type=str, help="Clinical inquiry or medical question")
    query_p.add_argument("--top-k", type=int, default=5, choices=[3, 5, 10])
    query_p.add_argument("--vector-db", default="chromadb", choices=["chromadb", "faiss"])
    query_p.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    query_p.add_argument("--pubmed", action="store_true", help="Force dynamic PubMed search")

    # Command: Benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run systematic RAGAS benchmarking matrix")
    bench_p.add_argument("--full", action="store_true", help="Run exhaustive full permutation matrix")

    # Command: Failures
    subparsers.add_parser("failures", help="Display logged failure modes and hallucinations")

    # Command: App
    subparsers.add_parser("app", help="Launch Streamlit Web App")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "failures":
        handle_failures(args)
    elif args.command == "app":
        handle_app(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
