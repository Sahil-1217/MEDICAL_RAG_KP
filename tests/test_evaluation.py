"""Unit and integration tests for STEP 7: RAGAS Evaluation Engine and Benchmark Runner."""

from pathlib import Path
import pytest
from src.evaluation import (
    EvaluationEngine,
    EvaluationSample,
    FailureModeLogger,
    BenchmarkRunner,
    MetricScores,
)
from src.retrieval import RetrievedChunk


@pytest.fixture
def eval_sample():
    """Fixture providing a standard clinical evaluation sample."""
    return EvaluationSample(
        question="What is the recommended initial dosage of Metformin for Type 2 Diabetes?",
        ground_truth_answer="First-line pharmacological management for Type 2 Diabetes is Metformin initiated at 500mg orally once daily.",
        ground_truth_facts=[
            "Metformin initiated at 500mg orally once daily",
            "First-line management for Type 2 Diabetes",
        ],
        category="Diabetes",
    )


@pytest.fixture
def sample_context():
    """Fixture providing retrieved context chunks."""
    c = RetrievedChunk(
        chunk_id="chunk_test_who_01",
        text="First-line pharmacological management for Type 2 Diabetes is Metformin initiated at 500mg orally once daily, titrated up to 1000mg twice daily with meals.",
        dense_score=0.92,
        sparse_score=0.88,
        rrf_score=0.95,
        rerank_score=0.98,
        source_name="WHO_Guideline_Hypertension_and_Diabetes.pdf",
        page_number=1,
    )
    return [c]


def test_failure_mode_logger(tmp_path):
    """Verify FailureModeLogger logs, persists, and exports markdown tables."""
    logger = FailureModeLogger(log_dir=tmp_path)
    r1 = logger.log_retrieval_failure(
        query="Treatment for rare pediatric condition",
        subtype="ZERO_HITS",
        details="No documents matched threshold",
    )
    assert r1.failure_id.startswith("ret_fail_")

    r2 = logger.log_hallucination_failure(
        query="Diabetes treatment",
        subtype="CONTRADICTION",
        details="Claim contradicted by WHO Guideline",
        evidence="WHO Guideline snippet",
    )
    assert r2.failure_id.startswith("hal_fail_")

    # Verify JSON persistence
    assert (tmp_path / "failure_log.json").exists()

    # Verify Markdown export
    md = logger.export_markdown_summary()
    assert "Failure Mode Audit Log" in md
    assert "ret_fail_" in md
    assert "hal_fail_" in md


def test_evaluation_engine_metrics(eval_sample, sample_context):
    """Verify RAGAS metrics computation produces valid normalized scores."""
    engine = EvaluationEngine()
    generated_answer = "Metformin is the first-line medication for type 2 diabetes started at 500mg once daily [WHO_Guideline, Page 1]."

    scores = engine.evaluate_sample(
        sample=eval_sample,
        generated_answer=generated_answer,
        retrieved_chunks=sample_context,
        latency=0.45,
    )

    assert isinstance(scores, MetricScores)
    assert 0.0 <= scores.faithfulness <= 1.0
    assert 0.0 <= scores.answer_relevancy <= 1.0
    assert 0.0 <= scores.context_recall <= 1.0
    assert scores.faithfulness >= 0.70  # Fully grounded answer should score high
    assert scores.context_recall >= 0.50
    assert scores.latency_seconds == 0.45


def test_benchmark_runner_quick_mode(tmp_path):
    """Verify BenchmarkRunner executes representative matrix and exports reports."""
    runner = BenchmarkRunner(output_dir=tmp_path)
    output = runner.run_benchmark_matrix(quick_mode=True)

    assert output["total_experiments"] >= 1
    assert Path(output["markdown_report_path"]).exists()
    assert Path(output["csv_report_path"]).exists()

    # Verify Markdown report content
    report_text = Path(output["markdown_report_path"]).read_text(encoding="utf-8")
    assert "Experimental Matrix Results" in report_text
    assert "Architectural Insights" in report_text
