"""LLM Multi-Agent System Module.

Features:
- QueryRouterAgent: Intent classifier, query optimizer, sub-query extractor,
  and dynamic metadata filter builder.
- VerifierAgent (Fact-Checker Verifier): Supervised NLI model (cross-encoder/nli-deberta-v3-small)
  evaluating answer claim entailment vs contradiction against retrieved context chunks.
  Detects hallucinations and low-grounding assertions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import CrossEncoder

from config import settings
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class QueryIntent(str, Enum):
    """Clinical query intent categories."""
    CLINICAL_GUIDELINE = "clinical_guideline"
    PHARMACOLOGY_DOSAGE = "pharmacology_dosage"
    DIAGNOSTIC_PROCEDURE = "diagnostic_procedure"
    RECENT_LITERATURE = "recent_literature"
    GENERAL_MEDICAL = "general_medical"


@dataclass
class RoutedQuery:
    """Output from the QueryRouterAgent."""
    original_query: str
    intent: QueryIntent
    optimized_query: str
    sub_queries: List[str]
    metadata_filters: Dict[str, Any]
    should_fetch_pubmed: bool
    target_topic_cluster: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "optimized_query": self.optimized_query,
            "sub_queries": self.sub_queries,
            "metadata_filters": self.metadata_filters,
            "should_fetch_pubmed": self.should_fetch_pubmed,
            "target_topic_cluster": self.target_topic_cluster,
        }


@dataclass
class ClaimVerification:
    """NLI grounding assessment for an individual generated assertion."""
    claim_text: str
    best_matching_chunk_id: Optional[str]
    status: str  # 'VERIFIED' (Entailment), 'CONTRADICTION', 'UNGROUNDED' (Neutral)
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    evidence_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "best_matching_chunk_id": self.best_matching_chunk_id,
            "status": self.status,
            "entailment_prob": round(self.entailment_prob, 3),
            "contradiction_prob": round(self.contradiction_prob, 3),
            "neutral_prob": round(self.neutral_prob, 3),
            "evidence_text": self.evidence_text[:120] + "..." if len(self.evidence_text) > 120 else self.evidence_text,
        }


@dataclass
class VerificationReport:
    """Complete Fact-Checker Verifier report."""
    total_claims: int
    verified_claims_count: int
    contradictions_count: int
    ungrounded_claims_count: int
    faithfulness_score: float  # verified / total_claims
    is_faithful: bool
    claim_details: List[ClaimVerification]
    detected_hallucinations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified_claims_count": self.verified_claims_count,
            "contradictions_count": self.contradictions_count,
            "ungrounded_claims_count": self.ungrounded_claims_count,
            "faithfulness_score": round(self.faithfulness_score, 3),
            "is_faithful": self.is_faithful,
            "claim_details": [c.to_dict() for c in self.claim_details],
            "detected_hallucinations": self.detected_hallucinations,
        }


# =============================================================================
# QUERY ROUTER AGENT
# =============================================================================

class QueryRouterAgent:
    """Analyzes incoming medical queries, determines intent, and builds targeted metadata filters."""

    PUBMED_TRIGGER_KEYWORDS = {
        "recent", "latest", "new", "trial", "study", "studies", "literature", "published",
        "pubmed", "efficacy", "clinical trial", "meta-analysis", "cohort", "2024", "2025", "2026",
    }

    PHARMACOLOGY_KEYWORDS = {
        "dose", "dosage", "drug", "medication", "prescribe", "treatment", "mg", "first-line",
        "adverse", "side effect", "contraindication", "metformin", "lisinopril", "ceftriaxone",
    }

    PROCEDURE_KEYWORDS = {
        "diagnosis", "diagnostic", "procedure", "ct scan", "mri", "ultrasound", "biopsy",
        "intubation", "ventilation", "ecg", "ekg", "blood test", "monitoring",
    }

    GUIDELINE_KEYWORDS = {
        "who", "cdc", "guideline", "protocol", "recommendation", "criteria", "management",
    }

    def route_query(self, query: str) -> RoutedQuery:
        """Route user query to appropriate pipeline pathways and construct metadata filters."""
        query_clean = query.strip()
        lower_q = query_clean.lower()

        # 1. Detect Intent
        intent = QueryIntent.GENERAL_MEDICAL
        should_fetch_pubmed = False
        target_cluster = None
        filters: Dict[str, Any] = {}

        if any(w in lower_q for w in self.PUBMED_TRIGGER_KEYWORDS):
            intent = QueryIntent.RECENT_LITERATURE
            should_fetch_pubmed = True
            filters["source_type"] = "PUBMED"
            target_cluster = "Pharmacology & Therapeutics"

        elif any(w in lower_q for w in self.PHARMACOLOGY_KEYWORDS):
            intent = QueryIntent.PHARMACOLOGY_DOSAGE
            target_cluster = "Pharmacology & Therapeutics"
            filters["cluster_label"] = "Pharmacology & Therapeutics"

        elif any(w in lower_q for w in self.PROCEDURE_KEYWORDS):
            intent = QueryIntent.DIAGNOSTIC_PROCEDURE
            target_cluster = "Laboratory & Procedures"

        elif any(w in lower_q for w in self.GUIDELINE_KEYWORDS):
            intent = QueryIntent.CLINICAL_GUIDELINE
            filters["source_type"] = "PDF"
            target_cluster = "Treatment Protocols & Guidelines"

        # 2. Query Optimization & Expansion
        optimized_query = self._optimize_query(query_clean)

        # 3. Sub-query extraction
        sub_queries = self._extract_sub_queries(query_clean, intent)

        logger.info(
            "QueryRouter: query='%s' -> intent=%s, pubmed=%s, filters=%s",
            query_clean[:40],
            intent.value,
            should_fetch_pubmed,
            filters,
        )

        return RoutedQuery(
            original_query=query_clean,
            intent=intent,
            optimized_query=optimized_query,
            sub_queries=sub_queries,
            metadata_filters=filters,
            should_fetch_pubmed=should_fetch_pubmed,
            target_topic_cluster=target_cluster,
        )

    def _optimize_query(self, query: str) -> str:
        """Expand acronyms and standardize medical terminology."""
        from src.ingestion import MedicalNLPProcessor
        for acronym, full in MedicalNLPProcessor.COMMON_ACRONYMS.items():
            pattern = rf"\b{re.escape(acronym)}\b"
            if re.search(pattern, query):
                query = re.sub(pattern, f"{acronym} ({full})", query)
        return query

    def _extract_sub_queries(self, query: str, intent: QueryIntent) -> List[str]:
        """Deconstruct multi-faceted queries into focused sub-queries."""
        sub_queries = [query]

        # Check for conjunctions 'and', 'or', 'along with'
        parts = re.split(r"\b(?:and|along with|as well as|versus|vs)\b", query, flags=re.IGNORECASE)
        if len(parts) > 1:
            for p in parts:
                clean_p = p.strip()
                if len(clean_p.split()) >= 2:
                    sub_queries.append(clean_p)

        if intent == QueryIntent.PHARMACOLOGY_DOSAGE:
            sub_queries.append(f"{query} starting dosage and administration")
        elif intent == QueryIntent.DIAGNOSTIC_PROCEDURE:
            sub_queries.append(f"{query} diagnostic imaging and monitoring criteria")

        return list(dict.fromkeys(sub_queries))[:3]


# =============================================================================
# VERIFIER AGENT (SUPERVISED NLI FACT-CHECKER)
# =============================================================================

class VerifierAgent:
    """Supervised NLI Grounding Verifier checking generated claims against context chunks."""

    _nli_model_instance: Optional[CrossEncoder] = None

    def __init__(
        self,
        model_name: str = settings.NLI_VERIFIER_MODEL,
        entailment_threshold: float = settings.NLI_ENTAILMENT_THRESHOLD,
        contradiction_threshold: float = settings.NLI_CONTRADICTION_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.model = self._load_model()

    def _load_model(self) -> CrossEncoder:
        if VerifierAgent._nli_model_instance is None:
            logger.info("Loading Supervised NLI Verifier model: %s", self.model_name)
            # nli-deberta-v3-small output labels: [contradiction, entailment, neutral]
            VerifierAgent._nli_model_instance = CrossEncoder(self.model_name)
        return VerifierAgent._nli_model_instance

    def _split_into_claims(self, text: str) -> List[str]:
        """Decompose generated response text into discrete claim sentences."""
        # Strip citation annotations [WHO Guidelines, Page 1] for clean NLI evaluation
        clean_text = re.sub(r"\[.*?\]", "", text)
        sentences = re.split(r"(?<=[.!?])\s+", clean_text.strip())
        claims = [s.strip() for s in sentences if len(s.strip().split()) >= 4]
        return claims

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probability distribution over [contradiction, entailment, neutral]."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    def verify_answer(
        self,
        generated_answer: str,
        context_chunks: List[RetrievedChunk],
    ) -> VerificationReport:
        """Verify all assertions in the generated answer against retrieved context chunks.

        Labels mapping for nli-deberta-v3-small:
            Index 0: Contradiction
            Index 1: Entailment
            Index 2: Neutral
        """
        claims = self._split_into_claims(generated_answer)
        if not claims or not context_chunks:
            return VerificationReport(
                total_claims=len(claims),
                verified_claims_count=0,
                contradictions_count=0,
                ungrounded_claims_count=len(claims),
                faithfulness_score=0.0 if claims else 1.0,
                is_faithful=False if claims else True,
                claim_details=[],
                detected_hallucinations=["No context chunks provided to verify assertions."] if claims else [],
            )

        claim_evaluations: List[ClaimVerification] = []
        hallucinations: List[str] = []

        # Extract focused premise units (sentences, paragraphs, and chunks) with chunk attribution
        candidate_premises: List[Tuple[str, RetrievedChunk]] = []
        for chunk in context_chunks:
            clean_c = re.sub(r"\[.*?\]", "", chunk.text)
            candidate_premises.append((clean_c, chunk))
            # Paragraph and table units
            for p in clean_c.splitlines():
                p_clean = p.strip()
                if "|" in p_clean:
                    parts = [pt.strip() for pt in p_clean.split("|") if pt.strip()]
                    if parts and not any(h in parts[0].lower() for h in ["condition", "parameter", "header", "---"]):
                        if len(parts) >= 3:
                            candidate_premises.append((f"For {parts[0]}, {parts[1]} starting dosage is {parts[2]}.", chunk))
                        candidate_premises.append((" ".join(parts), chunk))
                elif len(p_clean.split()) >= 4:
                    candidate_premises.append((p_clean, chunk))
            # Sentence units
            for s in re.split(r"(?<=[.!?])\s+", clean_c):
                s_clean = s.strip()
                if len(s_clean.split()) >= 4:
                    candidate_premises.append((s_clean, chunk))

        for claim in claims:
            # Score claim against candidate premises
            claim_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", claim.lower()))
            scored_premises = []
            for premise_text, chunk in candidate_premises:
                overlap = sum(1 for w in claim_words if w in premise_text.lower())
                scored_premises.append((overlap, premise_text, chunk))

            scored_premises.sort(key=lambda x: x[0], reverse=True)
            top_premises = scored_premises[:6] if scored_premises else [(0, context_chunks[0].text, context_chunks[0])]

            pairs = [[p_text, claim] for _, p_text, _ in top_premises]
            raw_logits = self.model.predict(pairs)
            probs = self._softmax(np.array(raw_logits))

            # Find best supporting premise (highest entailment probability)
            best_idx = int(np.argmax(probs[:, 1]))
            best_entailment = float(probs[best_idx, 1])
            best_contradiction = float(probs[best_idx, 0])
            best_neutral = float(probs[best_idx, 2])
            matching_chunk = top_premises[best_idx][2]
            best_premise_text = top_premises[best_idx][1]

            # Check for contradiction across evaluated premises
            max_contradiction_idx = int(np.argmax(probs[:, 0]))
            max_contradiction = float(probs[max_contradiction_idx, 0])

            if max_contradiction >= self.contradiction_threshold and max_contradiction > best_entailment:
                status = "CONTRADICTION"
                contradicting_chunk = top_premises[max_contradiction_idx][2]
                hallucinations.append(
                    f"Claim contradicted by {contradicting_chunk.source_name} (Page {contradicting_chunk.page_number}): '{claim}'"
                )
            elif best_entailment >= self.entailment_threshold:
                status = "VERIFIED"
            else:
                status = "UNGROUNDED"
                hallucinations.append(f"Ungrounded claim: '{claim}' (insufficient context evidence)")

            claim_evaluations.append(ClaimVerification(
                claim_text=claim,
                best_matching_chunk_id=matching_chunk.chunk_id,
                status=status,
                entailment_prob=best_entailment,
                contradiction_prob=best_contradiction,
                neutral_prob=best_neutral,
                evidence_text=best_premise_text,
            ))

        verified_count = sum(1 for c in claim_evaluations if c.status == "VERIFIED")
        contradiction_count = sum(1 for c in claim_evaluations if c.status == "CONTRADICTION")
        ungrounded_count = sum(1 for c in claim_evaluations if c.status == "UNGROUNDED")

        faithfulness = verified_count / len(claim_evaluations) if claim_evaluations else 1.0
        is_faithful = (contradiction_count == 0 and faithfulness >= 0.70)

        report = VerificationReport(
            total_claims=len(claim_evaluations),
            verified_claims_count=verified_count,
            contradictions_count=contradiction_count,
            ungrounded_claims_count=ungrounded_count,
            faithfulness_score=faithfulness,
            is_faithful=is_faithful,
            claim_details=claim_evaluations,
            detected_hallucinations=hallucinations,
        )

        logger.info(
            "VerifierReport: Faithfulness=%.2f (%d/%d claims verified, %d contradictions)",
            faithfulness,
            verified_count,
            len(claim_evaluations),
            contradiction_count,
        )
        return report
