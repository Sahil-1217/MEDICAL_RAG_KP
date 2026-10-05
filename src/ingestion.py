"""Data Ingestion & Medical NLP Preprocessing Module.

Features:
- PDFIngestor: Layout-aware PDF parser extracting text, section headers, page numbers,
  and structured tables using pdfplumber.
- PubMedIngestor: Dynamic client fetching peer-reviewed PubMed abstracts via Biopython Entrez.
- MedicalNLPProcessor: scispacy biomedical NER ('en_core_sci_sm'), medical entity categorization
  (DISEASE, DRUG, DOSAGE, PROCEDURE), acronym expansion, and metadata enrichment.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pdfplumber
from Bio import Entrez

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MedicalEntity:
    """Represents a biomedical entity extracted via NER or rule-based matching."""
    text: str
    label: str  # DISEASE, DRUG, DOSAGE, PROCEDURE, or ENTITY
    start_char: int
    end_char: int
    normalized_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "normalized_text": self.normalized_text or self.text.lower(),
        }


@dataclass
class PageContent:
    """Extracted content from an individual document page."""
    page_number: int
    text: str
    section_headers: List[str] = field(default_factory=list)
    tables: List[List[List[Optional[str]]]] = field(default_factory=list)
    raw_character_count: int = 0


@dataclass
class IngestedDocument:
    """Complete document object representing a parsed PDF or PubMed record."""
    doc_id: str
    source_name: str
    source_type: str  # 'PDF' (WHO/CDC/Clinical) or 'PUBMED'
    pages: List[PageContent]
    full_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: List[MedicalEntity] = field(default_factory=list)


# =============================================================================
# MEDICAL NLP PROCESSOR
# =============================================================================

class MedicalNLPProcessor:
    """Biomedical NLP processor leveraging scispacy, regex patterns, and acronym expansion."""

    # Curated medical acronym dictionary for normalization
    COMMON_ACRONYMS: Dict[str, str] = {
        "DM": "Diabetes Mellitus",
        "T2D": "Type 2 Diabetes",
        "T2DM": "Type 2 Diabetes Mellitus",
        "T1D": "Type 1 Diabetes",
        "T1DM": "Type 1 Diabetes Mellitus",
        "HTN": "Hypertension",
        "CAD": "Coronary Artery Disease",
        "CHF": "Congestive Heart Failure",
        "COPD": "Chronic Obstructive Pulmonary Disease",
        "ARDS": "Acute Respiratory Distress Syndrome",
        "CKD": "Chronic Kidney Disease",
        "ESRD": "End-Stage Renal Disease",
        "MI": "Myocardial Infarction",
        "CVA": "Cerebrovascular Accident (Stroke)",
        "DVT": "Deep Vein Thrombosis",
        "PE": "Pulmonary Embolism",
        "PCR": "Polymerase Chain Reaction",
        "RT-PCR": "Reverse Transcription Polymerase Chain Reaction",
        "ICU": "Intensive Care Unit",
        "ECG": "Electrocardiogram",
        "EKG": "Electrocardiogram",
        "MRI": "Magnetic Resonance Imaging",
        "CT": "Computed Tomography",
        "CBC": "Complete Blood Count",
        "NSAID": "Nonsteroidal Anti-inflammatory Drug",
        "ACEI": "Angiotensin-Converting Enzyme Inhibitor",
        "ARB": "Angiotensin II Receptor Blocker",
        "WHO": "World Health Organization",
        "CDC": "Centers for Disease Control and Prevention",
    }

    # Regex patterns for dosage detection
    DOSAGE_PATTERN = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg|ml|mL|IU|units|meq|mmol|mg/kg|mg/m2|mg/dL)\b"
        r"(?:\s*(?:daily|twice daily|bid|tid|qid|q\d+h|prn|po|iv|subq|intravenous|orally))?",
        re.IGNORECASE,
    )

    # Keywords for procedure detection
    PROCEDURE_KEYWORDS = {
        "biopsy", "endoscopy", "colonoscopy", "bronchoscopy", "intubation", "catheterization",
        "angioplasty", "stenting", "dialysis", "hemodialysis", "lumbar puncture", "thoracentesis",
        "paracentesis", "laparoscopy", "laparotomy", "resection", "excision", "mastectomy",
        "ultrasound", "echocardiogram", "radiography", "ct scan", "mri", "pet scan", "tracheostomy",
        "mechanical ventilation", "immunization", "vaccination", "chemotherapy", "radiation therapy",
    }

    # Keywords for common drugs and therapeutic agents
    DRUG_KEYWORDS = {
        "metformin", "insulin", "lisinopril", "amlodipine", "atorvastatin", "simvastatin",
        "levothyroxine", "albuterol", "omeprazole", "losartan", "gabapentin", "hydrochlorothiazide",
        "sertraline", "amoxicillin", "azithromycin", "ciprofloxacin", "doxycycline", "prednisone",
        "dexamethasone", "aspirin", "clopidogrel", "heparin", "warfarin", "apixaban", "enoxaparin",
        "remdesivir", "paxlovid", "nirmatrelvir", "ritonavir", "paracetamol", "acetaminophen",
        "ibuprofen", "naproxen", "morphine", "fentanyl", "propofol", "midazolam",
    }

    # Keywords for diseases and medical conditions
    DISEASE_KEYWORDS = {
        "diabetes", "hypertension", "pneumonia", "covid-19", "coronavirus", "sars-cov-2",
        "tuberculosis", "malaria", "influenza", "asthma", "bronchitis", "sepsis", "septic shock",
        "stroke", "infarction", "arrhythmia", "fibrillation", "carcinoma", "lymphoma", "leukemia",
        "hepatitis", "cirrhosis", "nephritis", "pancreatitis", "meningitis", "encephalitis",
    }

    def __init__(self, model_name: str = settings.SCISPACY_MODEL_NAME) -> None:
        """Initialize the biomedical NLP model."""
        self.model_name = model_name
        self.nlp = None
        self._load_model()

    def _load_model(self) -> None:
        """Load scispacy model with fallback to basic tokenization if unavailable."""
        import spacy
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info("Successfully loaded biomedical NER model: %s", self.model_name)
        except Exception as e:
            logger.warning("Could not load '%s' directly (%s). Attempting spaCy blank 'en'...", self.model_name, e)
            try:
                self.nlp = spacy.blank("en")
                logger.info("Loaded fallback blank English spaCy pipeline.")
            except Exception as ex:
                logger.error("Failed to initialize spaCy: %s", ex)
                self.nlp = None

    def clean_text(self, text: str) -> str:
        """Normalize whitespace, remove artifacts, join wrapped lines, and standardize punctuation."""
        if not text:
            return ""
        # Remove null bytes and non-printable control chars except \n and \t
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        # Normalize continuous whitespace within lines
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]

        # Join line-wrapped sentences within paragraphs while preserving section headers & tables
        merged_blocks: List[str] = []
        current_p: List[str] = []

        for line in lines:
            if not line:
                if current_p:
                    merged_blocks.append(" ".join(current_p))
                    current_p = []
            else:
                # If line is header, table, or bullet point, flush current paragraph
                is_header_or_table = (
                    line.isupper()
                    or line.startswith("|")
                    or line.startswith("[Table")
                    or re.match(r"^\d+\.\s+", line)
                    or "---" in line
                )
                if is_header_or_table:
                    if current_p:
                        merged_blocks.append(" ".join(current_p))
                        current_p = []
                    merged_blocks.append(line)
                else:
                    current_p.append(line)

        if current_p:
            merged_blocks.append(" ".join(current_p))

        return "\n\n".join(merged_blocks).strip()

    def expand_acronyms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Identify medical acronyms in text and return expanded mapping."""
        found_acronyms: Dict[str, str] = {}
        for acronym, expansion in self.COMMON_ACRONYMS.items():
            pattern = rf"\b{re.escape(acronym)}\b"
            if re.search(pattern, text):
                found_acronyms[acronym] = expansion
        return text, found_acronyms

    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract biomedical entities (DISEASE, DRUG, DOSAGE, PROCEDURE)."""
        entities: List[MedicalEntity] = []
        seen_spans = set()

        # 1. Regex Extraction for DOSAGE
        for match in self.DOSAGE_PATTERN.finditer(text):
            span = (match.start(), match.end())
            if span not in seen_spans:
                seen_spans.add(span)
                entities.append(MedicalEntity(
                    text=match.group(),
                    label="DOSAGE",
                    start_char=match.start(),
                    end_char=match.end(),
                    normalized_text=match.group().strip().lower(),
                ))

        # 2. scispacy NER model extraction
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    span = (ent.start_char, ent.end_char)
                    if span in seen_spans:
                        continue

                    raw_ent_text = ent.text.strip()
                    lower_ent = raw_ent_text.lower()
                    assigned_label = "ENTITY"

                    # Classify into specific medical rubrics
                    if any(drug in lower_ent for drug in self.DRUG_KEYWORDS):
                        assigned_label = "DRUG"
                    elif any(disease in lower_ent for disease in self.DISEASE_KEYWORDS):
                        assigned_label = "DISEASE"
                    elif any(proc in lower_ent for proc in self.PROCEDURE_KEYWORDS):
                        assigned_label = "PROCEDURE"
                    elif ent.label_ in {"DISEASE", "CHEMICAL", "DRUG"}:
                        assigned_label = "DRUG" if ent.label_ == "CHEMICAL" else "DISEASE"

                    seen_spans.add(span)
                    entities.append(MedicalEntity(
                        text=raw_ent_text,
                        label=assigned_label,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        normalized_text=lower_ent,
                    ))
            except Exception as e:
                logger.debug("scispacy inference warning: %s", e)

        # 3. Rule-based fallback keywords if model missed any key terms
        for keyword in self.PROCEDURE_KEYWORDS:
            for match in re.finditer(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append(MedicalEntity(
                        text=match.group(),
                        label="PROCEDURE",
                        start_char=match.start(),
                        end_char=match.end(),
                        normalized_text=keyword.lower(),
                    ))

        for drug in self.DRUG_KEYWORDS:
            for match in re.finditer(rf"\b{re.escape(drug)}\b", text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append(MedicalEntity(
                        text=match.group(),
                        label="DRUG",
                        start_char=match.start(),
                        end_char=match.end(),
                        normalized_text=drug.lower(),
                    ))

        for disease in self.DISEASE_KEYWORDS:
            for match in re.finditer(rf"\b{re.escape(disease)}\b", text, re.IGNORECASE):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append(MedicalEntity(
                        text=match.group(),
                        label="DISEASE",
                        start_char=match.start(),
                        end_char=match.end(),
                        normalized_text=disease.lower(),
                    ))

        # Sort entities by appearance in text
        entities.sort(key=lambda e: e.start_char)
        return entities

    def enrich_metadata(self, text: str, initial_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Produce comprehensive NLP metadata dictionary for text."""
        metadata = dict(initial_metadata or {})
        cleaned = self.clean_text(text)
        _, acronyms = self.expand_acronyms(cleaned)
        entities = self.extract_entities(cleaned)

        entity_summary = {
            "diseases": list({e.text for e in entities if e.label == "DISEASE"}),
            "drugs": list({e.text for e in entities if e.label == "DRUG"}),
            "dosages": list({e.text for e in entities if e.label == "DOSAGE"}),
            "procedures": list({e.text for e in entities if e.label == "PROCEDURE"}),
            "total_entities": len(entities),
        }

        metadata.update({
            "acronyms": acronyms,
            "entities": [e.to_dict() for e in entities],
            "entity_summary": entity_summary,
            "cleaned_text_length": len(cleaned),
        })
        return metadata


# =============================================================================
# PDF INGESTOR
# =============================================================================

class PDFIngestor:
    """Extracts raw text, structured section headers, page numbers, and tables from PDFs."""

    # Patterns for detecting section headers
    HEADER_REGEX = re.compile(
        r"^(?:[A-Z0-9\.\-\s]{3,60}|(?:Section|Chapter|Guideline|Protocol|Part)\s+\d+[:\.\-]?\s*.+|"
        r"(?:Abstract|Introduction|Methods|Results|Discussion|Conclusion|Recommendations|"
        r"Dosage\s+and\s+Administration|Contraindications|Adverse\s+Reactions|Clinical\s+Studies|"
        r"Epidemiology|Diagnosis|Treatment|Management|References)[:\.\-]?)$",
        re.IGNORECASE,
    )

    def __init__(self, nlp_processor: Optional[MedicalNLPProcessor] = None) -> None:
        self.nlp = nlp_processor or MedicalNLPProcessor()

    def parse_pdf(self, file_path: Union[str, Path]) -> Optional[IngestedDocument]:
        """Parse a PDF document with layout preservation and table extraction.

        Safely catches corrupted, truncated, or unreadable files and skips them gracefully.
        """
        path = Path(file_path)
        if not path.is_file():
            logger.warning("PDF file not found: %s", path)
            return None

        logger.info("Parsing PDF document: %s", path.name)
        pages_content: List[PageContent] = []
        aggregated_text_parts: List[str] = []

        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                if total_pages == 0:
                    logger.warning("PDF document is empty (0 pages): %s", path.name)
                    return None

                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1

                    # 1. Extract text preserving layout
                    raw_text = page.extract_text(layout=True) or ""
                    cleaned_text = self.nlp.clean_text(raw_text)

                    # 2. Extract tables
                    extracted_tables: List[List[List[Optional[str]]]] = []
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                # Filter empty rows
                                cleaned_table = [
                                    [cell.strip() if cell else None for cell in row]
                                    for row in table if any(row)
                                ]
                                if cleaned_table:
                                    extracted_tables.append(cleaned_table)
                    except Exception as ex:
                        logger.debug("Table extraction error on page %d: %s", page_num, ex)

                    # 3. Detect section headers
                    section_headers = self._detect_headers(cleaned_text)

                    # Append formatted table representations to text for RAG indexing
                    if extracted_tables:
                        table_texts = []
                        for t_idx, tbl in enumerate(extracted_tables):
                            table_texts.append(f"\n[Table {t_idx + 1} - Page {page_num}]")
                            for row in tbl:
                                filtered_cells = [str(c) if c is not None else "" for c in row]
                                table_texts.append(" | ".join(filtered_cells))
                        cleaned_text += "\n" + "\n".join(table_texts)

                    page_obj = PageContent(
                        page_number=page_num,
                        text=cleaned_text,
                        section_headers=section_headers,
                        tables=extracted_tables,
                        raw_character_count=len(cleaned_text),
                    )
                    pages_content.append(page_obj)
                    aggregated_text_parts.append(f"--- PAGE {page_num} ---\n{cleaned_text}")

        except Exception as e:
            logger.error("Failed to parse PDF '%s' (corrupted, damaged, or truncated download: %s). Skipping file.", path.name, e)
            return None

        full_text = "\n\n".join(aggregated_text_parts)
        if not full_text.strip():
            logger.warning("PDF has no extractable text: %s", path.name)
            return None

        doc_id = f"pdf_{path.stem}"

        # Global document metadata
        initial_meta = {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "total_pages": total_pages,
            "source_type": "PDF",
        }
        enriched_meta = self.nlp.enrich_metadata(full_text[:5000], initial_meta)

        return IngestedDocument(
            doc_id=doc_id,
            source_name=path.name,
            source_type="PDF",
            pages=pages_content,
            full_text=full_text,
            metadata=enriched_meta,
        )

    def _detect_headers(self, text: str) -> List[str]:
        """Heuristically identify section headers from text lines."""
        headers: List[str] = []
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str or len(line_str) > 80:
                continue
            # Check uppercase or specific header regex
            if line_str.isupper() and len(line_str.split()) <= 6:
                headers.append(line_str)
            elif self.HEADER_REGEX.match(line_str):
                headers.append(line_str)
            elif re.match(r"^\d+(\.\d+)*\s+[A-Z]", line_str):
                headers.append(line_str)
        return list(dict.fromkeys(headers))  # Preserve order, unique


# =============================================================================
# PUBMED INGESTOR (BIOPYTHON ENTREZ)
# =============================================================================

class PubMedIngestor:
    """Connects to NCBI PubMed Entrez API to dynamically retrieve peer-reviewed medical abstracts."""

    def __init__(
        self,
        email: str = settings.NCBI_EMAIL,
        api_key: Optional[str] = settings.NCBI_API_KEY,
        nlp_processor: Optional[MedicalNLPProcessor] = None,
    ) -> None:
        self.email = email
        self.api_key = api_key
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key
        self.nlp = nlp_processor or MedicalNLPProcessor()

    def _clean_query_for_pubmed(self, query: str) -> str:
        """Strip conversational filler words so NCBI Entrez matches clinical keywords."""
        stop_words = {
            "what", "is", "the", "are", "of", "for", "in", "to", "and", "a", "an", "recommended",
            "how", "do", "you", "can", "please", "tell", "me", "about",
        }
        words = re.findall(r"\b[a-zA-Z0-9\-_]+\b", query)
        filtered = [w for w in words if w.lower() not in stop_words]
        if filtered:
            return " ".join(filtered)
        return query

    def search_and_fetch(self, query: str, max_results: int = 5) -> List[IngestedDocument]:
        """Search PubMed by query string and retrieve full metadata and abstracts.

        Args:
            query: Medical search query (e.g. 'COVID-19 dexamethasone clinical trial').
            max_results: Maximum number of PubMed citations to retrieve.

        Returns:
            List of IngestedDocument objects.
        """
        clean_search_term = self._clean_query_for_pubmed(query)
        logger.info("Executing PubMed Entrez search: '%s' (cleaned: '%s', max_results=%d)", query, clean_search_term, max_results)
        documents: List[IngestedDocument] = []

        try:
            # 1. Search PubMed for PMIDs using cleaned term, fallback to original query
            search_handle = Entrez.esearch(db="pubmed", term=clean_search_term, retmax=max_results, sort="relevance")
            search_results = Entrez.read(search_handle)
            search_handle.close()

            pmid_list = search_results.get("IdList", [])
            if not pmid_list and clean_search_term != query:
                logger.info("No hits with cleaned term. Trying raw query on PubMed: %s", query)
                search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
                search_results = Entrez.read(search_handle)
                search_handle.close()
                pmid_list = search_results.get("IdList", [])

            if not pmid_list:
                logger.info("No PubMed records found for query: %s", query)
                return []

            logger.info("Found %d PubMed IDs: %s. Fetching records...", len(pmid_list), pmid_list)

            # 2. Fetch full XML records for PMIDs
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), rettype="xml", retmode="xml")
            xml_data = fetch_handle.read()
            fetch_handle.close()

            # 3. Parse XML records
            documents = self._parse_pubmed_xml(xml_data)

        except Exception as e:
            logger.error("Error communicating with PubMed Entrez API: %s", e)

        return documents

    def _parse_pubmed_xml(self, xml_bytes: Union[str, bytes]) -> List[IngestedDocument]:
        """Parse raw XML output from PubMed efetch into IngestedDocument objects."""
        documents: List[IngestedDocument] = []
        try:
            if isinstance(xml_bytes, str):
                xml_bytes = xml_bytes.encode("utf-8")
            root = ET.fromstring(xml_bytes)

            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//MedlineCitation/PMID")
                pmid = pmid_elem.text if pmid_elem is not None else "UnknownPMID"

                # Title
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "Untitled PubMed Article"

                # Abstract
                abstract_texts = []
                for abs_elem in article.findall(".//Abstract/AbstractText"):
                    label = abs_elem.get("Label")
                    text_content = abs_elem.text or ""
                    if label:
                        abstract_texts.append(f"{label}: {text_content}")
                    else:
                        abstract_texts.append(text_content)
                abstract = "\n".join(abstract_texts).strip()

                if not abstract:
                    abstract = "(No abstract available in PubMed record)"

                # Journal
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else "Medical Journal"

                # Publication Date
                pub_year_elem = article.find(".//JournalIssue/PubDate/Year")
                pub_year = pub_year_elem.text if pub_year_elem is not None else ""

                # Authors
                author_names = []
                for author in article.findall(".//AuthorList/Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    if last is not None and last.text:
                        name = f"{last.text}"
                        if fore is not None and fore.text:
                            name = f"{fore.text} {name}"
                        author_names.append(name)

                # MeSH Headings
                mesh_terms = []
                for mesh in article.findall(".//MeshHeading/DescriptorName"):
                    if mesh.text:
                        mesh_terms.append(mesh.text)

                combined_content = f"Title: {title}\nJournal: {journal} ({pub_year})\nAuthors: {', '.join(author_names[:5])}\n\nAbstract:\n{abstract}"
                cleaned_text = self.nlp.clean_text(combined_content)

                page_obj = PageContent(
                    page_number=1,
                    text=cleaned_text,
                    section_headers=[title, "Abstract"],
                    tables=[],
                    raw_character_count=len(cleaned_text),
                )

                doc_meta = {
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "pub_year": pub_year,
                    "authors": author_names,
                    "mesh_terms": mesh_terms,
                    "source_type": "PUBMED",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                }
                enriched_meta = self.nlp.enrich_metadata(cleaned_text, doc_meta)

                doc = IngestedDocument(
                    doc_id=f"pmid_{pmid}",
                    source_name=f"PubMed: {title[:50]}...",
                    source_type="PUBMED",
                    pages=[page_obj],
                    full_text=cleaned_text,
                    metadata=enriched_meta,
                )
                documents.append(doc)

        except Exception as e:
            logger.error("Error parsing PubMed XML: %s", e)

        return documents


# =============================================================================
# GOOGLE DRIVE INGESTOR
# =============================================================================

class GoogleDriveIngestor:
    """Downloads healthcare PDFs or research paper folders directly from Google Drive share links."""

    def __init__(self, download_dir: Union[str, Path] = settings.RAW_PDF_DIR) -> None:
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def download_from_url(self, url: str) -> List[Path]:
        """Download single PDF file or entire folder from a Google Drive share link.

        Supports:
        - Public Google Drive Folder: https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
        - Public Google Drive File: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
        - Shortened or standard share URLs

        Returns:
            List of downloaded PDF file Paths.
        """
        import gdown

        clean_url = url.strip()
        if not clean_url:
            raise ValueError("Google Drive URL cannot be empty.")

        logger.info("Initiating Google Drive download from URL: %s", clean_url)
        downloaded_paths: List[Path] = []

        try:
            # Check if URL represents a folder
            is_folder = "folders/" in clean_url or ("/drive/u/" in clean_url and "folders" in clean_url)

            if is_folder:
                logger.info("Detected Google Drive folder link. Downloading folder contents to %s...", self.download_dir)
                downloaded = gdown.download_folder(
                    url=clean_url,
                    output=str(self.download_dir),
                    quiet=False,
                    use_cookies=False,
                )
                if downloaded:
                    downloaded_paths = [Path(p) for p in downloaded if str(p).lower().endswith(".pdf")]
            else:
                logger.info("Detected Google Drive file link. Downloading document to %s...", self.download_dir)
                output_file = gdown.download(
                    url=clean_url,
                    output=str(self.download_dir) + "/",
                    quiet=False,
                    fuzzy=True,
                )
                if output_file and Path(output_file).is_file():
                    downloaded_paths.append(Path(output_file))

            # Scan download_dir for all PDFs in case folder placed them in subdirectories
            for pdf_file in self.download_dir.rglob("*.pdf"):
                if pdf_file not in downloaded_paths:
                    downloaded_paths.append(pdf_file)

            logger.info("Google Drive ingestion completed. Total PDFs retrieved: %d", len(downloaded_paths))
            return downloaded_paths

        except Exception as e:
            logger.error("Error downloading from Google Drive: %s", e)
            raise RuntimeError(f"Google Drive download failed: {e}")
