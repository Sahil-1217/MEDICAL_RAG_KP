"""Unit and integration tests for STEP 2: Data Ingestion & Medical NLP Preprocessing."""

from pathlib import Path
import pytest
from src.ingestion import (
    MedicalNLPProcessor,
    MedicalEntity,
    PDFIngestor,
    PubMedIngestor,
    IngestedDocument,
)


@pytest.fixture
def nlp_processor():
    """Fixture providing initialized MedicalNLPProcessor instance."""
    return MedicalNLPProcessor()


@pytest.fixture
def pdf_ingestor(nlp_processor):
    """Fixture providing PDFIngestor instance."""
    return PDFIngestor(nlp_processor=nlp_processor)


def test_text_cleaning(nlp_processor):
    """Verify text cleaning removes non-printable characters and normalizes whitespace."""
    dirty_text = "Patient  has   severe \x00 pain\t\t and\n\n\n\nfever."
    cleaned = nlp_processor.clean_text(dirty_text)
    assert "\x00" not in cleaned
    assert "  " not in cleaned
    assert "Patient has severe pain and" in cleaned


def test_acronym_expansion(nlp_processor):
    """Verify medical acronym detection and expansion."""
    text = "Patient diagnosed with HTN and T2DM admitted to the ICU."
    _, acronyms = nlp_processor.expand_acronyms(text)
    assert "HTN" in acronyms
    assert acronyms["HTN"] == "Hypertension"
    assert "T2DM" in acronyms
    assert "Diabetes" in acronyms["T2DM"]
    assert "ICU" in acronyms


def test_biomedical_ner_extraction(nlp_processor):
    """Verify extraction and categorization of DISEASE, DRUG, DOSAGE, and PROCEDURE."""
    clinical_note = (
        "Patient with Type 2 Diabetes and Hypertension was prescribed "
        "Metformin 500mg orally daily and Lisinopril 10mg daily. "
        "A chest CT scan and electrocardiogram were ordered."
    )
    entities = nlp_processor.extract_entities(clinical_note)
    assert len(entities) > 0

    labels = {e.label for e in entities}
    assert "DRUG" in labels
    assert "DOSAGE" in labels
    assert "DISEASE" in labels or "ENTITY" in labels

    # Check specific extractions
    entity_texts = [e.text.lower() for e in entities]
    assert any("metformin" in t for t in entity_texts)
    assert any("500mg" in t for t in entity_texts)


def test_metadata_enrichment(nlp_processor):
    """Verify enriched metadata structure."""
    text = "Treatment for COVID-19 includes Dexamethasone 6mg daily."
    enriched = nlp_processor.enrich_metadata(text, {"source": "CDC_TEST"})
    assert enriched["source"] == "CDC_TEST"
    assert "entity_summary" in enriched
    assert "acronyms" in enriched
    assert enriched["entity_summary"]["total_entities"] >= 1


def test_pdf_ingestor_who(pdf_ingestor):
    """Verify layout-preserving PDF extraction and table parsing on WHO document."""
    who_pdf = Path("data/raw_pdfs/WHO_Guideline_Hypertension_and_Diabetes.pdf")
    assert who_pdf.exists(), "Sample WHO PDF must be generated before test"

    doc = pdf_ingestor.parse_pdf(who_pdf)
    assert isinstance(doc, IngestedDocument)
    assert doc.source_type == "PDF"
    assert len(doc.pages) == 2
    assert "Metformin" in doc.full_text
    assert "Lisinopril" in doc.full_text

    # Page 1 must have extracted table
    page1 = doc.pages[0]
    assert len(page1.tables) >= 1
    assert any("Type 2 Diabetes" in str(cell) for row in page1.tables[0] for cell in row)


def test_pdf_ingestor_cdc(pdf_ingestor):
    """Verify layout-preserving PDF extraction and table parsing on CDC document."""
    cdc_pdf = Path("data/raw_pdfs/CDC_Clinical_Procedure_Pneumonia_and_Sepsis.pdf")
    assert cdc_pdf.exists(), "Sample CDC PDF must be generated before test"

    doc = pdf_ingestor.parse_pdf(cdc_pdf)
    assert isinstance(doc, IngestedDocument)
    assert len(doc.pages) >= 1
    assert "Ceftriaxone" in doc.full_text
    assert "Dexamethasone" in doc.full_text
    assert len(doc.pages[0].tables) >= 1


def test_pubmed_xml_parsing(nlp_processor):
    """Verify PubMed XML record parser with authentic Medline XML structure."""
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>38000001</PMID>
          <Article>
            <ArticleTitle>Efficacy of Metformin in Type 2 Diabetes Management</ArticleTitle>
            <Journal>
              <Title>Journal of Clinical Endocrinology</Title>
              <JournalIssue>
                <PubDate><Year>2024</Year></PubDate>
              </JournalIssue>
            </Journal>
            <Abstract>
              <AbstractText Label="BACKGROUND">Metformin remains the initial drug of choice for type 2 diabetes mellitus.</AbstractText>
              <AbstractText Label="RESULTS">Significantly lowered HbA1c with low hypoglycemia risk.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>John</ForeName>
              </Author>
            </AuthorList>
          </Article>
          <MeshHeadingList>
            <MeshHeading><DescriptorName>Diabetes Mellitus, Type 2</DescriptorName></MeshHeading>
            <MeshHeading><DescriptorName>Metformin</DescriptorName></MeshHeading>
          </MeshHeadingList>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    ingestor = PubMedIngestor(nlp_processor=nlp_processor)
    docs = ingestor._parse_pubmed_xml(sample_xml)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "pmid_38000001"
    assert doc.metadata["pmid"] == "38000001"
    assert "Metformin remains the initial drug" in doc.full_text
    assert doc.metadata["journal"] == "Journal of Clinical Endocrinology"
    assert "Diabetes Mellitus, Type 2" in doc.metadata["mesh_terms"]
