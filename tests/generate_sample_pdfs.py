"""Script to generate realistic WHO and CDC clinical guidance PDFs for testing."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_who_guideline_pdf(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#003366'),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#004080'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = styles['Normal']
    body_style.fontSize = 10
    body_style.leading = 14

    story = []

    # Title & Metadata
    story.append(Paragraph("World Health Organization (WHO) Guidelines", title_style))
    story.append(Paragraph("<b>Clinical Management of Hypertension and Type 2 Diabetes Mellitus</b>", h2_style))
    story.append(Paragraph("<b>Publication Date:</b> 2024 | <b>Document Reference:</b> WHO/NCD/2024.1", body_style))
    story.append(Spacer(1, 14))

    # Page 1: Introduction & Epidemiology
    story.append(Paragraph("<b>1. Introduction and Epidemiology</b>", h2_style))
    story.append(Paragraph(
        "Hypertension (HTN) and Type 2 Diabetes Mellitus (T2DM) are major contributors to cardiovascular disease (CVD) "
        "and chronic kidney disease (CKD) worldwide. Early screening with blood pressure measurements and fasting blood glucose "
        "or glycated hemoglobin (HbA1c) is essential. The target systolic blood pressure for adults with diagnosed HTN is less than 130 mmHg, "
        "and diastolic blood pressure should be maintained below 80 mmHg.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Pharmacological Management
    story.append(Paragraph("<b>2. Pharmacological Treatment Protocols</b>", h2_style))
    story.append(Paragraph(
        "First-line pharmacological management for Type 2 Diabetes is Metformin initiated at 500mg orally once daily, "
        "titrated up to 1000mg twice daily with meals to minimize gastrointestinal discomfort. "
        "For hypertension, first-line therapy includes Angiotensin-Converting Enzyme Inhibitors (ACEI) such as Lisinopril 10mg daily, "
        "or Angiotensin II Receptor Blockers (ARB) such as Losartan 50mg daily. Calcium channel blockers like Amlodipine 5mg daily "
        "are recommended for patients requiring dual combination therapy.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Table of Medications
    table_data = [
        ["Condition", "First-Line Medication", "Starting Dosage", "Target Maximum"],
        ["Type 2 Diabetes", "Metformin", "500mg orally daily", "2000mg daily"],
        ["Hypertension", "Lisinopril", "10mg orally daily", "40mg daily"],
        ["Hypertension", "Amlodipine", "5mg orally daily", "10mg daily"],
        ["Dyslipidemia", "Atorvastatin", "20mg orally daily", "80mg daily"]
    ]
    t = Table(table_data, colWidths=[110, 120, 130, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#004080')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Page Break
    story.append(PageBreak())

    # Page 2: Diagnostic Procedures & Follow-up
    story.append(Paragraph("<b>3. Diagnostic Procedures and Monitoring</b>", h2_style))
    story.append(Paragraph(
        "Patients diagnosed with hypertension and diabetes must undergo annual electrocardiogram (ECG), "
        "fundoscopic retinal examination to evaluate for diabetic retinopathy, and urinary albumin-to-creatinine ratio (uACR) "
        "to detect early diabetic nephropathy. If renal failure is suspected, renal ultrasound and estimated glomerular filtration rate (eGFR) "
        "must be performed immediately.",
        body_style
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>4. Clinical Recommendations & Lifestyle</b>", h2_style))
    story.append(Paragraph(
        "Lifestyle interventions remain the cornerstone of therapy: sodium intake should be reduced to less than 2g/day (equivalent to 5g salt), "
        "moderate physical activity of at least 150 minutes per week is strongly advised, and smoking cessation protocols must be enacted.",
        body_style
    ))

    doc.build(story)
    print(f"Created WHO PDF: {output_path}")


def create_cdc_pneumonia_pdf(output_path: Path):
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#800000'),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#990000'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = styles['Normal']
    body_style.fontSize = 10
    body_style.leading = 14

    story = []

    story.append(Paragraph("Centers for Disease Control and Prevention (CDC)", title_style))
    story.append(Paragraph("<b>Clinical Protocol: Severe Pneumonia, COVID-19, and Sepsis Management</b>", h2_style))
    story.append(Paragraph("<b>Reference:</b> CDC-ID-2024-08 | Clinical Procedure Bulletin", body_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>1. Emergency Triage and Diagnostic Imaging</b>", h2_style))
    story.append(Paragraph(
        "Patients presenting with acute respiratory distress, fever, and hypoxemia must be evaluated for community-acquired pneumonia (CAP) "
        "or viral pneumonia including SARS-CoV-2 (COVID-19). Diagnostic evaluation includes chest computed tomography (CT scan), "
        "arterial blood gas (ABG) analysis, and blood cultures prior to antibiotic initiation. "
        "In severe hypoxemic respiratory failure where PaO2/FiO2 ratio falls below 150 mmHg, immediate endotracheal intubation "
        "and mechanical ventilation in the intensive care unit (ICU) are indicated.",
        body_style
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>2. Antimicrobial and Antiviral Regimens</b>", h2_style))
    story.append(Paragraph(
        "For hospitalized patients with severe pneumonia, initiate Ceftriaxone 1g or 2g IV once daily combined with Azithromycin 500mg IV daily. "
        "In patients with verified severe COVID-19 requiring supplemental oxygen, Dexamethasone 6mg orally or intravenously once daily "
        "for up to 10 days is recommended. Remdesivir 200mg loading dose IV on day 1 followed by 100mg daily for 5 days accelerates clinical recovery.",
        body_style
    ))
    story.append(Spacer(1, 10))

    table_data = [
        ["Medication", "Indication", "Route & Dosage", "Duration"],
        ["Ceftriaxone", "Bacterial Pneumonia", "1g - 2g IV daily", "5 - 7 days"],
        ["Azithromycin", "Atypical Coverage", "500mg IV/oral daily", "3 - 5 days"],
        ["Dexamethasone", "COVID-19 Hypoxemia", "6mg oral/IV daily", "Up to 10 days"],
        ["Remdesivir", "Severe COVID-19", "200mg loading, 100mg qd", "5 days"]
    ]
    t = Table(table_data, colWidths=[100, 130, 130, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#990000')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t)

    doc.build(story)
    print(f"Created CDC PDF: {output_path}")


if __name__ == "__main__":
    out_dir = Path("data/raw_pdfs")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_who_guideline_pdf(out_dir / "WHO_Guideline_Hypertension_and_Diabetes.pdf")
    create_cdc_pneumonia_pdf(out_dir / "CDC_Clinical_Procedure_Pneumonia_and_Sepsis.pdf")
