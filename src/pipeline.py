# Pipeline to extract text from PDF, run NER, and apply validation rules

from src.ocr.extractor import extract_text
from src.ner.model import extract_entities
from src.validation.rules import validate_entities, classify_clauses

def group_entities(entities):
    grouped = {}
    for e in entities:
        label = e["label"]
        grouped.setdefault(label, []).append(e["text"])
    return grouped

PRODUCTION_CONFIDENCE_THRESHOLD = 0.7


def extract_entities_from_pdf(pdf_path: str, strict_mode: bool = False):
    """
    End-to-end pipeline:
    PDF → Text → NER → Validation → Clause Extraction → JSON
    """

    # Step 1: Extract text (OCR or direct)
    text = extract_text(pdf_path)

    # Step 2: Run NER (BERT-based)
    entities = extract_entities(text, strict_mode=strict_mode)

    # Step 3: Additional validation layer (optional but good)
    validated = validate_entities(entities)

    # Step 4: Production filtering (final safety layer)
    if strict_mode:
        validated = [
            e for e in validated
            if e.get("confidence", 0.0) >= PRODUCTION_CONFIDENCE_THRESHOLD
        ]

    # Step 5: Clause extraction
    clauses = classify_clauses(text)

    # Final Output
    return {
    "text": text,
    "entities": group_entities(validated),
    "clauses": clauses
}