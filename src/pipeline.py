#4 Add API + pipeline 
## src/pipeline.py
from src.ocr.extractor import extract_text
from src.ner.model import load_model, extract_entities
from src.validation.rules import validate_entities, classify_clauses

DEFAULT_MODEL_DIR = "models/ner"
PRODUCTION_CONFIDENCE_THRESHOLD = 0.7


def extract_entities_from_pdf(pdf_path: str, model_dir: str = DEFAULT_MODEL_DIR, strict_mode: bool = False):
    text = extract_text(pdf_path)
    model = load_model(model_dir)
    entities = extract_entities(model, text, strict_mode=strict_mode)
    validated = validate_entities(entities)

    # Production strict output: filter by confidence for downstream systems
    if strict_mode:
        validated = [e for e in validated if e.get("confidence", 0.0) >= PRODUCTION_CONFIDENCE_THRESHOLD]

    clauses = classify_clauses(text)
    return {"text": text, "entities": validated, "clauses": clauses}