# 2 NER model training + inference
## This uses spaCy (transfer learning) and expects training data under data/annotations/… (JSONL or .spacy).

import re
import spacy
from spacy.tokens import DocBin
from pathlib import Path
from typing import List, Dict, Iterable

STOP_WORDS = {"the", "and", "of", "in", ",", "(", ")", "this", "that", "agreement"}
ENTITY_CLEANUP = {"llc", "inc", "co.", ", llc", "ltd", "corp", "co"}

# Regex-based label correction
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Z][a-z]+ \d{1,2}\b")
MONEY_PATTERN = re.compile(r"\$?\d+(,\d{3})*(\.\d+)?")

def validate_entity(entity: Dict[str, object]) -> bool:
    text = str(entity.get("text", "")).strip()
    label = str(entity.get("label", "")).upper()

    if not text:
        return False

    lc = text.lower().strip()
    if lc in ENTITY_CLEANUP:
        return False
    if len(text) < 3:
        return False
    if lc in STOP_WORDS:
        return False
    if label == "MONEY" and not any(ch.isdigit() for ch in text):
        return False
    if label == "DATE" and not DATE_PATTERN.search(text):
        return False
    if label == "CLAUSE" and len(text) < 5:
        return False
    # reject meaningless uppercase single tokens
    if text.isupper() and len(text) <= 2:
        return False
    # reject improbable numeric-only labels except money
    if label not in {"MONEY", "DATE"} and text.replace(" ", "").isdigit():
        return False

    return True


def correct_entity_labels(text: str, label: str) -> str:
    """Use regex hints to correct common entity-label mismatches."""
    text = text.strip()
    if DATE_PATTERN.search(text):
        return "DATE"
    if MONEY_PATTERN.search(text):
        return "MONEY"
    return label


LEGAL_TERM_REGEX = [
    (re.compile(r"\bthis agreement\b", flags=re.IGNORECASE), "LAW"),
    (re.compile(r"\bgoverning law\b", flags=re.IGNORECASE), "LAW"),
    (re.compile(r"\btermination\b", flags=re.IGNORECASE), "CLAUSE"),
    (re.compile(r"\bconfidentiality\b", flags=re.IGNORECASE), "CLAUSE"),
]


def get_legal_term_matches(text: str) -> List[Dict]:
    matches: List[Dict] = []
    for pattern, label in LEGAL_TERM_REGEX:
        for m in pattern.finditer(text):
            matches.append({
                "text": m.group(0),
                "label": label,
                "start": m.start(),
                "end": m.end(),
            })
    return matches


# Data Loading
def load_spacy_docs_from_spacy_file(path: str) -> DocBin:
    return DocBin().from_disk(path)

# Loading JSONL annotation data
def load_training_data_from_jsonl(path: str) -> Iterable[Dict]:
    """
    Expect doccano-style JSONL records:
      { "text": "...", "entities": [[start, end, label], ...] }
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

# Main Training Function
def train_ner(
    output_dir: str,
    train_path: str,
    dev_path: str = None,
    base_model: str = "en_core_web_sm",
    n_iter: int = 30,
):  
    # Load the Base Model (fallback to blank English model if not installed)
    try:
        nlp = spacy.load(base_model)
    except OSError:
        # If the pre-trained model isn't installed, fall back to a blank English model.
        # To use the pre-trained model, install it with:
        #   python -m spacy download en_core_web_sm
        nlp = spacy.blank("en")

    # Ensure NER Pipeline Exists
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from training set
    for sample in load_training_data_from_jsonl(train_path):
        for ent in sample.get("entities", []):
            ner.add_label(ent[2])
    
    # Prepare Training Data (spaCy >=3 expects Example objects)
    from spacy.training import Example
    from spacy.util import compounding, minibatch

    train_data = []
    for sample in load_training_data_from_jsonl(train_path):
        train_data.append((sample["text"], {"entities": sample.get("entities", [])}))

    train_examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        train_examples.append(Example.from_dict(doc, annotations))

    # Optional dev set (used for evaluation only)
    dev_examples = []
    if dev_path:
        for sample in load_training_data_from_jsonl(dev_path):
            doc = nlp.make_doc(sample["text"])
            dev_examples.append(Example.from_dict(doc, {"entities": sample.get("entities", [])}))

    # Disable Other Pipelines During Training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes), nlp.select_pipes(enable="ner"):
        # Initialize the optimizer (spaCy 3+)
        optimizer = nlp.initialize(lambda: train_examples)

        # Training Loop
        for itn in range(n_iter):
            losses = {}
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.2, sgd=optimizer, losses=losses)
            # optional: print or log losses
            # (add validation/evaluation here if desired)
    # Save Model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)

# Model Loading
def load_model(model_dir: str):
    return spacy.load(model_dir)

# Entity Extraction (Inference)
HIGH_CONFIDENCE_THRESHOLD = 0.9

def extract_entities(model, text: str, strict_mode: bool = False, confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD):
    doc = model(text)
    validated: List[Dict] = []

    # Add rule-based legal term matches alongside model entities
    rule_matches = get_legal_term_matches(text)
    for match in rule_matches:
        if validate_entity(match):
            validated.append({**match, "source": "rule", "confidence": 1.0})

    model_entities = []
    for ent in doc.ents:
        text_str = ent.text.strip()
        label = ent.label_.upper()

        # Regex-based correction
        label = correct_entity_labels(text_str, label)

        candidate = {
            "text": text_str,
            "label": label,
            "start": ent.start_char,
            "end": ent.end_char,
            "source": "model",
            "confidence": 0.88,
        }

        # Keep only known contract entity types
        if label not in {"DATE", "MONEY", "ORG", "PERSON", "LAW", "CLAUSE"}:
            continue

        # Post-process money formats (normalize if needed)
        if label == "MONEY":
            candidate["text"] = text_str.strip(".,;:$")

        if validate_entity(candidate):
            model_entities.append(candidate)

    # If strict mode, keep only high confidence model entities
    if strict_mode:
        model_entities = [e for e in model_entities if e.get("confidence", 0.0) >= confidence_threshold]

    # Merge adjacent ORG tokens into one entity
    merged_entities: List[Dict] = []
    model_entities.sort(key=lambda e: (e["start"], -e["end"]))
    i = 0
    while i < len(model_entities):
        current = model_entities[i]
        if current["label"] == "ORG":
            merged = current.copy()
            j = i + 1
            while j < len(model_entities):
                next_ent = model_entities[j]
                if next_ent["label"] == "ORG" and next_ent["start"] <= merged["end"] + 4:
                    joined_text = f"{merged['text']} {next_ent['text']}".strip()
                    merged["text"] = re.sub(r"\s+", " ", joined_text)
                    merged["end"] = max(merged["end"], next_ent["end"])
                    j += 1
                    continue
                break
            merged_entities.append(merged)
            i = j
        else:
            merged_entities.append(current)
            i += 1

    validated.extend(merged_entities)
    return validated
