import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# =========================
# CONFIG
# =========================

MODEL_NAME = "models/legal_bert_ner"

# Lowered threshold (IMPORTANT)
CONFIDENCE_THRESHOLD = 0.65

VALID_LABELS = {"DATE", "MONEY", "ORG", "PERSON", "LAW", "CLAUSE"}

STOP_WORDS = {"the", "and", "of", "in", ",", "(", ")", "this", "that", "agreement"}
ENTITY_CLEANUP = {"llc", "inc", "co.", ", llc", "ltd", "corp", "co"}

DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b[A-Z][a-z]+ \d{1,2}\b")
MONEY_PATTERN = re.compile(r"\$?\d+(,\d{3})*(\.\d+)?")

# =========================
# LOAD MODEL (once)
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="first"  # 🔥 improved
)

# =========================
# VALIDATION
# =========================

def validate_entity(entity: Dict) -> bool:
    text = entity.get("text", "").strip()
    label = entity.get("label", "").upper()

    if not text:
        return False

    lc = text.lower()

    if lc in STOP_WORDS or lc in ENTITY_CLEANUP:
        return False

    if len(text) < 3:
        return False

    if label == "DATE" and not DATE_PATTERN.search(text):
        return False

    if label == "MONEY" and not any(ch.isdigit() for ch in text):
        return False

    if label not in VALID_LABELS:
        return False

    if text.isupper() and len(text) <= 2:
        return False

    if label not in {"DATE", "MONEY"} and text.replace(" ", "").isdigit():
        return False

    return True


# =========================
# SMART LABEL CORRECTION
# =========================

def correct_entity_labels(text: str, label: str) -> str:
    # Only override when VERY confident pattern
    if label != "DATE" and DATE_PATTERN.search(text):
        return "DATE"

    if label != "MONEY" and MONEY_PATTERN.search(text):
        return "MONEY"

    return label


# =========================
# RULE BOOST
# =========================

LEGAL_TERM_REGEX = [
    (re.compile(r"\btermination\b", re.I), "CLAUSE"),
    (re.compile(r"\bconfidentiality\b", re.I), "CLAUSE"),
    (re.compile(r"\bgoverning law\b", re.I), "LAW"),
]

def get_rule_entities(text: str) -> List[Dict]:
    results = []
    for pattern, label in LEGAL_TERM_REGEX:
        for m in pattern.finditer(text):
            results.append({
                "text": m.group(0),
                "label": label,
                "start": m.start(),
                "end": m.end(),
                "confidence": 0.95,
                "source": "rule"
            })
    return results


# =========================
# CLEAN TOKEN TEXT
# =========================

def clean_text(word: str) -> str:
    # Remove BERT subword artifacts
    word = word.replace("##", "")
    return word.strip()


# =========================
# MAIN EXTRACTION
# =========================

def extract_entities(text: str, strict_mode: bool = True) -> List[Dict]:

    raw_entities = ner_pipeline(text)
    processed_entities: List[Dict] = []

    for ent in raw_entities:
        word = clean_text(ent["word"])
        label = ent["entity_group"].upper()
        score = float(ent["score"])

        label = correct_entity_labels(word, label)

        entity = {
            "text": word,
            "label": label,
            "start": ent["start"],
            "end": ent["end"],
            "confidence": score,
            "source": "model"
        }

        # Confidence filtering
        if strict_mode and score < CONFIDENCE_THRESHOLD:
            continue

        # Validation filtering
        if not validate_entity(entity):
            continue

        # Normalize MONEY
        if label == "MONEY":
            entity["text"] = word.strip(".,;:$")

        processed_entities.append(entity)

    # Merge entities
    processed_entities = merge_entities(processed_entities)

    # Add rule-based entities
    rule_entities = get_rule_entities(text)

    final_entities = processed_entities.copy()

    for rule_ent in rule_entities:
        if not overlaps(rule_ent, processed_entities):
            final_entities.append(rule_ent)

    return final_entities


# =========================
# MERGING LOGIC (IMPROVED)
# =========================

def merge_entities(entities: List[Dict]) -> List[Dict]:
    entities = sorted(entities, key=lambda x: (x["start"], x["end"]))
    merged = []

    for ent in entities:
        if not merged:
            merged.append(ent)
            continue

        last = merged[-1]

        # Merge if same label and close
        if ent["label"] == last["label"] and ent["start"] <= last["end"] + 3:
            last["text"] = f"{last['text']} {ent['text']}".strip()
            last["end"] = max(last["end"], ent["end"])
            last["confidence"] = max(last["confidence"], ent["confidence"])
        else:
            merged.append(ent)

    return merged


def overlaps(ent, entity_list):
    for e in entity_list:
        if not (ent["end"] < e["start"] or ent["start"] > e["end"]):
            return True
    return False