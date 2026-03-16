#3 Add rule-based validation

import re
from typing import Dict, List

# Regex Patterns

## Date Pattern (YYYY-MM-DD)
### NOTE: We intentionally do not enforce strict numeric validity (e.g. month/day ranges).
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

## Amount Pattern
### The regex is meant to match common currency formats (USD/$) in free-form text.
AMOUNT_RE = re.compile(r"(?:USD|US\$|\$)\s?\d+(?:,\d{3})*(?:\.\d+)?")

## Clause pattern (for simple label validation)
CLAUSE_RE = re.compile(r"\b(clause|agreement|section|article)\b", flags=re.IGNORECASE)

## Termination Clause Pattern
TERMINATION_RE = re.compile(r"\b(termination|terminate|terminated)\b", flags=re.IGNORECASE)

# Individual Validation Functions

## Date Validation
def validate_date(entity_text: str) -> bool:
    return bool(DATE_RE.search(entity_text))

## Amount Validation
def validate_amount(entity_text: str) -> bool:
    return bool(AMOUNT_RE.search(entity_text))

## Termination Validation
def validate_termination(entity_text: str) -> bool:
    return bool(TERMINATION_RE.search(entity_text))

# Clause classification (simple rule-based)

def classify_clauses(text: str) -> List[Dict]:
    """Extract clause-like spans and classify them by type."""

    clause_patterns = {
        "termination_clause": [r"\bterminate\b", r"\btermination\b", r"\bend of term\b"],
        "payment_clause": [r"\bpay\b", r"\bpayment\b", r"\bfee\b", r"\binvoice\b"],
        "confidentiality_clause": [r"\bconfidential\b", r"\bnon[- ]disclosure\b"],
        "governing_law_clause": [r"\bgoverning law\b", r"\blaw of\b", r"\bjurisdiction\b"],
        "indemnification_clause": [r"\bindemnify\b", r"\bindemnification\b"],
        "force_majeure_clause": [r"\bforce majeure\b", r"\bacts of god\b"],
    }

    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    clauses: List[Dict] = []

    for sentence in sentences:
        lower = sentence.lower()
        for clause_type, patterns in clause_patterns.items():
            for pat in patterns:
                if re.search(pat, lower):
                    clauses.append({"type": clause_type, "text": sentence})
                    break
            else:
                continue
            break

    return clauses

# Main Validation Pipeline

## Function
def validate_entities(entities: List[Dict]) -> List[Dict]:
    validated = []
    # Loop Through Entities
    for ent in entities:
        label = ent.get("label", "").lower()
        text = ent.get("text", "")
        ok = True

        if label == "date":
            ok = validate_date(text)
        elif label in {"money", "amount", "dollar_amount"}:
            ok = validate_amount(text)
        elif label in {"termination", "termination_clause"}:
            ok = validate_termination(text)
        elif label == "clause":
            ok = bool(CLAUSE_RE.search(text))
        elif label == "law":
            ok = len(text) > 3

        validated.append({**ent, "valid": ok})
    return validated