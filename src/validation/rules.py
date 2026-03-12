#3 Add rule-based validation

import re
from typing import Dict, List

# Regex Patterns

## Date Pattern (YYYY-MM-DD)
### NOTE: We intentionally do not enforce strict numeric validity (e.g. month/day ranges).
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

## Amount Pattern
### The regex is meant to match common currency formats (USD/$) in free-form text.
AMOUNT_RE = re.compile(r"(?:USD|US\$|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")

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

# Main Validation Pipeline

## Function
def validate_entities(entities: List[Dict]) -> List[Dict]:
    validated = []
    # Loop Through Entities
    for ent in entities:
        # Extract Label and Text
        label = ent.get("label", "").lower()
        text = ent.get("text", "")
        ok = True
        # Run Correct Validator
        if label == "date":
            ok = validate_date(text)
        elif label in {"amount", "dollar_amount"}:
            ok = validate_amount(text)
        elif label in {"termination", "termination_clause"}:
            ok = validate_termination(text)
        # (Party names can be skipped or optionally validated)

        # Append Validation Result
        validated.append({**ent, "valid": ok})
    return validated