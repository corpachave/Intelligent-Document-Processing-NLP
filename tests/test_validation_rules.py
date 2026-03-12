import os
import sys
from pathlib import Path
import unittest

# Ensure tests can import the project modules regardless of cwd
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.validation import rules


class TestValidationRules(unittest.TestCase):
    def test_validate_date(self):
        self.assertTrue(rules.validate_date("2025-12-31"))
        self.assertTrue(rules.validate_date("Ends on 2025-12-31."))
        self.assertFalse(rules.validate_date("12/31/2025"))
        # Note: the regex does not validate month/day ranges, only format
        self.assertTrue(rules.validate_date("2025-13-01"))

    def test_validate_amount(self):
        self.assertTrue(rules.validate_amount("$1,000.00"))
        self.assertTrue(rules.validate_amount("USD 250"))
        self.assertTrue(rules.validate_amount("US$ 3,500"))
        self.assertFalse(rules.validate_amount("1000 dollars"))
        # note: the regex matches '$1000' portion even if decimals are malformed
        self.assertTrue(rules.validate_amount("$1000.0"))

    def test_validate_termination(self):
        self.assertTrue(rules.validate_termination("Termination shall occur."))
        self.assertTrue(rules.validate_termination("terminated by either party"))
        self.assertTrue(rules.validate_termination("to terminate the agreement"))
        self.assertFalse(rules.validate_termination("This is not about ending"))

    def test_validate_entities(self):
        entities = [
            {"label": "DATE", "text": "2025-12-31"},
            {"label": "amount", "text": "$1,000.00"},
            {"label": "termination", "text": "termination"},
            {"label": "party", "text": "Acme Corp"},
            {"label": "date", "text": "31-12-2025"},
        ]
        validated = rules.validate_entities(entities)
        # verify valid flags
        self.assertTrue(validated[0]["valid"])
        self.assertTrue(validated[1]["valid"])
        self.assertTrue(validated[2]["valid"])
        self.assertTrue(validated[3]["valid"])  # party is always ok
        self.assertFalse(validated[4]["valid"])  # bad date format


if __name__ == "__main__":
    unittest.main()
