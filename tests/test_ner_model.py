import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure tests can import the `src` package even if run with a different current working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import spacy

from src.ner import model


class DummyEnt:
    def __init__(self, text, label, start, end):
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end


class DummyDoc:
    def __init__(self, ents):
        self.ents = ents


class DummyModel:
    def __init__(self, ents):
        self._ents = ents

    def __call__(self, text):
        return DummyDoc(self._ents)


class TestNerModel(unittest.TestCase):
    def test_load_training_data_from_jsonl(self):
        data = {"text": "Hello world", "entities": [[0, 5, "GREETING"]]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
            temp_path = f.name

        try:
            loaded = list(model.load_training_data_from_jsonl(temp_path))
            self.assertEqual(loaded, [data])
        finally:
            os.remove(temp_path)

    def test_extract_entities_on_dummy_model(self):
        ents = [DummyEnt("Apple", "ORG", 0, 5)]
        dummy = DummyModel(ents)
        extracted = model.extract_entities(dummy, "Apple is a company.")
        self.assertEqual(extracted, [{"text": "Apple", "label": "ORG", "start": 0, "end": 5, "source": "model", "confidence": 0.88}])

    def test_extract_entities_filters_short_and_stop_words(self):
        ents = [
            DummyEnt("of", "ORG", 0, 2),
            DummyEnt("John Doe", "PERSON", 3, 11),
            DummyEnt("$100", "MONEY", 12, 16),
        ]
        dummy = DummyModel(ents)
        extracted = model.extract_entities(dummy, "of John Doe $100")
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0]["label"], "PERSON")
        self.assertEqual(extracted[1]["label"], "MONEY")

    def test_load_spacy_docs_from_spacy_file(self):
        nlp = spacy.blank("en")
        doc = nlp("Hello world")
        doc_bin = spacy.tokens.DocBin(docs=[doc])

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".spacy", delete=False) as f:
            path = f.name
            f.write(doc_bin.to_bytes())

        try:
            loaded = model.load_spacy_docs_from_spacy_file(path)
            self.assertIsInstance(loaded, spacy.tokens.DocBin)
            self.assertEqual(len(list(loaded.get_docs(nlp.vocab))), 1)
        finally:
            os.remove(path)

    def test_rule_based_legal_term_matching(self):
        nlp = spacy.blank("en")
        extracted = model.extract_entities(nlp, "This Agreement covers confidentiality and governing law.")
        labels = {e["label"] for e in extracted}
        texts = {e["text"].lower() for e in extracted}
        self.assertIn("LAW", labels)
        self.assertIn("CLAUSE", labels)
        self.assertIn("this agreement", texts)
        self.assertIn("governing law", texts)

    def test_merge_organization_tokens(self):
        class OrgDoc:
            def __init__(self, ents):
                self.ents = ents

        class OrgModel:
            def __init__(self, ents):
                self._ents = ents
            def __call__(self, text):
                return OrgDoc(self._ents)

        ents = [DummyEnt("AZZ SURFACE", "ORG", 0, 11), DummyEnt("TECHNOLOGIES", "ORG", 12, 24), DummyEnt("TAMPA LLC", "ORG", 25, 34)]
        dummy = OrgModel(ents)
        extracted = model.extract_entities(dummy, "AZZ SURFACE TECHNOLOGIES TAMPA LLC")
        orgs = [e for e in extracted if e["label"] == "ORG"]
        self.assertTrue(any("AZZ SURFACE TECHNOLOGIES TAMPA LLC" == e["text"] for e in orgs))

    def test_extract_entities_strict_mode_filters_low_confidence(self):
        ents = [DummyEnt("Apple", "ORG", 0, 5)]
        dummy = DummyModel(ents)
        extracted = model.extract_entities(dummy, "Apple is a company.", strict_mode=True)
        # model entities are default confidence 0.88, strict should filter them out
        self.assertEqual(extracted, [])

    def test_extract_entities_strict_mode_keeps_rule_matches(self):
        nlp = spacy.blank("en")
        extracted = model.extract_entities(nlp, "This Agreement covers governing law.", strict_mode=True)
        labels = {e["label"] for e in extracted}
        self.assertIn("LAW", labels)


if __name__ == "__main__":
    unittest.main()
