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
        self.assertEqual(extracted, [{"text": "Apple", "label": "ORG", "start": 0, "end": 5}])

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


if __name__ == "__main__":
    unittest.main()
