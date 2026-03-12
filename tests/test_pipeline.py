import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure tests can import the `src` package even if run with a different current working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import extract_entities_from_pdf


class TestPipeline(unittest.TestCase):
    @patch("src.pipeline.extract_text", return_value="dummy text")
    @patch("src.pipeline.load_model", return_value="dummy model")
    @patch("src.pipeline.extract_entities", return_value=[{"text": "a", "label": "X", "start": 0, "end": 1}])
    @patch("src.pipeline.validate_entities", return_value=[{"text": "a", "label": "X", "start": 0, "end": 1, "valid": True}])
    def test_extract_entities_from_pdf_calls_subcomponents(
        self, mock_validate, mock_extract_entities, mock_load_model, mock_extract_text
    ):
        result = extract_entities_from_pdf("dummy.pdf", model_dir="models/ner")

        mock_extract_text.assert_called_once_with("dummy.pdf")
        mock_load_model.assert_called_once_with("models/ner")
        mock_extract_entities.assert_called_once_with("dummy model", "dummy text")
        mock_validate.assert_called_once()

        self.assertEqual(result, {"text": "dummy text", "entities": [{"text": "a", "label": "X", "start": 0, "end": 1, "valid": True}]})
