import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

# Ensure tests can import the `src` package even if run with a different current working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.api.app import app


class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("src.api.app.extract_entities_from_pdf")
    def test_extract_pdf_success(self, mock_extract):
        mock_extract.return_value = {"text": "abc", "entities": []}

        response = self.client.post(
            "/extract",
            files={"file": ("test.pdf", b"%PDF-1.4\n%EOF", "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "abc", "entities": []})
        mock_extract.assert_called_once()

    def test_extract_pdf_invalid_extension(self):
        response = self.client.post(
            "/extract",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
