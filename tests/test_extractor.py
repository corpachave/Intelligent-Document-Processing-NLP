import os
import sys
import unittest
from pathlib import Path

# Ensure tests can import the project modules regardless of cwd
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.ocr import extractor


class TestExtractor(unittest.TestCase):
    def test_is_scanned_pdf(self):
        # Test with a known digital PDF
        self.assertFalse(extractor.is_scanned_pdf("data/raw_pdfs/loan_agreement_01.pdf"))
        # Note: we don't have a guaranteed scanned PDF in the repo, so we can't test True case easily

    def test_extract_text_from_digital_pdf(self):
        text = extractor.extract_text("data/raw_pdfs/loan_agreement_01.pdf")
        self.assertIn("Agreement", text)
        self.assertGreater(len(text), 100)  # Should extract substantial text

    def test_extract_text_from_sample(self):
        text = extractor.extract_text("data/raw_pdfs/scanned_pdf_samples/Image_pdf_merged_5pages.pdf")
        self.assertIn("Agreement", text)  # adjust based on your sample document