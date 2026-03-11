"""OCR package.

Expose a small public API for OCR extraction.
"""

from .extractor import extract_text, is_scanned_pdf

__all__ = ["extract_text", "is_scanned_pdf"]
