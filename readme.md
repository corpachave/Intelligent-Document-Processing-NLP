# Legal Contract Intelligence System

Building an NLP-based system to convert unstructured legal PDF contracts into structured, searchable data.

Design an end-to-end pipeline including OCR, text preprocessing, Named Entity Recognition (NER), and clause classification using fine-tuned transformer models (BERT/Legal-BERT). Extract key entities such as parties, loan amounts, dates and legal clauses, and generate structured JSON outputs for database storage.

Implement backend APIs using FastAPI and integrated PostgreSQL / Elasticsearch to enable contract-level search, filtering and clause retrieval.

Evaluate models using Precision, Recall, and F1-score and deploy the system using Docker and AWS.

## Prerequisites

This project uses **Tesseract OCR** via `pytesseract` for scanned PDF text extraction. You must have the Tesseract binary installed and accessible to Python.

### Windows (recommended)

1. Install Tesseract using `winget`:

```powershell
winget install -e --id tesseract-ocr.tesseract --accept-source-agreements --accept-package-agreements
```

2. Ensure the installation path is set in your environment variables (for example):

```powershell
setx TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

3. Restart your terminal/IDE so the environment variable is picked up.

Alternatively, you can set `TESSERACT_PATH` or `TESSERACT_CMD` to the location of `tesseract.exe`.

> If you see `pytesseract.pytesseract.TesseractNotFoundError`, it means Python cannot find the Tesseract binary on your system PATH or via the configured environment variables.
