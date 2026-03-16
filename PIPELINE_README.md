# Fintech Document Processing Pipeline

A complete pipeline for intelligent document processing using OCR, NER, and validation for fintech applications.

## Quick Start

```bash
# Run the complete pipeline (train model, run tests, start API)
python run_pipeline.py --all

# Or run individual components
python run_pipeline.py --train          # Train NER model
python run_pipeline.py --process path/to/document.pdf  # Process single PDF
python run_pipeline.py --test           # Run unit tests
python run_pipeline.py --api            # Start API server
```

## Pipeline Components

1. **OCR Extraction** - Handles both digital and scanned PDFs
2. **NER Training** - Trains spaCy model on contract entities
3. **Entity Extraction** - Identifies dates, amounts, parties, etc.
4. **Validation** - Applies business rules to extracted entities
5. **API Server** - REST API for document processing

## API Usage

Once the API is running (`python run_pipeline.py --api`), you can:

- **Upload PDF**: `POST /extract` with PDF file
- **View docs**: `GET /docs` for interactive API documentation

## Project Structure

```
├── run_pipeline.py          # Main pipeline runner
├── src/
│   ├── ocr/extractor.py     # PDF text extraction
│   ├── ner/model.py         # NER training & inference
│   ├── validation/rules.py  # Entity validation
│   ├── pipeline.py          # End-to-end processing
│   └── api/app.py           # FastAPI server
├── data/
│   └── annotations/         # Training data (JSONL)
├── models/ner/              # Trained NER model
├── tests/                   # Unit tests
└── docker/                  # Containerization
```

## Requirements

- Python 3.8+
- Dependencies: `pip install -r requirements.txt`
- Training data in `data/annotations/train.jsonl`

## Example Output

```json
{
  "text": "This Agreement is made on 2025-12-01...",
  "entities": [
    {
      "text": "2025-12-01",
      "label": "DATE",
      "start": 25,
      "end": 35,
      "valid": true
    }
  ]
}
```