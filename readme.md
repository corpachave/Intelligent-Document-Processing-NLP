# LexiScan Auto – Legal Contract Entity Extractor

An end-to-end Intelligent Document Processing (IDP) system for extracting structured information from legal contracts using OCR and NLP.

## Overview

**LexiScan Auto** is designed for financial/legal organizations that process large volumes of contracts. It automates the extraction of key entities such as:

- Dates  
- Monetary Values  
- Organizations  
- Persons  
- Legal Clauses  

The system converts raw PDFs (including scanned documents) into structured, machine-readable JSON.

## Architecture

PDF → OCR → Text Cleaning → BERT NER → Validation → Clause Extraction → API → JSON Output

## Tech Stack

- Python  
- Transformers (Hugging Face)  
- Legal-BERT (`nlpaueb/legal-bert-base-uncased`)  
- FastAPI  
- Tesseract OCR  
- pdf2image  
- Docker  

## Key Features

### OCR Integration
- Supports both native PDFs and scanned documents  
- Converts images → text using Tesseract  

### Custom NER (Legal-BERT)
- Fine-tuned transformer model  
- BIO tagging format  
- Extracts:
  - DATE, MONEY, ORG, PERSON, LAW, CLAUSE  

### Post-processing & Validation
- Regex validation (DATE, MONEY)  
- Stopword filtering  
- Confidence threshold filtering  
- Entity cleanup  

### Clause Extraction
- Detects:
  - Payment clauses  
  - Governing law clauses  
  - Termination-related clauses  

### Clean API Output

Entities are grouped for production usability:

```json
{
  "entities": {
    "ORG": ["ABC Corp"],
    "DATE": ["2024-01-01"],
    "MONEY": ["$5000"]
  }
}

■ FastAPI Microservice

Upload PDF → Get structured JSON
Interactive Swagger UI

■ Project Structure

├── src/
│   ├── api/            # FastAPI application
│   ├── ner/            # BERT NER model + inference
│   ├── ocr/            # OCR pipeline
│   ├── validation/     # Rule-based validation
│   └── pipeline.py     # End-to-end pipeline
│
├── models/             # Trained Legal-BERT model
├── data/               # Training data (JSONL)
├── scripts/            # Training scripts
├── tests/              # Unit tests
├── run_pipeline.py     # CLI runner
└── README.md 

■ How to Run?

1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Run Full Pipeline (CLI)

Process a PDF:

python run_pipeline.py --process data/raw_pdfs/sample.pdf

Save output:

python run_pipeline.py --process sample.pdf --output result.json

3️⃣ Start API Server

python run_pipeline.py --api

Open Swagger UI:

http://localhost:8001/docs

4️⃣ API Usage

POST /extract

Upload a PDF → get structured output:

{
  "text": "...",
  "entities": {
    "ORG": ["ABC Corp"],
    "MONEY": ["$5000"]
  },
  "clauses": [
    {
      "type": "payment_clause",
      "text": "The borrower shall pay..."
    }
  ]
}

■ Model Details

Base Model: Legal-BERT
Task: Token Classification (NER)
Labels:
O, B/I-DATE, B/I-MONEY, B/I-ORG, B/I-PERSON, B/I-LAW, B/I-CLAUSE
Metrics:
Precision, Recall, F1-score

■ Testing

python run_pipeline.py --test

■ Docker (Optional)

Build:
docker build -t lexiscan-auto .

Run:
docker run -p 8001:8001 lexiscan-auto

■ Production Highlights

▪ Modular architecture
▪ Scalable API design
▪ Clean structured outputs
▪ OCR + NLP integration
▪ Real-world legal use case

■ Future Improvements

▪ Replace rule-based clause extraction with transformer-based model
▪ Add document classification
▪ Improve OCR accuracy with layout-aware models
▪ Deploy on cloud (AWS/GCP)

■ Use Case

A financial law firm can:
▪ Upload contracts
▪ Automatically extract key entities
▪ Index documents for search
▪ Reduce manual review time

■ Author

Sione Corpachave

■ Project Status
End-to-end pipeline complete
▪ API working
▪ Model integrated
▪ Ready for demo & deployment