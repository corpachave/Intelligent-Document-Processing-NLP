# Fintech Intelligent Document Processing (NLP)

An end-to-end Document AI pipeline that converts unstructured legal/financial PDF documents into structured, searchable data using OCR, NLP, and rule-based validation.

## Project Overview

Legal and financial institutions deal with thousands of contracts in PDF format. These documents are:
- Unstructured  
- Lengthy  
- Difficult to search manually  

This project builds an automated system that:
- Extracts text from both digital and scanned PDFs
- Identifies key entities (dates, parties, amounts, etc.)
- Detects important legal clauses
- Converts everything into structured JSON format

## Key Features

### 1. OCR Integration
- Handles digital and scanned PDFs
- Uses Tesseract OCR for image-to-text conversion

### 2. Named Entity Recognition (NER)
- Extracts Dates, Organizations, Money, Legal entities
- Built using spaCy

### 3. Clause Extraction
- Detects Payment, Termination, Governing Law clauses

### 4. Validation Layer
- Regex correction
- Stopword filtering
- Entity cleanup

### 5. API Integration
- Built with FastAPI
- Upload PDF → Get JSON output

## Project Architecture

PDF → Detection → OCR → Preprocessing → NER → Rules → Clauses → Validation → JSON

## Project Structure

├── run_pipeline.py  
├── src/  
│   ├── ocr/extractor.py  
│   ├── ner/model.py  
│   ├── validation/rules.py  
│   ├── pipeline.py  
│   └── api/app.py  
├── data/  
├── models/  
├── tests/  

## Installation

```bash
git clone https://github.com/corpachave/Fintech---Intelligent-Document-Processing-NLP-.git
cd Fintech---Intelligent-Document-Processing-NLP
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py --all
python run_pipeline.py --train
python run_pipeline.py --process file.pdf
python run_pipeline.py --api
```

## API

POST /extract → Upload PDF  
GET /docs → Swagger UI  

## Example Output

```json
{
  "text": "Agreement...",
  "entities": [{"text": "2025-12-01", "label": "DATE"}],
  "clauses": [{"type": "payment_clause"}]
}
```

## Dataset

Stored in:
data/annotations/train.jsonl

## Testing

```bash
python -m pytest
```

## Future Improvements

- BERT fine-tuning  
- Better accuracy  
- Multi-language support  

## Author

K. SIONE CORPACHAVE

## Summary

A complete Document AI pipeline combining OCR + NLP + API to convert contracts into structured data.
