# Legal Contract Intelligence System

Built an NLP-based system to convert unstructured legal PDF contracts into structured, searchable data.

Designed an end-to-end pipeline including OCR, text preprocessing, Named Entity Recognition (NER), and clause classification using fine-tuned transformer models (BERT/Legal-BERT). Extracted key entities such as parties, loan amounts, dates and legal clauses, and generated structured JSON outputs for database storage.

Implemented backend APIs using FastAPI and integrated PostgreSQL / Elasticsearch to enable contract-level search, filtering and clause retrieval.

Evaluated models using Precision, Recall, and F1-score and deployed the system using Docker and AWS.