# API Layer for Contract Entity Extraction

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import tempfile
from src.pipeline import extract_entities_from_pdf
from typing import Dict, List

app = FastAPI(title="LexiScan Auto - Contract Entity Extractor")


# Clause model (unchanged)
class Clause(BaseModel):
    type: str
    text: str


# FIXED response model
class ExtractResponse(BaseModel):
    text: str
    entities: Dict[str, List[str]]
    clauses: List[Clause]   # 🔥 FIXED HERE


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(file: UploadFile = File(...), strict: bool = False):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = extract_entities_from_pdf(tmp_path, strict_mode=strict)

        # Ensure correct structure
        if "entities" not in result or not isinstance(result["entities"], dict):
            result["entities"] = {}

        if "clauses" not in result or not isinstance(result["clauses"], list):
            result["clauses"] = []

        return result

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass