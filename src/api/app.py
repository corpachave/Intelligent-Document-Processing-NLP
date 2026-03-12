# API Layer for Contract Entity Extraction

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import tempfile
from src.pipeline import extract_entities_from_pdf

app = FastAPI(title="LexiScan Auto - Contract Entity Extractor")


class ExtractionResponse(BaseModel):
    text: str
    entities: list


@app.post("/extract", response_model=ExtractionResponse)
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = extract_entities_from_pdf(tmp_path)
        finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return result