# 2 NER model training + inference
## This uses spaCy (transfer learning) and expects training data under data/annotations/… (JSONL or .spacy).
import spacy
from spacy.tokens import DocBin
from pathlib import Path
from typing import List, Dict, Iterable

# Data Loading
def load_spacy_docs_from_spacy_file(path: str) -> DocBin:
    return DocBin().from_disk(path)

# Loading JSONL annotation data
def load_training_data_from_jsonl(path: str) -> Iterable[Dict]:
    """
    Expect doccano-style JSONL records:
      { "text": "...", "entities": [[start, end, label], ...] }
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

# Main Training Function
def train_ner(
    output_dir: str,
    train_path: str,
    dev_path: str = None,
    base_model: str = "en_core_web_sm",
    n_iter: int = 30,
):  
    # Load the Base Model
    nlp = spacy.load(base_model)
    
    # Ensure NER Pipeline Exists
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from training set
    for sample in load_training_data_from_jsonl(train_path):
        for ent in sample.get("entities", []):
            ner.add_label(ent[2])
    
    # Prepare Training Data
    train_data = []
    for sample in load_training_data_from_jsonl(train_path):
        train_data.append((sample["text"], {"entities": sample.get("entities", [])}))
    
    # Disable Other Pipelines During Training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes), nlp.select_pipes(enable="ner"):
        # Initialize the optimizer
        optimizer = nlp.resume_training()
        # Training Loop
        for itn in range(n_iter):
            losses = {}
            for text, annotations in train_data:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.2, losses=losses)
            # optional: print or log losses
    # Save Model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)

# Model Loading
def load_model(model_dir: str):
    return spacy.load(model_dir)

# Entity Extraction (Inference)
def extract_entities(model, text: str):
    doc = model(text)
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
