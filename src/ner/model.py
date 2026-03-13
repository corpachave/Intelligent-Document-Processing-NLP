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
    # Load the Base Model (fallback to blank English model if not installed)
    try:
        nlp = spacy.load(base_model)
    except OSError:
        # If the pre-trained model isn't installed, fall back to a blank English model.
        # To use the pre-trained model, install it with:
        #   python -m spacy download en_core_web_sm
        nlp = spacy.blank("en")

    # Ensure NER Pipeline Exists
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add labels from training set
    for sample in load_training_data_from_jsonl(train_path):
        for ent in sample.get("entities", []):
            ner.add_label(ent[2])
    
    # Prepare Training Data (spaCy >=3 expects Example objects)
    from spacy.training import Example
    from spacy.util import compounding, minibatch

    train_data = []
    for sample in load_training_data_from_jsonl(train_path):
        train_data.append((sample["text"], {"entities": sample.get("entities", [])}))

    train_examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        train_examples.append(Example.from_dict(doc, annotations))

    # Optional dev set (used for evaluation only)
    dev_examples = []
    if dev_path:
        for sample in load_training_data_from_jsonl(dev_path):
            doc = nlp.make_doc(sample["text"])
            dev_examples.append(Example.from_dict(doc, {"entities": sample.get("entities", [])}))

    # Disable Other Pipelines During Training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes), nlp.select_pipes(enable="ner"):
        # Initialize the optimizer (spaCy 3+)
        optimizer = nlp.initialize(lambda: train_examples)

        # Training Loop
        for itn in range(n_iter):
            losses = {}
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.2, sgd=optimizer, losses=losses)
            # optional: print or log losses
            # (add validation/evaluation here if desired)
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
