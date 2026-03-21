from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import load_dataset

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": "data/annotations/train_bert.json",
    "validation": "data/annotations/dev_bert.json"
})

label_list = [
    "O",
    "B-DATE", "I-DATE",
    "B-MONEY", "I-MONEY",
    "B-ORG", "I-ORG",
    "B-PERSON", "I-PERSON",
    "B-LAW", "I-LAW",
    "B-CLAUSE", "I-CLAUSE"
]

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def encode_labels(example):
    example["labels"] = [label2id[label] for label in example["labels"]]
    return example

dataset = dataset.map(encode_labels)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
data_collator = DataCollatorForTokenClassification(tokenizer)

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

def compute_metrics(p):
    predictions, labels = p

    # Convert logits → label IDs
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, lab in zip(predictions, labels):
        curr_preds = []
        curr_labels = []

        for p_id, l_id in zip(pred, lab):
            if l_id == -100:
                continue  # ignore padding

            curr_preds.append(id2label[p_id])
            curr_labels.append(id2label[l_id])

        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# Training config
training_args = TrainingArguments(
    output_dir="./models/legal_bert_ner",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./models/legal_bert_ner")
tokenizer.save_pretrained("./models/legal_bert_ner")
