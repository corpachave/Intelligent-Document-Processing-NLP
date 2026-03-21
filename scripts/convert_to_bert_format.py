import json
from transformers import AutoTokenizer

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def convert(sample):
    text = sample["text"]
    entities = sample.get("entities", [])

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # Convert to normal dict (THIS FIXES YOUR ERROR)
    encoding = dict(encoding)

    labels = ["O"] * len(encoding["input_ids"])

    for start, end, label in entities:
        for i, (s, e) in enumerate(encoding["offset_mapping"]):
            if s == e:
                continue

            if s >= start and e <= end:
                if s == start:
                    labels[i] = f"B-{label}"
                else:
                    labels[i] = f"I-{label}"

    # Remove offset mapping (not needed for training)
    encoding.pop("offset_mapping")

    encoding["labels"] = labels

    return encoding


def process_file(input_path, output_path):
    output_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            output_data.append(convert(sample))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    process_file(
        "data/annotations/train.jsonl",
        "data/annotations/train_bert.json"
    )

    process_file(
        "data/annotations/dev.jsonl",
        "data/annotations/dev_bert.json"
    )

    print("Conversion complete!")