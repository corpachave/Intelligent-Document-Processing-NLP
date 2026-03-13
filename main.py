# Entrypoint CLI

from src.pipeline import extract_entities_from_pdf
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR + NER extraction on a PDF")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--out", help="Path to write JSON output", default=None)
    args = parser.parse_args()

    result = extract_entities_from_pdf(args.pdf_path)
    output_json = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_json)
    else:
        print(output_json)
