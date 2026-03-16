#!/usr/bin/env python3
"""
Complete Pipeline Runner for Fintech Document Processing

This script runs the entire pipeline:
1. Train NER model (if needed)
2. Process PDFs with OCR + NER
3. Validate extracted entities
4. Run tests
5. Start API server (optional)

Usage:
    python run_pipeline.py --train                    # Train model only
    python run_pipeline.py --process path/to/pdf     # Process single PDF
    python run_pipeline.py --test                    # Run tests only
    python run_pipeline.py --api                     # Start API server
    python run_pipeline.py --all                     # Run everything
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import extract_entities_from_pdf
from src.ner.model import train_ner
from src.ocr.extractor import extract_text


def check_requirements():
    """Check if all required files and models exist."""
    print("🔍 Checking requirements...")

    # Check training data
    train_file = ROOT_DIR / "data" / "annotations" / "train.jsonl"
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        return False

    # Check if model exists
    model_dir = ROOT_DIR / "models" / "ner"
    if not model_dir.exists():
        print(f"⚠️  NER model not found: {model_dir}")
        print("   Will train model automatically...")
    else:
        print(f"✅ NER model found: {model_dir}")

    print("✅ Requirements check passed")
    return True


def train_model(force: bool = False):
    """Train the NER model if it doesn't exist or force retrain."""
    print("\n🤖 Training NER Model...")

    model_dir = ROOT_DIR / "models" / "ner"
    train_file = ROOT_DIR / "data" / "annotations" / "train.jsonl"
    dev_file = ROOT_DIR / "data" / "annotations" / "dev.jsonl"

    if model_dir.exists() and not force:
        print(f"✅ Model already exists at {model_dir}")
        return True

    try:
        train_ner(
            output_dir=str(model_dir),
            train_path=str(train_file),
            dev_path=str(dev_file) if dev_file.exists() else None,
            n_iter=30
        )
        print(f"✅ Model trained successfully at {model_dir}")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


def process_pdf(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """Process a single PDF through the entire pipeline."""
    print(f"\n📄 Processing PDF: {pdf_path}")

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        # Extract text first
        print("   📖 Extracting text...")
        text = extract_text(pdf_path)
        print(f"   ✅ Extracted {len(text)} characters")

        # Extract entities
        print("   🧠 Extracting entities...")
        result = extract_entities_from_pdf(pdf_path)

        # Show summary
        entities = result.get("entities", [])
        print(f"   ✅ Found {len(entities)} entities")

        # Count by type
        entity_counts = {}
        for ent in entities:
            label = ent.get("label", "UNKNOWN")
            entity_counts[label] = entity_counts.get(label, 0) + 1

        print("   📊 Entity breakdown:")
        for label, count in entity_counts.items():
            print(f"      {label}: {count}")

        # Save output if requested
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"   💾 Saved results to: {output_path}")

        return result

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise


def run_tests():
    """Run all unit tests."""
    print("\n🧪 Running Tests...")

    try:
        # Use unittest to run tests
        result = subprocess.run([
            sys.executable, "-m", "unittest", "discover", "tests/", "-v"
        ], capture_output=True, text=True, cwd=ROOT_DIR)

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed!")
            return False

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


def start_api(host: str = "0.0.0.0", port: int = 8001):
    """Start the FastAPI server."""
    print(f"\n🚀 Starting API server on {host}:{port}...")

    try:
        # Import here to avoid loading if not needed
        import uvicorn

        print("🌐 API endpoints:")
        print(f"   POST http://localhost:{port}/extract - Upload PDF for processing")
        print(f"   GET  http://localhost:{port}/docs - Interactive API documentation")
        print("\nPress Ctrl+C to stop the server\n")

        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start API: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete Pipeline Runner for Fintech Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --train                    # Train model only
  python run_pipeline.py --process data/raw_pdfs/sample.pdf  # Process single PDF
  python run_pipeline.py --test                     # Run tests only
  python run_pipeline.py --api                      # Start API server
  python run_pipeline.py --all                      # Run everything
        """
    )

    parser.add_argument("--train", action="store_true", help="Train NER model")
    parser.add_argument("--process", metavar="PDF_PATH", help="Process single PDF")
    parser.add_argument("--output", metavar="OUTPUT_PATH", help="Save processing results to JSON file")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline (train + test + api)")
    parser.add_argument("--force-train", action="store_true", help="Force retrain even if model exists")

    args = parser.parse_args()

    # If no specific action, show help
    if not any([args.train, args.process, args.test, args.api, args.all]):
        parser.print_help()
        return

    print("🚀 Fintech Document Processing Pipeline")
    print("=" * 50)

    success = True

    # Check requirements first
    if not check_requirements():
        return

    # Handle --all flag
    if args.all:
        args.train = True
        args.test = True
        args.api = True

    # Train model
    if args.train or args.all:
        if not train_model(force=args.force_train):
            success = False

    # Process PDF
    if args.process:
        try:
            result = process_pdf(args.process, args.output)
            print(f"\n📋 Processing complete! Found {len(result.get('entities', []))} entities")
        except Exception as e:
            print(f"❌ PDF processing failed: {e}")
            success = False

    # Run tests
    if args.test or args.all:
        if not run_tests():
            success = False

    # Start API
    if args.api or args.all:
        if success:  # Only start API if everything else succeeded
            start_api()
        else:
            print("❌ Skipping API start due to previous failures")

    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()