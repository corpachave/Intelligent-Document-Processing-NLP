# Training script (use datasets in data)
from src.ner.model import train_ner

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training JSONL (doccano style)")
    parser.add_argument("--dev", required=False, help="Path to dev JSONL")
    parser.add_argument("--output", default="models/ner", help="Where to save the trained model")
    parser.add_argument("--iters", type=int, default=30, help="Training iterations")
    args = parser.parse_args()

    train_ner(output_dir=args.output, train_path=args.train, dev_path=args.dev, n_iter=args.iters)

    