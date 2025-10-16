
import argparse
from .solver import generate_submission
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="submission.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    generate_submission(args.eval_dir, out_path=args.out, limit=args.limit)
