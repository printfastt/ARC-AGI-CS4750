
# ARC Prize 2025 — Full(er) Baseline

A readable, dependency-free solver package for **ARC Prize 2025** (ARC‑AGI‑2). It learns a variety of human-ish transforms, backs off to short program search, and writes the required `submission.json` with **two attempts per test input**.

> Works locally with the public ARC‑AGI‑2 repo and on Kaggle (no‑internet, notebook‑only).

## Features

- Grid toolkit: rotations, flips, transpose, integer scale up/down, nz‑bbox crop, translation, connected components, border drawing, tiling.
- Learners (exact fits over train pairs):
  - rotation / flip / transpose
  - global color permutation (bijective) across all training pairs
  - same‑shape translation with 0 fill
  - crop‑to‑nonzero bounding box
  - largest/smallest component extraction
  - detect & add uniform border (thickness 1–3)
  - integer scale up/down (nearest‑neighbor up; block‑mode down)
  - small compositions (recolor∘rotation, rotation∘recolor)
- Short program search (length ≤ 2 from a small library) to catch simple multi‑step rules.
- Always emits **two** predictions per test input, as required by Kaggle evaluation.

## Quick start (Kaggle)

```python
from arc25_full import generate_submission
eval_dir = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"  # adjust if needed
generate_submission(eval_dir, out_path="submission.json")
```

Then **Commit → Submit to Competition**.

## Local dev (with ARC‑AGI‑2)

1) Clone https://github.com/ai-art-dev99/ARC-Prize-2025  
2) Run:

```bash
python -m arc25_full.solver --help  # or use the API
```

or via Python:

```python
from arc25_full import generate_submission
generate_submission("/path/to/ARC-AGI-2/data/evaluation", "submission.json", limit=10)
```

## Notes

- This starter is intentionally conservative and fast. Extend by adding richer transforms and a smarter search policy if you want more coverage.
- Keep the notebook offline and within Kaggle efficiency limits.
