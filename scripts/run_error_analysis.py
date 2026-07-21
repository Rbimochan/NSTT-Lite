"""Step 12: categorize ASR errors on the fine-tuned checkpoint's real test predictions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.error_analysis import run_error_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "finetuned"
MAX_SAMPLES = 150  # matches the Step 9 eval subset


def main() -> None:
    result = run_error_analysis(PROJECT_ROOT, CHECKPOINT_DIR, device="cpu", max_samples=MAX_SAMPLES)
    print(f"Analyzed {result['num_samples']} utterances.")
    print("Category counts:", result["category_counts"])
    print("Sample paths:", result["sample_paths"])
    print("Category paths:", result["category_paths"])


if __name__ == "__main__":
    main()
