"""Step 6: smoke-test fine-tuning run against the real manifests (a few steps only)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models" / "smoke_test"


def main() -> None:
    result = train_and_save(
        PROJECT_ROOT, OUTPUT_DIR, smoke_test=True, max_train=32, max_eval=8
    )
    print("Smoke test complete.")
    print("train_loss:", result["train_loss"])
    print("eval_metrics:", result["eval_metrics"])
    print("checkpoint_dir:", result["checkpoint_dir"])


if __name__ == "__main__":
    main()
