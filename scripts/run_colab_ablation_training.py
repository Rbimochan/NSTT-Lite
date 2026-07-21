"""Plan 4, Step 2 (ablation half): leaky-split training at MATCHED step count
to run_colab_training.py, fixing plan3's 300-vs-100-step confound.

Run this on Colab, AFTER run_colab_training.py has finished and printed its
final global_step (read from models/finetuned_full/trainer_state.json), so
both runs use the identical step budget. Set MATCHED_STEPS below to that value
before running -- do not leave it at the placeholder.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/NSTT-Lite")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.training import create_trainer

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests_leaky"
OUTPUT_DIR = PROJECT_ROOT / "models" / "finetuned_leaky_full"


def _read_matched_steps() -> int:
    state_path = PROJECT_ROOT / "models" / "finetuned_full" / "trainer_state.json"
    if not state_path.exists():
        raise SystemExit(
            f"{state_path} not found -- run run_colab_training.py first so there is "
            "a step count to match."
        )
    state = json.loads(state_path.read_text())
    return state["global_step"]


def main() -> None:
    matched_steps = _read_matched_steps()
    print(f"Matching run_colab_training.py's step count: {matched_steps}")

    trainer, processor = create_trainer(
        PROJECT_ROOT, OUTPUT_DIR, manifest_dir=MANIFEST_DIR, max_train=None, max_eval=None
    )
    trainer.args.max_steps = matched_steps
    trainer.args.num_train_epochs = 5  # ignored once max_steps is set, kept for clarity

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("Matched-step leaky-split ablation training complete.")
    print("train_loss:", train_result.training_loss)
    print("eval_metrics:", eval_metrics)
    print(f"Checkpoint saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
