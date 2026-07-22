"""Step 9 ablation: same architecture/hyperparameters/step count as run_full_training.py,
but trained on the leaky (utterance-random) split instead of the speaker-disjoint one."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import EarlyStoppingCallback

from src.training import create_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests_leaky"
OUTPUT_DIR = PROJECT_ROOT / "models" / "finetuned_leaky"
MAX_STEPS = 100  # reduced vs. Step 8's 300 to keep the ablation tractable locally


def main() -> None:
    trainer, processor = create_trainer(
        PROJECT_ROOT, OUTPUT_DIR, manifest_dir=MANIFEST_DIR, max_eval=100
    )
    trainer.args.max_steps = MAX_STEPS
    trainer.args.eval_steps = 50
    trainer.args.save_steps = 50
    trainer.args.logging_steps = 10
    trainer.args.load_best_model_at_end = True
    trainer.args.metric_for_best_model = "wer"
    trainer.args.greater_is_better = False
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("Leaky-split ablation training complete.")
    print("train_loss:", train_result.training_loss)
    print("eval_metrics:", eval_metrics)


if __name__ == "__main__":
    main()
