"""Step 8: full (scaled-down local) fine-tuning run on the real train split.

Plan 4/5 spec is max_steps=2000 on a Colab T4 GPU. No CUDA GPU is available on
this machine, so max_steps is reduced to make a local CPU/MPS run finish in a
practical timeframe. Architecture and remaining hyperparameters (AdamW, batch
2, grad-accum 4, early stopping patience 2 on val WER) are unchanged from the
roadmap's Plan 4 spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import EarlyStoppingCallback

from src.training import create_trainer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models" / "finetuned"
MAX_STEPS = 300  # scaled down from Plan 4's 2000 due to no local GPU


def main() -> None:
    trainer, processor = create_trainer(PROJECT_ROOT, OUTPUT_DIR, max_eval=100)
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

    print("Full (scaled) training complete.")
    print("train_loss:", train_result.training_loss)
    print("eval_metrics:", eval_metrics)
    print("best checkpoint:", trainer.state.best_model_checkpoint)


if __name__ == "__main__":
    main()
