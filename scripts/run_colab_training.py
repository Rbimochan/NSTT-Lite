"""Plan 4, Step 1: full-scale fine-tuning on Colab T4 GPU.

Run this INSIDE Colab, not locally -- it assumes CUDA is available (fp16
kicks in automatically once it is, per src/training.py's build_training_arguments)
and that the repo + data live under /content/drive/MyDrive/NSTT-Lite (adjust
PROJECT_ROOT below if your Drive path differs).

Per plan4.md Step 1: AdamW, FP16, batch 2, grad-accum 4, max 5 epochs / 2000
steps (whichever comes first), early stopping patience 2 on val WER.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/NSTT-Lite")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # local fallback for a dry run

sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import EarlyStoppingCallback

from src.training import create_trainer

OUTPUT_DIR = PROJECT_ROOT / "models" / "finetuned_full"
MAX_EPOCHS = 5


def main() -> None:
    if not torch.cuda.is_available():
        print(
            "WARNING: no CUDA device visible. In Colab: Runtime -> Change runtime "
            "type -> GPU (T4), then Runtime -> Restart session before re-running. "
            "Continuing anyway will train on CPU and defeat the point of this script."
        )

    # Full val split for a real WER curve -- affordable on a T4, unlike CPU.
    trainer, processor = create_trainer(PROJECT_ROOT, OUTPUT_DIR, max_train=None, max_eval=None)
    trainer.args.num_train_epochs = MAX_EPOCHS
    trainer.args.load_best_model_at_end = True
    trainer.args.metric_for_best_model = "wer"
    trainer.args.greater_is_better = False
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"max_steps={trainer.args.max_steps}, num_train_epochs={trainer.args.num_train_epochs}")
    print("SCREENSHOT NOW: GPU runtime type + this printed device info (Plan 4 Step 6).")

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))

    print("Full-scale training complete.")
    print("train_loss:", train_result.training_loss)
    print("eval_metrics:", eval_metrics)
    print(f"Best checkpoint saved to: {OUTPUT_DIR}")
    print("SCREENSHOT NOW: TensorBoard loss/WER curves (Plan 4 Step 6).")
    print(f"TensorBoard: %load_ext tensorboard && %tensorboard --logdir {OUTPUT_DIR / 'runs'}")


if __name__ == "__main__":
    main()
