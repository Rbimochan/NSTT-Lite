"""Phase 6: fine-tune gagan3012/wav2vec2-xlsr-nepali on the speaker-disjoint
OpenSLR-54 train split.

Usage:
    python scripts/run_xlsr_train.py --smoke                 # local sanity check
    python scripts/run_xlsr_train.py                          # full run (Colab T4)
    python scripts/run_xlsr_train.py --resume true            # after a disconnect

MLflow tracking goes to ./mlruns (experiment "phase6-finetune").
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import torch

from src.xlsr_training import train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None, help="'true' or a checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
        or (PROJECT_ROOT / ("models/xlsr-ft-smoke" if args.smoke else "models/xlsr-ft"))
    )
    resume = (
        True if args.resume == "true" else args.resume if args.resume else None
    )

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print("Device:", json.dumps(info, indent=2))
    print("Output:", output_dir)
    print("Mode:", "smoke" if args.smoke else "full")
    if not args.smoke and not torch.cuda.is_available():
        print(
            "WARNING: full training without CUDA will be extremely slow -- "
            "this mode is intended for a Colab T4 GPU."
        )

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("phase6-finetune")

    result = train_and_save(
        PROJECT_ROOT,
        output_dir,
        smoke_test=args.smoke,
        resume_from_checkpoint=resume,
        seed=args.seed,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"created_at": stamp, "device": info, "smoke_test": args.smoke, **result}
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out = reports_dir / f"phase6_train_{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print("train_loss:", result["train_loss"])
    print("eval_metrics:", result["eval_metrics"])
    print("global_step:", result["global_step"])
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
