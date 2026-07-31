"""Phase 3: continue fine-tuning the audited xlsr-ft checkpoint (already
Nepali-general fine-tuned) on the small Nepali-English medical corpus
(pilot + batch2 + batch3, 125 clips / ~8.6 min, 5 speakers) built by
build_medical_combined_manifest.py.

This is a first small-scale continuation run, not the final model --
per data/medical_corpus_README.md, 5 speakers total is still below the
3-5-per-gender / 1-2-hour target. Treat this as validating the pipeline
and getting an early read on term-survival improvement, not a finished
checkpoint.

Usage:
    python scripts/run_medical_finetune.py            # full (small) run, local CPU/MPS
    python scripts/run_medical_finetune.py --smoke     # quick sanity check only
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import torch

from src.xlsr_training import train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = str(PROJECT_ROOT / "models" / "xlsr-ft")
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests_medical_combined"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
        or (PROJECT_ROOT / ("models/xlsr-medical-ft-smoke" if args.smoke else "models/xlsr-medical-ft"))
    )

    info = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    print("Device:", json.dumps(info, indent=2))
    print("Base checkpoint:", BASE_CHECKPOINT)
    print("Manifest dir:", MANIFEST_DIR)
    print("Output:", output_dir)

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("phase3-medical-finetune")

    result = train_and_save(
        PROJECT_ROOT,
        output_dir,
        smoke_test=args.smoke,
        seed=args.seed,
        manifest_dir=MANIFEST_DIR,
        model_id=BASE_CHECKPOINT,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"created_at": stamp, "device": info, "smoke_test": args.smoke, "manifest_dir": str(MANIFEST_DIR), **result}
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = "medical_smoke" if args.smoke else "medical_full"
    out = reports_dir / f"phase3_train_{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print("train_loss:", result["train_loss"])
    print("eval_metrics:", result["eval_metrics"])
    print("global_step:", result["global_step"])
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
