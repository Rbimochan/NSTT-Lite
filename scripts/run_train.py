"""Phase 6: Whisper-small fine-tuning (smoke test or full Colab run)."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.training import train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def device_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on NSTT-Lite manifests")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short end-to-end smoke test (few steps, small subset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint/output directory (default: models/whisper-small-ft[-smoke])",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path, or 'true' to auto-resume latest in output-dir",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models" / (
            "whisper-small-ft-smoke" if args.smoke else "whisper-small-ft"
        )
    else:
        output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir

    resume: str | bool | None = args.resume
    if resume == "true":
        resume = True

    print("Device:", json.dumps(device_info(), indent=2))
    print(f"Output: {output_dir}")
    print(f"Mode: {'smoke' if args.smoke else 'full'}")

    result = train_and_save(
        PROJECT_ROOT,
        output_dir,
        smoke_test=args.smoke,
        resume_from_checkpoint=resume,
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "smoke" if args.smoke else "full"
    summary_path = reports_dir / f"phase6_train_{tag}_{stamp}.json"
    payload = {
        "created_at": stamp,
        "device": device_info(),
        **result,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
