"""Plan 4, Step 2: evaluate a checkpoint on the FULL test split (not the
150-utterance local-tractability subset used in Plan 3). Run on Colab GPU --
transcribing 1,675 utterances with generate() on CPU would take hours.

Usage (from Colab, after run_colab_training.py / run_colab_ablation_training.py):
    NSTT_CHECKPOINT_DIR=models/finetuned_full NSTT_MANIFEST_DIR=data/manifests \
        NSTT_EVAL_NAME=finetuned_full python scripts/run_full_test_eval.py
    NSTT_CHECKPOINT_DIR=models/finetuned_leaky_full NSTT_MANIFEST_DIR=data/manifests_leaky \
        NSTT_EVAL_NAME=leaky_full python scripts/run_full_test_eval.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import EvalResult, compute_wer_cer, export_results, load_checkpoint, transcribe_file
from src.manifests import read_jsonl_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checkpoint_dir = Path(os.environ.get("NSTT_CHECKPOINT_DIR", PROJECT_ROOT / "models" / "finetuned_full"))
    manifest_dir = Path(os.environ.get("NSTT_MANIFEST_DIR", PROJECT_ROOT / "data" / "manifests"))
    eval_name = os.environ.get("NSTT_EVAL_NAME", "finetuned_full")

    device = "cuda"
    model, processor = load_checkpoint(checkpoint_dir, device=device)

    rows = read_jsonl_manifest(manifest_dir / "test.jsonl")
    print(f"Transcribing FULL test split: {len(rows)} utterances with {checkpoint_dir.name}...")

    references, hypotheses = [], []
    for i, row in enumerate(rows):
        hyp = transcribe_file(model, processor, PROJECT_ROOT / row["audio_path"], device)
        references.append(row["transcript"])
        hypotheses.append(hyp)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"{eval_name}: WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances (FULL test split)")

    reports_dir = PROJECT_ROOT / "reports"
    result = EvalResult(
        dataset="SLR54", split="test", num_utterances=len(rows), wer=wer, cer=cer,
        note=f"{eval_name} (full test split)",
    )
    export_results([result], reports_dir)

    summary_path = reports_dir / f"eval_{eval_name}_full.json"
    summary_path.write_text(
        json.dumps({"wer": wer, "cer": cer, "num_utterances": len(rows), "note": eval_name}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
