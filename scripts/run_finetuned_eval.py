"""Step 9: evaluate the fine-tuned checkpoint on the real test set, same
methodology as the Step 5 zero-shot baseline (150 test utterances), for a
fair apples-to-apples WER/CER comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import EvalResult, compute_wer_cer, export_results, load_checkpoint, transcribe_file
from src.manifests import read_jsonl_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_TEST_UTTERANCES = 150


def main(checkpoint_dir: Path, dataset_name: str, output_suffix: str) -> None:
    device = "cpu"
    model, processor = load_checkpoint(checkpoint_dir, device=device)

    rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")[:N_TEST_UTTERANCES]
    print(f"Transcribing {len(rows)} test utterances with {checkpoint_dir.name}...")

    references, hypotheses = [], []
    for i, row in enumerate(rows):
        hyp = transcribe_file(model, processor, PROJECT_ROOT / row["audio_path"], device)
        references.append(row["transcript"])
        hypotheses.append(hyp)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"{dataset_name}: WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances")

    reports_dir = PROJECT_ROOT / "reports"
    result = EvalResult(
        dataset="SLR54", split="test", num_utterances=len(rows), wer=wer, cer=cer, note=dataset_name
    )
    csv_path, md_path = export_results([result], reports_dir)

    summary_path = reports_dir / f"eval_{output_suffix}.json"
    summary_path.write_text(
        json.dumps({"wer": wer, "cer": cer, "num_utterances": len(rows), "note": dataset_name}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    ckpt = PROJECT_ROOT / "models" / "finetuned"
    main(ckpt, "Fine-tuned (speaker-disjoint split, 300 steps)", "finetuned")
