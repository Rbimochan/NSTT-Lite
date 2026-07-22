"""Step 9 ablation: evaluate the leaky-split checkpoint on its own test split,
same methodology as run_finetuned_eval.py, then record the WER gap vs the
speaker-disjoint fine-tuned result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import EvalResult, compute_wer_cer, export_results, load_checkpoint, transcribe_file
from src.manifests import read_jsonl_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_TEST_UTTERANCES = 150
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "finetuned_leaky"


def main() -> None:
    device = "cpu"
    model, processor = load_checkpoint(CHECKPOINT_DIR, device=device)

    rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests_leaky" / "test.jsonl")[:N_TEST_UTTERANCES]
    print(f"Transcribing {len(rows)} leaky-split test utterances...")

    references, hypotheses = [], []
    for i, row in enumerate(rows):
        hyp = transcribe_file(model, processor, PROJECT_ROOT / row["audio_path"], device)
        references.append(row["transcript"])
        hypotheses.append(hyp)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    note = "Fine-tuned (leaky utterance-random split, 100 steps ablation)"
    print(f"Leaky-split ablation: WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances")

    reports_dir = PROJECT_ROOT / "reports"
    result = EvalResult(dataset="SLR54", split="test", num_utterances=len(rows), wer=wer, cer=cer, note=note)
    export_results([result], reports_dir)

    speaker_disjoint = json.loads((reports_dir / "eval_finetuned.json").read_text())
    comparison = {
        "speaker_disjoint": {**speaker_disjoint, "train_steps": 300},
        "leaky_utterance_random": {"wer": wer, "cer": cer, "num_utterances": len(rows), "train_steps": 100},
        "wer_gap_leaky_minus_disjoint": round(wer - speaker_disjoint["wer"], 4),
        "caveat": (
            "Step counts differ (300 vs 100) for local tractability, so this gap conflates "
            "speaker leakage with unequal training budget -- directionally informative, not a "
            "controlled ablation. A fair comparison would retrain both at identical step counts."
        ),
    }
    (reports_dir / "ablation_wer_gap.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
