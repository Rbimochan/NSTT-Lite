"""XLS-R baseline audit: gagan3012/wav2vec2-xlsr-nepali evaluated in-domain on
OpenSLR-43 (gauravparajuli/slr43) -- the same corpus it was originally
trained/self-evaluated on. Sanity-checks the model's published 5.97% WER claim.

Single purpose, single model, single dataset -- no other tracks in this project.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.wer_metrics import compute_wer_cer
from src.xlsr_baseline import (
    SELF_REPORTED_WER,
    build_openslr43_manifest_rows,
    load_openslr43_test_dataset,
    load_xlsr_model_and_processor,
    resample_if_needed,
    transcribe_ctc,
    write_openslr43_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SAMPLES = 150


def main() -> None:
    device = "cpu"
    print("Loading gagan3012/wav2vec2-xlsr-nepali...")
    model, processor = load_xlsr_model_and_processor()
    model.to(device)

    dataset = load_openslr43_test_dataset(max_samples=N_SAMPLES)
    rows = build_openslr43_manifest_rows(dataset)
    write_openslr43_manifest(rows, PROJECT_ROOT / "data" / "manifests" / "xlsr_openslr43_test_manifest.csv")

    references, hypotheses = [], []
    for i, row in enumerate(dataset):
        audio = row["audio"]
        speech = resample_if_needed(audio["array"], audio["sampling_rate"])
        hyp = transcribe_ctc(model, processor, speech, device)
        references.append(rows[i]["text"])
        hypotheses.append(hyp)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dataset)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"WER={wer:.4f} CER={cer:.4f} on {len(dataset)} utterances")

    results = {
        "model": "gagan3012/wav2vec2-xlsr-nepali",
        "dataset": "OpenSLR-43 (gauravparajuli/slr43)",
        "num_utterances": len(dataset),
        "wer": wer,
        "cer": cer,
        "self_reported_wer": SELF_REPORTED_WER,
        "note": (
            "Measured on a fixed-seed slice of the corpus (no independent held-out "
            "test split exists upstream). Single-speaker (female) corpus, no "
            "gender/demographic metadata -- no breakdown possible or attempted."
        ),
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "xlsr_baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
