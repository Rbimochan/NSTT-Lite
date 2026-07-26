"""Phase 4 — Baseline Establishment (The Audit).

Evaluates gagan3012/wav2vec2-xlsr-nepali zero-shot on:
  (a) OpenSLR-43 (its own single-speaker training corpus) -- in-domain
      reproduction of the self-reported 5.97% WER claim;
  (b) this project's speaker-disjoint OpenSLR-54 test split -- out-of-domain
      generalization check.

Both runs are logged to MLflow (local ./mlruns store): params, WER/CER,
and per-utterance example artifacts. The two corpora are never merged.

Note: data/manifests/test.jsonl is speaker-ordered, so the SLR54 sample is
drawn after a seeded shuffle -- a plain [:N] slice would land on very few
speakers (this exact bug happened once before in this repo's history).
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import soundfile as sf
import torch

from src.manifests import read_jsonl_manifest
from src.wer_metrics import compute_wer_cer
from src.xlsr_baseline import (
    SELF_REPORTED_WER,
    XLSR_MODEL_ID,
    build_openslr43_manifest_rows,
    load_openslr43_test_dataset,
    load_xlsr_model_and_processor,
    resample_if_needed,
    transcribe_ctc,
    write_openslr43_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENT_NAME = "phase4-audit"


def device_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
    }


def eval_slr43(model, processor, device: str, n_samples: int) -> dict:
    dataset = load_openslr43_test_dataset(max_samples=n_samples)
    rows = build_openslr43_manifest_rows(dataset)
    write_openslr43_manifest(rows, PROJECT_ROOT / "data" / "manifests" / "xlsr_openslr43_test_manifest.csv")

    references, hypotheses, examples = [], [], []
    for i, row in enumerate(dataset):
        audio = row["audio"]
        speech = resample_if_needed(audio["array"], audio["sampling_rate"])
        hyp = transcribe_ctc(model, processor, speech, device)
        references.append(rows[i]["text"])
        hypotheses.append(hyp)
        examples.append({"utterance_id": rows[i]["utterance_id"], "reference": rows[i]["text"], "hypothesis": hyp})
        if (i + 1) % 25 == 0:
            print(f"  [SLR43 in-domain] {i + 1}/{len(dataset)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"[SLR43 in-domain] WER={wer:.4f} CER={cer:.4f} on {len(dataset)} utterances")
    return {"dataset": "OpenSLR-43 (in-domain)", "num_utterances": len(dataset), "wer": wer, "cer": cer, "examples": examples}


def eval_slr54(model, processor, device: str, n_samples: int, seed: int) -> dict:
    all_rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
    shuffled = list(all_rows)
    random.Random(seed).shuffle(shuffled)
    rows = shuffled[:n_samples] if n_samples else shuffled

    references, hypotheses, examples = [], [], []
    for i, row in enumerate(rows):
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        speech = resample_if_needed(audio, sr)
        hyp = transcribe_ctc(model, processor, speech, device)
        references.append(row["transcript"])
        hypotheses.append(hyp)
        examples.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "gender_pseudo_label": row["gender"],
                "reference": row["transcript"],
                "hypothesis": hyp,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  [SLR54 out-of-domain] {i + 1}/{len(rows)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"[SLR54 out-of-domain] WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances")
    return {"dataset": "OpenSLR-54 speaker-disjoint test (out-of-domain)", "num_utterances": len(rows), "wer": wer, "cer": cer, "examples": examples}


def log_run(name: str, result: dict, params: dict) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({"wer": result["wer"], "cer": result["cer"]})
        examples_path = REPORTS_DIR / f"phase4_{name}_examples.jsonl"
        with examples_path.open("w", encoding="utf-8") as f:
            for ex in result["examples"]:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        mlflow.log_artifact(str(examples_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = device_info()
    print("Device:", json.dumps(info, indent=2))

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {XLSR_MODEL_ID}...")
    model, processor = load_xlsr_model_and_processor()
    model.to(device)

    slr43 = eval_slr43(model, processor, device, args.n_samples)
    slr54 = eval_slr54(model, processor, device, args.n_samples, args.seed)

    common = {"model": XLSR_MODEL_ID, "n_samples": args.n_samples, "device": device, **info}
    log_run("slr43_in_domain", slr43, {**common, "dataset": "gauravparajuli/slr43"})
    log_run("slr54_out_of_domain", slr54, {**common, "dataset": "OpenSLR-54 test.jsonl", "seed": args.seed})

    summary = {
        "model": XLSR_MODEL_ID,
        "self_reported_wer": SELF_REPORTED_WER,
        "device": info,
        "in_domain_openslr43": {k: v for k, v in slr43.items() if k != "examples"},
        "out_of_domain_openslr54": {k: v for k, v in slr54.items() if k != "examples"},
        "corpus_separation_note": "OpenSLR-43 and OpenSLR-54 results are never merged into a single number.",
        "slr54_gender_note": (
            "SLR54 rows carry an F0-pitch gender pseudo-label (165Hz threshold), "
            "not verified metadata -- used only for Phase 9 per-group analysis."
        ),
    }
    out_path = REPORTS_DIR / "phase4_audit_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")
    print("MLflow store: ./mlruns  (view with: mlflow ui)")


if __name__ == "__main__":
    main()
