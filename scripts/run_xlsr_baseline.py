"""XLS-R baseline audit: gagan3012/wav2vec2-xlsr-nepali evaluated (1) in-domain
on OpenSLR-43 (its own training corpus, single female speaker) and (2) on
NSTT-Lite's existing SLR54 test manifest as a separate, out-of-domain diversity
check. These two datasets and their results are kept fully separate throughout --
never merged into one "OpenSLR" number.

Mirrors scripts/run_baseline.py's structure for the existing Whisper zero-shot
baseline, adapted for a CTC model instead of a seq2seq one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import compute_wer_cer
from src.manifests import read_jsonl_manifest
from src.xlsr_baseline import (
    XLSR_MODEL_ID,
    build_openslr43_manifest_rows,
    load_openslr43_test_dataset,
    load_xlsr_model_and_processor,
    resample_if_needed,
    transcribe_ctc,
    write_openslr43_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_SLR43_SAMPLES = 150   # matches this repo's existing 150-utterance eval convention
N_SLR54_SAMPLES = 150


def eval_openslr43(model, processor, device: str) -> dict:
    """In-domain: same corpus the model was originally trained/self-evaluated on."""
    dataset = load_openslr43_test_dataset(max_samples=N_SLR43_SAMPLES)
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
            print(f"  [OpenSLR-43] {i + 1}/{len(dataset)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"[OpenSLR-43, in-domain] WER={wer:.4f} CER={cer:.4f} on {len(dataset)} utterances")
    return {
        "dataset": "OpenSLR-43 (gauravparajuli/slr43)",
        "role": "in-domain (model's own training/self-eval corpus)",
        "num_utterances": len(dataset),
        "wer": wer,
        "cer": cer,
        "demographic_breakdown": None,
        "demographic_note": (
            "OpenSLR-43 has no speaker/gender metadata and is a single-speaker "
            "(female) corpus -- no within-corpus demographic breakdown is possible."
        ),
        "published_wer_comparison": {
            "self_reported_wer": 0.0597,
            "measured_wer": wer,
            "note": "measured on a 150-utterance slice of the same corpus, not an independent test set",
        },
    }


def eval_slr54_diversity_check(model, processor, device: str) -> dict:
    """Out-of-domain: a different, multi-speaker OpenSLR corpus (SLR54), reusing
    NSTT-Lite's existing test manifest and its F0-pseudo-label gender field.

    test.jsonl is ordered by speaker, so a plain [:N] slice can land entirely
    within one speaker's gender by chance (it did: the first 150 rows were
    100% "male"). Shuffle with this repo's standard seed=42 before sampling so
    the demographic breakdown actually has both classes represented."""
    import random

    all_rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
    shuffled = list(all_rows)
    random.Random(42).shuffle(shuffled)
    rows = shuffled[:N_SLR54_SAMPLES]

    import soundfile as sf

    references, hypotheses, genders = [], [], []
    for i, row in enumerate(rows):
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        speech = resample_if_needed(audio, sr)
        hyp = transcribe_ctc(model, processor, speech, device)
        references.append(row["transcript"])
        hypotheses.append(hyp)
        genders.append(row["gender"])
        if (i + 1) % 25 == 0:
            print(f"  [SLR54 diversity check] {i + 1}/{len(rows)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"[OpenSLR-54, out-of-domain diversity check] WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances")

    breakdown = {}
    for label in ("male", "female"):
        idx = [i for i, g in enumerate(genders) if g == label]
        if not idx:
            continue
        sub_wer, sub_cer = compute_wer_cer([references[i] for i in idx], [hypotheses[i] for i in idx])
        breakdown[label] = {"num_utterances": len(idx), "wer": sub_wer, "cer": sub_cer}

    return {
        "dataset": "OpenSLR-54 (this repo's existing Whisper-Small corpus)",
        "role": "out-of-domain diversity check -- DIFFERENT corpus from OpenSLR-43, not directly comparable",
        "num_utterances": len(rows),
        "wer": wer,
        "cer": cer,
        "demographic_breakdown": breakdown,
        "demographic_note": (
            "Breakdown uses NSTT-Lite's `gender` field, an F0-pitch acoustic "
            "pseudo-label (165Hz threshold), NOT verified/ground-truth gender. "
            "Results measure agreement with that heuristic, not true gender."
        ),
    }


def main() -> None:
    device = "cpu"
    print(f"Loading {XLSR_MODEL_ID}...")
    model, processor = load_xlsr_model_and_processor()
    model.to(device)

    slr43_result = eval_openslr43(model, processor, device)
    slr54_result = eval_slr54_diversity_check(model, processor, device)

    results = {
        "model": "gagan3012/wav2vec2-xlsr-nepali",
        "model_family": "XLS-R / wav2vec2 (CTC) -- NOT the same architecture as this repo's Whisper-Small track",
        "purpose": (
            "One-time comparison baseline to sanity-check the model's self-reported "
            "5.97% WER and inform the backbone choice already made in Plans 1-4. "
            "Not a second fine-tuning/deployment track."
        ),
        "openslr43_in_domain": slr43_result,
        "openslr54_diversity_check": slr54_result,
        "corpus_separation_note": (
            "OpenSLR-43 and OpenSLR-54 are DIFFERENT corpora and are never merged "
            "into a single 'OpenSLR' number anywhere in this report."
        ),
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "xlsr_baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
