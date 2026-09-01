"""Phase 9 — Error Analysis.

Runs the fine-tuned checkpoint on the same 150-utterance OpenSLR-54
out-of-domain sample used throughout (seed=42 shuffle, matching Phase 4/7),
and produces:
  - per-utterance error categories (heuristic, jiwer-alignment based)
  - per-speaker WER spread
  - WER by gender PSEUDO-LABEL group (F0-threshold, not verified gender)
  - worst-N examples table
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.manifests import read_jsonl_manifest
from src.wer_metrics import compute_wer_cer
from src.xlsr_baseline import clean_transcript, resample_if_needed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "xlsr-ft"
N_SAMPLES = 150
SEED = 42

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
ERROR_CATEGORIES = [
    "phonetic_confusion",
    "oov_rare_vocabulary",
    "noise_degradation",
    "code_switching",
    "dialect_accent",
    "other",
]


def transcribe(model, processor, speech, device: str) -> str:
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def categorize_errors(reference: str, hypothesis: str) -> list[str]:
    """Heuristic error tagging -- same conventions used earlier in this
    project's history (src/error_analysis.py, Whisper track, now ported)."""
    ref, hyp = reference.strip(), hypothesis.strip()
    categories: list[str] = []

    if LATIN_RE.search(ref):
        categories.append("code_switching")
    if not hyp:
        categories.append("noise_degradation")
    elif ref and jiwer.wer([ref], [hyp]) >= 0.95:
        categories.append("noise_degradation")

    ref_words = set(ref.split())
    hyp_words = set(hyp.split())
    missing = ref_words - hyp_words
    if missing and any(len(w) >= 4 for w in missing):
        categories.append("oov_rare_vocabulary")

    if DEVANAGARI_RE.search(ref) and ref != hyp:
        alignment = jiwer.process_characters(ref, hyp)
        if alignment.substitutions > 0:
            categories.append("phonetic_confusion")
        if alignment.deletions > len(ref) * 0.3:
            categories.append("dialect_accent")

    if not categories:
        categories.append("other")
    return sorted(set(categories))


def main() -> None:
    device = "cpu"
    processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_DIR)
    model.to(device).eval()

    all_rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
    shuffled = list(all_rows)
    random.Random(SEED).shuffle(shuffled)
    rows = shuffled[:N_SAMPLES]

    per_utterance = []
    for i, row in enumerate(rows):
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        speech = resample_if_needed(audio, sr)
        ref = clean_transcript(row["transcript"])
        hyp = transcribe(model, processor, speech, device)
        wer, cer = compute_wer_cer([ref], [hyp])
        categories = categorize_errors(ref, hyp)
        per_utterance.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "gender_pseudo_label": row["gender"],
                "reference": ref,
                "hypothesis": hyp,
                "wer": wer,
                "cer": cer,
                "categories": categories,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    # Aggregate: error category counts
    category_counts = Counter()
    for u in per_utterance:
        for c in u["categories"]:
            category_counts[c] += 1

    # Aggregate: per-speaker WER
    by_speaker = defaultdict(list)
    for u in per_utterance:
        by_speaker[u["speaker_id"]].append(u["wer"])
    speaker_wer = {
        spk: {"mean_wer": sum(ws) / len(ws), "n": len(ws)}
        for spk, ws in by_speaker.items()
    }
    speaker_wers_sorted = sorted(speaker_wer.items(), key=lambda kv: kv[1]["mean_wer"])

    # Aggregate: WER by gender pseudo-label group
    by_gender = defaultdict(lambda: {"refs": [], "hyps": []})
    for u in per_utterance:
        by_gender[u["gender_pseudo_label"]]["refs"].append(u["reference"])
        by_gender[u["gender_pseudo_label"]]["hyps"].append(u["hypothesis"])
    gender_breakdown = {}
    for label, d in by_gender.items():
        wer, cer = compute_wer_cer(d["refs"], d["hyps"])
        gender_breakdown[label] = {"wer": wer, "cer": cer, "num_utterances": len(d["refs"])}

    # Worst 10 examples by WER
    worst = sorted(per_utterance, key=lambda u: u["wer"], reverse=True)[:10]

    overall_wer, overall_cer = compute_wer_cer(
        [u["reference"] for u in per_utterance], [u["hypothesis"] for u in per_utterance]
    )

    results = {
        "checkpoint": str(CHECKPOINT_DIR),
        "num_utterances": len(per_utterance),
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
        "error_category_counts": dict(category_counts),
        "error_category_note": (
            "Heuristic, regex/jiwer-alignment-based tagging -- categories can overlap "
            "per utterance and are not mutually exclusive."
        ),
        "gender_pseudo_label_breakdown": gender_breakdown,
        "gender_pseudo_label_note": (
            "gender_pseudo_label is an F0-pitch threshold heuristic (165Hz), NOT verified "
            "ground truth -- this breakdown measures WER by acoustic pitch-threshold group, "
            "not by true gender."
        ),
        "speaker_wer_spread": {
            "min": speaker_wers_sorted[0],
            "max": speaker_wers_sorted[-1],
            "num_speakers": len(speaker_wer),
        },
        "worst_10_examples": worst,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase9_error_analysis_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "phase9_per_utterance.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in per_utterance), encoding="utf-8"
    )

    print(f"\nOverall WER={overall_wer:.4f} CER={overall_cer:.4f}")
    print("Error categories:", dict(category_counts))
    print("Gender pseudo-label breakdown:", gender_breakdown)
    print(f"Speaker WER spread: best={speaker_wers_sorted[0]}, worst={speaker_wers_sorted[-1]}")
    print("Saved: reports/phase9_error_analysis_results.json, reports/phase9_per_utterance.jsonl")


if __name__ == "__main__":
    main()
