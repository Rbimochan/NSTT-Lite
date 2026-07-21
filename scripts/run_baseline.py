"""Step 5: zero-shot Whisper-small baseline WER/CER + majority-class gender baseline."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.evaluation import compute_wer_cer, export_results, EvalResult
from src.manifests import read_jsonl_manifest
from src.training import LANGUAGE, TASK, WHISPER_MODEL_ID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_TEST_UTTERANCES = 150


def transcribe(model, processor, audio_path: Path, device: str) -> str:
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred_ids = model.generate(**inputs, language=LANGUAGE, task=TASK)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


def main() -> None:
    device = "cpu"
    print(f"Loading {WHISPER_MODEL_ID} (zero-shot, no fine-tuning)...")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_ID, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
    model.to(device)
    model.eval()

    rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")[:N_TEST_UTTERANCES]
    print(f"Transcribing {len(rows)} test utterances (zero-shot baseline)...")

    references, hypotheses, output_rows = [], [], []
    for i, row in enumerate(rows):
        audio_path = PROJECT_ROOT / row["audio_path"]
        hyp = transcribe(model, processor, audio_path, device)
        references.append(row["transcript"])
        hypotheses.append(hyp)
        output_rows.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "gender": row["gender"],
                "reference": row["transcript"],
                "hypothesis": hyp,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")

    wer, cer = compute_wer_cer(references, hypotheses)
    print(f"Zero-shot baseline: WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    result = EvalResult(
        dataset="SLR54",
        split="test",
        num_utterances=len(rows),
        wer=wer,
        cer=cer,
        note="Zero-shot baseline (openai/whisper-small, no fine-tuning)",
    )
    export_results([result], reports_dir)

    with (reports_dir / "baseline_asr_examples.jsonl").open("w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Majority-class gender baseline over the full test split (not just the 150-sample subset)
    all_test_rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
    gender_counts = Counter(r["gender"] for r in all_test_rows)
    majority_gender, majority_count = gender_counts.most_common(1)[0]
    baseline_acc = majority_count / len(all_test_rows)

    baseline_summary = {
        "majority_gender": majority_gender,
        "gender_counts": dict(gender_counts),
        "total_test_utterances": len(all_test_rows),
        "majority_class_accuracy": round(baseline_acc, 4),
        "note": "Predicts the majority F0-pseudo-label gender for every utterance; not a trained classifier.",
    }
    (reports_dir / "baseline_gender.json").write_text(
        json.dumps(baseline_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Majority-class gender baseline: {majority_gender} -> accuracy={baseline_acc:.4f}")
    print("Saved: reports/wer_cer_results.{csv,md}, reports/baseline_asr_examples.jsonl, reports/baseline_gender.json")


if __name__ == "__main__":
    main()
