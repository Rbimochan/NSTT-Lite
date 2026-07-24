"""Phase 4: zero-shot ASR baseline WER/CER on the real speaker-disjoint test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.evaluation import compute_wer_cer, export_results, EvalResult
from src.manifests import read_jsonl_manifest
from src.training import LANGUAGE, TASK

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def transcribe(model, processor, audio_path: Path, device: str) -> str:
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred_ids = model.generate(**inputs, language=LANGUAGE, task=TASK)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


def main(model_id: str, n_samples: int) -> None:
    device = "cpu"
    print(f"Loading {model_id} (zero-shot, no fine-tuning)...")
    processor = WhisperProcessor.from_pretrained(model_id, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()

    rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")[:n_samples]
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
    print(f"{model_id}: WER={wer:.4f} CER={cer:.4f} on {len(rows)} utterances")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_tag = model_id.split("/")[-1]
    result = EvalResult(
        dataset="SLR54", split="test", num_utterances=len(rows), wer=wer, cer=cer,
        note=f"Zero-shot baseline ({model_id}, no fine-tuning)",
    )
    export_results([result], reports_dir)

    with (reports_dir / f"baseline_{model_tag}_examples.jsonl").open("w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = reports_dir / f"baseline_{model_tag}.json"
    summary_path.write_text(
        json.dumps({"model": model_id, "wer": wer, "cer": cer, "num_utterances": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--n_samples", type=int, default=150)
    args = parser.parse_args()
    main(args.model, args.n_samples)
