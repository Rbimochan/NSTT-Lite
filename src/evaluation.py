"""WER/CER evaluation for NSTT."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.manifests import read_jsonl_manifest
from src.preprocessing import TARGET_SAMPLE_RATE, normalize_transcript_nfc
from src.training import LANGUAGE, TASK, WHISPER_MODEL_ID

FLEURS_CONFIG = "ne_np"


@dataclass
class EvalResult:
    dataset: str
    split: str
    num_utterances: int
    wer: float
    cer: float
    note: str = ""


@dataclass
class UtterancePrediction:
    utterance_id: str
    reference: str
    hypothesis: str
    speaker_id: str = ""
    duration_s: float = 0.0
    audio_path: str = ""


def normalize_pair_for_scoring(reference: str, hypothesis: str) -> tuple[str, str]:
    """Apply identical NFC normalization to reference and hypothesis."""
    return normalize_transcript_nfc(reference), normalize_transcript_nfc(hypothesis)


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    """Compute WER and CER after NFC-normalizing both sides."""
    refs = []
    hyps = []
    for ref, hyp in zip(references, hypotheses):
        r, h = normalize_pair_for_scoring(ref, hyp)
        refs.append(r)
        hyps.append(h)
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return wer, cer


def load_checkpoint(
    checkpoint_dir: Path,
    device: str | torch.device = "cpu",
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
    # Smoke checkpoints may lack tokenizer files; fall back to base processor.
    try:
        processor = WhisperProcessor.from_pretrained(checkpoint_dir)
    except OSError:
        processor = WhisperProcessor.from_pretrained(
            WHISPER_MODEL_ID, language=LANGUAGE, task=TASK
        )
    model.to(device)
    model.eval()
    return model, processor


def transcribe_array(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio: np.ndarray,
    sampling_rate: int,
    device: str | torch.device,
) -> str:
    if sampling_rate != TARGET_SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=TARGET_SAMPLE_RATE)
        sampling_rate = TARGET_SAMPLE_RATE

    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred_ids = model.generate(**inputs, language=LANGUAGE, task=TASK)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0]


def transcribe_file(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: Path,
    device: str | torch.device,
) -> str:
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return transcribe_array(model, processor, audio, sr, device)


def evaluate_manifest(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    manifest_path: Path,
    project_root: Path,
    device: str | torch.device,
    *,
    dataset_name: str = "SLR54",
    split_name: str = "test",
    max_samples: int | None = None,
) -> EvalResult:
    rows = read_jsonl_manifest(manifest_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    references: list[str] = []
    hypotheses: list[str] = []
    for row in rows:
        hyp = transcribe_file(
            model, processor, project_root / row["audio_path"], device
        )
        references.append(row["transcript"])
        hypotheses.append(hyp)

    wer, cer = compute_wer_cer(references, hypotheses)
    return EvalResult(
        dataset=dataset_name,
        split=split_name,
        num_utterances=len(rows),
        wer=wer,
        cer=cer,
    )


def collect_manifest_predictions(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    manifest_path: Path,
    project_root: Path,
    device: str | torch.device,
    *,
    max_samples: int | None = None,
) -> list[UtterancePrediction]:
    """Transcribe manifest rows and return per-utterance reference/hypothesis pairs."""
    rows = read_jsonl_manifest(manifest_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    predictions: list[UtterancePrediction] = []
    for row in rows:
        hyp = transcribe_file(
            model, processor, project_root / row["audio_path"], device
        )
        ref, hyp_n = normalize_pair_for_scoring(row["transcript"], hyp)
        predictions.append(
            UtterancePrediction(
                utterance_id=row["utterance_id"],
                reference=ref,
                hypothesis=hyp_n,
                speaker_id=row.get("speaker_id", ""),
                duration_s=float(row.get("duration_s", 0.0)),
                audio_path=row.get("audio_path", ""),
            )
        )
    return predictions


def load_fleurs_test(max_samples: int | None = None):
    """Load FLEURS Nepali (ne_np) test split for out-of-domain evaluation."""
    ds = load_dataset("google/fleurs", FLEURS_CONFIG, split="test", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def evaluate_hf_dataset(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    dataset,
    device: str | torch.device,
    *,
    dataset_name: str,
    split_name: str = "test",
    reference_key: str = "transcription",
    max_samples: int | None = None,
) -> EvalResult:
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    references: list[str] = []
    hypotheses: list[str] = []
    for row in dataset:
        audio = row["audio"]
        hyp = transcribe_array(
            model, processor, audio["array"], audio["sampling_rate"], device
        )
        references.append(row[reference_key])
        hypotheses.append(hyp)

    wer, cer = compute_wer_cer(references, hypotheses)
    return EvalResult(
        dataset=dataset_name,
        split=split_name,
        num_utterances=len(dataset),
        wer=wer,
        cer=cer,
        note="Out-of-domain (FLEURS ne_np)",
    )


def export_results(results: list[EvalResult], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "wer_cer_results.csv"
    md_path = reports_dir / "wer_cer_results.md"

    rows = [
        {
            "dataset": r.dataset,
            "split": r.split,
            "utterances": r.num_utterances,
            "wer": round(r.wer, 4),
            "cer": round(r.cer, 4),
            "note": r.note,
        }
        for r in results
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "split", "utterances", "wer", "cer", "note"]
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# WER/CER Evaluation Results",
        "",
        "> **Note:** If the checkpoint is from the T-003 smoke test (3 training steps), "
        "these numbers are pipeline sanity checks only — not representative of a fully trained model.",
        "",
        "| Dataset | Split | Utterances | WER | CER | Note |",
        "|---------|-------|------------|-----|-----|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['split']} | {r['utterances']} | "
            f"{r['wer']:.2%} | {r['cer']:.2%} | {r['note']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def run_evaluation(
    project_root: Path,
    checkpoint_dir: Path,
    *,
    device: str | None = None,
    max_in_domain: int | None = None,
    max_out_of_domain: int | None = 5,
    include_fleurs: bool = True,
) -> list[EvalResult]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_checkpoint(checkpoint_dir, device=device)

    results: list[EvalResult] = []
    results.append(
        evaluate_manifest(
            model,
            processor,
            project_root / "data" / "manifests" / "test.jsonl",
            project_root,
            device,
            dataset_name="SLR54",
            split_name="test",
            max_samples=max_in_domain,
        )
    )

    if include_fleurs:
        fleurs = load_fleurs_test(max_samples=max_out_of_domain)
        results.append(
            evaluate_hf_dataset(
                model,
                processor,
                fleurs,
                device,
                dataset_name="FLEURS",
                split_name="test",
                reference_key="transcription",
            )
        )

    export_results(results, project_root / "reports")
    return results
