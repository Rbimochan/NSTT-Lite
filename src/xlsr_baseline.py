"""XLS-R (wav2vec2) baseline audit for gagan3012/wav2vec2-xlsr-nepali on OpenSLR-43.

This project has one purpose: evaluate this model on OpenSLR-43
(gauravparajuli/slr43), the same corpus it was originally trained/self-evaluated
on, to sanity-check its published 5.97% WER claim. No other model, no other
corpus, no fine-tuning track.

OpenSLR-43 has no speaker/gender metadata (verified: HF dataset_info only lists
`audio` and `text` features) and is a single-speaker (female) corpus -- no
demographic breakdown is possible or attempted here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

XLSR_MODEL_ID = "gagan3012/wav2vec2-xlsr-nepali"
OPENSLR43_DATASET_ID = "gauravparajuli/slr43"
TARGET_SAMPLE_RATE = 16_000
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"\“]'
SELF_REPORTED_WER = 0.0597


def load_xlsr_model_and_processor(
    model_id: str = XLSR_MODEL_ID,
) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    return model, processor


def load_openslr43_test_dataset(max_samples: int | None = None) -> Dataset:
    """Only a 'train' split exists upstream -- there is no separate held-out
    test split, so this project treats a fixed-seed slice of it as the
    in-domain eval set."""
    dataset = load_dataset(OPENSLR43_DATASET_ID, split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def clean_transcript(text: str) -> str:
    return re.sub(CHARS_TO_IGNORE_REGEX, "", text).lower()


def resample_if_needed(speech_array: np.ndarray, sampling_rate: int) -> np.ndarray:
    if sampling_rate == TARGET_SAMPLE_RATE:
        return speech_array
    import torchaudio

    resampled = torchaudio.functional.resample(
        torch.tensor(speech_array), sampling_rate, TARGET_SAMPLE_RATE
    )
    return resampled.numpy()


def transcribe_ctc(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    speech_array: np.ndarray,
    device: str,
) -> str:
    """Greedy CTC decode -- XLS-R/wav2vec2 is a CTC model, not seq2seq."""
    inputs = processor(speech_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def build_openslr43_manifest_rows(dataset: Dataset) -> list[dict]:
    """CSV-manifest rows referencing the HF dataset by index rather than storing
    audio locally -- raw audio stays out of git."""
    rows = []
    for i, row in enumerate(dataset):
        audio = row["audio"]
        duration_s = len(audio["array"]) / audio["sampling_rate"]
        rows.append(
            {
                "utterance_id": f"slr43_{i:05d}",
                "hf_dataset": OPENSLR43_DATASET_ID,
                "hf_index": i,
                "text": clean_transcript(row["text"]),
                "duration_s": round(duration_s, 4),
            }
        )
    return rows


def write_openslr43_manifest(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "hf_dataset", "hf_index", "text", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)
