"""XLS-R (wav2vec2) baseline audit for gagan3012/wav2vec2-xlsr-nepali.

Migrated from a standalone Colab notebook (Cells 1-4: dataset load, model load,
preprocessing) into this repo's src/ convention, following src/slr54.py's shape.

Motivation: the model's published 5.97% WER is self-reported on OpenSLR-43, a
single-speaker (female) Nepali corpus -- the same corpus used for its own
training/eval split. This module evaluates that claim in-domain (SLR43) and,
separately, checks whether it holds on a different, multi-speaker corpus
(NSTT-Lite's existing SLR54 test manifest) as a diversity sanity check.

IMPORTANT -- do not conflate the two datasets or their results:
- OpenSLR-43 (gauravparajuli/slr43) is a DIFFERENT OpenSLR corpus from OpenSLR-54,
  which the rest of this repo uses for Whisper-Small fine-tuning. Keep file/report
  naming distinct (xlsr_openslr43_* vs the existing slr54-based manifests).
- OpenSLR-43 has no speaker/gender metadata at all (verified: HF dataset_info
  only lists `audio` and `text` features) -- it is a single-speaker corpus, so no
  demographic WER breakdown is possible on it.
- SLR54's `gender` field (used for the diversity check below) is itself an F0-pitch
  acoustic pseudo-label, not ground truth (see src/preprocessing.py). Any WER
  breakdown by that field measures agreement with the heuristic, not true gender.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

XLSR_MODEL_ID = "gagan3012/wav2vec2-xlsr-nepali"
OPENSLR43_DATASET_ID = "gauravparajuli/slr43"
TARGET_SAMPLE_RATE = 16_000
CHARS_TO_IGNORE_REGEX = r'[\,\?\.\!\-\;\:\"\“]'


@dataclass
class Slr43Utterance:
    hf_index: int
    text: str
    duration_s: float


def load_xlsr_model_and_processor(
    model_id: str = XLSR_MODEL_ID,
) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    return model, processor


def load_openslr43_test_dataset(max_samples: int | None = None) -> Dataset:
    """Load OpenSLR-43 (single-speaker, female Nepali corpus). Only a 'train'
    split exists upstream -- there is no separate held-out test split, so this
    project treats a fixed-seed slice of it as the in-domain eval set."""
    dataset = load_dataset(OPENSLR43_DATASET_ID, split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def clean_transcript(text: str) -> str:
    """Adapted from the original notebook's Cell 4 preprocessing."""
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
    """Greedy CTC decode -- XLS-R/wav2vec2 is a CTC model, not seq2seq, so this
    does not use generate() the way the Whisper scripts elsewhere in this repo do."""
    inputs = processor(speech_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def build_openslr43_manifest_rows(dataset: Dataset) -> list[dict]:
    """CSV-manifest rows referencing the HF dataset by index rather than storing
    audio locally -- raw audio stays out of git, per this repo's convention."""
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
