# Appendix A — Project Proposal (Verbatim)

## Project Proposal — NSTT-Lite: Auditing and Repairing a Published Nepali ASR Model

**Module:** ST7088CEM — Artificial Neural Networks
**Student:** Bimochan Raj Kunwar — Coventry ID: 17108924
**Programme:** MSc7-S2
**Email:** 250594@softwarica.edu.np

### Problem

`gagan3012/wav2vec2-xlsr-nepali` is a published Hugging Face XLS-R (wav2vec2)
model for Nepali speech recognition that self-reports **5.97% WER** — a
remarkably strong number for a low-resource language. However, that figure was
measured on the model's own training corpus, OpenSLR-43, which is effectively
single-speaker (one female voice). My preliminary experiments confirm the claim
technically holds in-domain (I measured **4.91% WER** on OpenSLR-43) but the
same model collapses to roughly **65% WER** on multi-speaker Nepali speech
(OpenSLR-54) — a >13x degradation. The published benchmark therefore does not
describe real-world performance. This project audits that claim rigorously and
then repairs the model.

### Tasks

1. **Benchmark audit.** Reproduce the self-reported figure in-domain
   (OpenSLR-43) and measure the same checkpoint zero-shot on a speaker-disjoint
   multi-speaker test split (OpenSLR-54), quantifying the generalization gap.
2. **Fine-tuning.** Fine-tune the same XLS-R model (CTC objective) on a
   ~15-hour, 160-speaker, speaker-disjoint OpenSLR-54 training subset to close
   that gap.
3. **Generalization re-evaluation.** Evaluate original vs. fine-tuned
   checkpoints on both test sets — including a catastrophic-forgetting check
   on the original corpus — plus error analysis and a speaker-leakage ablation.

### Datasets

- **OpenSLR-54** — crowdsourced multi-speaker Nepali ASR corpus (Kjartansson
  et al., SLTU 2018), 157,905 utterances, CC BY-SA 4.0.
  Link: https://www.openslr.org/54/
  (Working subset already prepared: 15,171 utterances / 160 speakers /
  15.03 hours, speaker-disjoint 80/10/10 split, zero speaker overlap.)
- **OpenSLR-43** — the model's own training corpus (single-speaker female
  Nepali TTS-style data), used only for in-domain reproduction of the claim.
  Link: https://www.openslr.org/43/

The two corpora are kept strictly separate in all results.

### Method and infrastructure

Fine-tuning uses the Hugging Face `transformers` CTC training stack on a Colab
T4 GPU (FP16, batch 2, gradient accumulation, early stopping on validation
WER). All experiments — audit runs, training, and re-evaluation — are tracked
with **MLflow** (parameters, WER/CER metrics, artifacts), giving a reproducible
evidence trail; environment versions are pinned and device screenshots
captured throughout.

### Work plan (10 phases)

| Phase | Deliverable |
|---|---|
| 1 | This proposal |
| 2 | Environment + MLflow reproducibility setup |
| 3 | Data preparation (complete: speaker-disjoint 15hr subset) |
| 4 | Baseline audit: in-domain vs. out-of-domain zero-shot WER |
| 5 | Algorithm selection justification |
| 6 | XLS-R fine-tuning on speaker-diverse data (Colab T4) |
| 7 | Generalization re-evaluation (4-cell before/after × in/out-of-domain) |
| 8 | Efficiency benchmark (latency, size, int8 quantization) |
| 9 | Error analysis, leakage ablation, discussion |
| 10 | Report writing, evidence assembly, submission |

### Achievability

The audit half is already demonstrated end-to-end at small scale (the 4.91% /
~65% preliminary numbers above), and the data pipeline (download,
preprocessing, speaker-disjoint splitting) is built and validated. The main
remaining cost is the Phase 6 GPU fine-tuning run, which fits a free Colab T4
budget with checkpointed, resumable training. Every phase produces a concrete,
checkable artifact (a metric, a checkpoint, an MLflow run), so progress is
verifiable throughout rather than only at submission.


\newpage

# Appendix B — Full Code Listing

All source code for this project, in full. External-source adaptations are cited in each module's docstring where applicable (the Hugging Face wav2vec2 fine-tuning blog for the CTC training loop and data collator).

## `src/__init__.py`

```python
"""Reusable NSTT modules (preprocessing, metrics) — populated in later tasks."""

DEFAULT_SEED = 42
```

## `src/manifests.py`

```python
"""Manifest read/write helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

SplitName = Literal["train", "val", "test"]

MANIFEST_FIELDS = [
    "utterance_id",
    "audio_path",
    "transcript",
    "speaker_id",
    "gender",
    "duration_s",
    "split",
]


def write_jsonl_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in MANIFEST_FIELDS})


def read_jsonl_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_split_manifests(splits: dict[SplitName, list[dict]], manifest_dir: Path) -> dict[str, Path]:
    """Write train/val/test jsonl manifests and return paths."""
    paths: dict[str, Path] = {}
    for split_name, rows in splits.items():
        out = manifest_dir / f"{split_name}.jsonl"
        write_jsonl_manifest(rows, out)
        paths[split_name] = out
    all_rows = splits["train"] + splits["val"] + splits["test"]
    all_path = manifest_dir / "all.jsonl"
    write_jsonl_manifest(all_rows, all_path)
    paths["all"] = all_path
    return paths
```

## `src/pipeline.py`

```python
"""End-to-end SLR54 data preparation pipeline."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from src.manifests import write_split_manifests
from src.preprocessing import (
    classify_gender_from_pitch,
    estimate_mean_f0_hz,
    process_utterance,
)
from src.slr54 import (
    attach_audio_paths,
    download_tsv,
    download_zips,
    extract_zips,
    index_audio_files,
    parse_utt_spk_text_tsv,
)
from src.splits import count_speaker_overlap, speaker_disjoint_split

DEFAULT_SEED = 42
DEFAULT_TARGET_HOURS = 15.0
GENDER_SAMPLES_PER_SPEAKER = 3


def _estimate_speaker_gender(
    speaker_records: list[dict], project_root: Path, *, num_samples: int = GENDER_SAMPLES_PER_SPEAKER
) -> str:
    """Acoustic pseudo-label for one speaker: mean-F0 over a few of their processed clips.

    OpenSLR54 ships no gender metadata; this is a heuristic proxy, not ground truth.
    """
    sample = speaker_records[:num_samples]
    f0_values: list[float] = []
    for record in sample:
        audio_path = project_root / record["audio_path"]
        waveform, sr = sf.read(audio_path)
        f0 = estimate_mean_f0_hz(waveform, sr)
        if f0 is not None:
            f0_values.append(f0)
    if not f0_values:
        return "unknown"
    return classify_gender_from_pitch(sum(f0_values) / len(f0_values))


def run_slr54_pipeline(
    project_root: Path,
    *,
    corpus_dir: Path | None = None,
    zip_names: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    max_utterances: int | None = None,
    target_hours: float | None = DEFAULT_TARGET_HOURS,
    skip_download: bool = True,
) -> dict:
    """Preprocess, split, and write manifests for SLR54.

    corpus_dir points at an already-downloaded/extracted corpus
    (tsv + data/<xx>/*.flac directly underneath it, no download step).
    Set skip_download=False to fall back to the original download-then-extract
    flow into data/raw/slr54 instead.

    target_hours caps the processed subset by total audio duration, selecting
    whole speakers (shuffled by seed) until the cap is reached, so the
    speaker-disjoint split downstream is unaffected. Set to None to process
    every utterance with attached audio.
    """
    processed_dir = project_root / "data" / "processed"
    manifest_dir = project_root / "data" / "manifests"

    if skip_download:
        extract_dir = corpus_dir or (project_root / "data" / "openslr54_ne")
        tsv_path = extract_dir / "utt_spk_text.tsv"
    else:
        raw_dir = project_root / "data" / "raw" / "slr54"
        extract_dir = raw_dir / "extracted"
        tsv_path = download_tsv(raw_dir)
        zip_paths = download_zips(raw_dir, zip_names=zip_names)
        extract_zips(raw_dir, zip_paths, extract_dir)

    utterances = parse_utt_spk_text_tsv(tsv_path)
    tsv_count = len(utterances)
    audio_index = index_audio_files(extract_dir)
    utterances = attach_audio_paths(utterances, audio_index)

    if max_utterances is not None:
        utterances = utterances[:max_utterances]

    by_speaker: dict[str, list] = defaultdict(list)
    for utt in utterances:
        by_speaker[utt.speaker_id].append(utt)

    speaker_order = list(by_speaker.keys())
    random.Random(seed).shuffle(speaker_order)

    target_seconds = target_hours * 3600 if target_hours is not None else None
    processed_records: list[dict] = []
    speaker_records: dict[str, list[dict]] = defaultdict(list)
    skipped_no_audio = 0
    skipped_duration = 0
    total_duration_s = 0.0

    for speaker_id in speaker_order:
        if target_seconds is not None and total_duration_s >= target_seconds:
            break
        for utt in by_speaker[speaker_id]:
            if utt.audio_path is None:
                skipped_no_audio += 1
                continue
            row = process_utterance(
                utterance_id=utt.utterance_id,
                speaker_id=utt.speaker_id,
                transcript=utt.transcript,
                input_audio=utt.audio_path,
                processed_audio_dir=processed_dir,
            )
            if row is None:
                skipped_duration += 1
                continue
            processed_records.append(row)
            speaker_records[speaker_id].append(row)
            total_duration_s += row["duration_s"]

    for speaker_id, records in speaker_records.items():
        gender = _estimate_speaker_gender(records, project_root)
        for record in records:
            record["gender"] = gender

    splits = speaker_disjoint_split(processed_records, seed=seed)
    manifest_paths = write_split_manifests(splits, manifest_dir)

    overlap = count_speaker_overlap(splits)
    return {
        "manifest_paths": manifest_paths,
        "splits": splits,
        "stats": {
            "utterances_in_tsv": tsv_count,
            "utterances_with_audio": len(utterances),
            "processed_after_filter": len(processed_records),
            "skipped_no_audio": skipped_no_audio,
            "skipped_duration": skipped_duration,
            "total_duration_hours": round(total_duration_s / 3600, 2),
            "num_speakers": len(speaker_records),
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "speaker_overlap": overlap,
            "seed": seed,
        },
    }
```

## `src/preprocessing.py`

```python
"""Audio and transcript preprocessing for NSTT."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0

# Mean-F0 threshold separating male/female speech in the standard adult voice
# pitch ranges (~85-180Hz male, ~165-255Hz female); midpoint is the common
# heuristic cutoff. OpenSLR54 ships no gender metadata, so this acoustic proxy
# is used as a pseudo-label, not ground truth -- must be reported as such.
GENDER_PITCH_THRESHOLD_HZ = 165.0


def normalize_transcript_nfc(text: str) -> str:
    """NFC-normalize Devanagari transcripts to avoid encoding mismatches."""
    return unicodedata.normalize("NFC", text.strip())


def is_nfc_normalized(text: str) -> bool:
    return text == unicodedata.normalize("NFC", text)


def duration_in_bounds(duration_s: float) -> bool:
    return MIN_DURATION_S < duration_s < MAX_DURATION_S


def load_resample_mono(audio_path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Load audio, resample to target_sr, collapse to mono. Returns (waveform, duration_s)."""
    waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration_s = len(waveform) / target_sr
    return waveform, duration_s


def process_audio_file(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> float:
    """Resample to 16 kHz mono WAV and return duration in seconds."""
    waveform, duration_s = load_resample_mono(input_path, target_sr=target_sr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, target_sr)
    return duration_s


def estimate_mean_f0_hz(waveform: np.ndarray, sr: int) -> float | None:
    """Estimate mean voiced fundamental frequency (F0) in Hz via librosa pYIN.

    Returns None if no voiced frames are detected (e.g. silence/noise).
    """
    f0, voiced_flag, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    voiced_f0 = f0[voiced_flag]
    if voiced_f0.size == 0:
        return None
    return float(np.nanmean(voiced_f0))


def classify_gender_from_pitch(mean_f0_hz: float | None) -> str:
    """Acoustic pseudo-label: 'male'/'female' by mean-F0 threshold, 'unknown' if undetected."""
    if mean_f0_hz is None or np.isnan(mean_f0_hz):
        return "unknown"
    return "female" if mean_f0_hz >= GENDER_PITCH_THRESHOLD_HZ else "male"


def process_utterance(
    utterance_id: str,
    speaker_id: str,
    transcript: str,
    input_audio: Path,
    processed_audio_dir: Path,
) -> dict | None:
    """Process one utterance; return manifest row dict or None if filtered out."""
    normalized = normalize_transcript_nfc(transcript)
    output_path = processed_audio_dir / f"{utterance_id}.wav"
    duration_s = process_audio_file(input_audio, output_path)

    if not duration_in_bounds(duration_s):
        if output_path.exists():
            output_path.unlink()
        return None

    return {
        "utterance_id": utterance_id,
        "audio_path": str(Path("data/processed") / f"{utterance_id}.wav"),
        "transcript": normalized,
        "speaker_id": speaker_id,
        "duration_s": round(duration_s, 4),
    }
```

## `src/slr54.py`

```python
"""OpenSLR SLR54 download and ingest utilities.

Dataset: https://www.openslr.org/54/
Citation: Kjartansson et al., SLTU 2018 (see start.md / SLR54 LICENSE).
"""

from __future__ import annotations

import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

SLR54_BASE_URL = "https://www.openslr.org/resources/54"
SLR54_TSV_NAME = "utt_spk_text.tsv"
SLR54_ZIP_NAMES = [f"asr_nepali_{i:x}" for i in range(16)]  # 0-9, a-f
SLR54_ZIP_FILES = [f"{name}.zip" for name in SLR54_ZIP_NAMES]
SLR54_AUDIO_SUBDIR = Path("asr_nepali/data")


@dataclass(frozen=True)
class Slr54Utterance:
    utterance_id: str
    speaker_id: str
    transcript: str
    audio_path: Path | None = None


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def download_tsv(raw_dir: Path) -> Path:
    """Download the SLR54 utterance/speaker/transcript index."""
    dest = raw_dir / SLR54_TSV_NAME
    _download_file(f"{SLR54_BASE_URL}/{SLR54_TSV_NAME}", dest)
    return dest


def download_zips(raw_dir: Path, zip_names: list[str] | None = None) -> list[Path]:
    """Download SLR54 audio archives (default: all 16 zips)."""
    names = zip_names or SLR54_ZIP_FILES
    paths: list[Path] = []
    for name in names:
        dest = raw_dir / name
        _download_file(f"{SLR54_BASE_URL}/{name}", dest)
        paths.append(dest)
    return paths


def extract_zips(raw_dir: Path, zip_paths: list[Path], extract_dir: Path) -> None:
    """Extract downloaded zip archives into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    for zip_path in zip_paths:
        marker = extract_dir / f".extracted_{zip_path.name}"
        if marker.exists():
            continue
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        marker.touch()


def parse_utt_spk_text_tsv(tsv_path: Path) -> list[Slr54Utterance]:
    """Parse utt_spk_text.tsv: FileID, UserID (speaker), transcription."""
    utterances: list[Slr54Utterance] = []
    with tsv_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = re.split(r"\t+", line)
            if len(fields) < 3:
                continue
            file_id, speaker_id, transcript = fields[0], fields[1], fields[2]
            utterances.append(
                Slr54Utterance(
                    utterance_id=file_id,
                    speaker_id=speaker_id,
                    transcript=transcript,
                )
            )
    return utterances


def index_audio_files(extract_dir: Path) -> dict[str, Path]:
    """Map utterance_id (flac stem) to absolute audio path."""
    audio_root = extract_dir / SLR54_AUDIO_SUBDIR
    if not audio_root.exists():
        # Some archives may flatten paths differently; fall back to recursive search.
        flac_files = list(extract_dir.rglob("*.flac"))
    else:
        flac_files = list(audio_root.rglob("*.flac"))

    return {p.stem: p for p in flac_files}


def attach_audio_paths(
    utterances: list[Slr54Utterance], audio_index: dict[str, Path]
) -> list[Slr54Utterance]:
    """Return utterances that have a matching audio file on disk."""
    attached: list[Slr54Utterance] = []
    for utt in utterances:
        audio_path = audio_index.get(utt.utterance_id)
        if audio_path is None:
            continue
        attached.append(
            Slr54Utterance(
                utterance_id=utt.utterance_id,
                speaker_id=utt.speaker_id,
                transcript=utt.transcript,
                audio_path=audio_path,
            )
        )
    return attached
```

## `src/splits.py`

```python
"""Speaker-disjoint dataset splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Literal

SplitName = Literal["train", "val", "test"]

DEFAULT_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def speaker_disjoint_split(
    records: list[dict],
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[SplitName, list[dict]]:
    """Assign records to train/val/test with no speaker appearing in multiple splits."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_speaker[record["speaker_id"]].append(record)

    speakers = list(by_speaker.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_speakers = set(speakers[:n_train])
    val_speakers = set(speakers[n_train : n_train + n_val])
    test_speakers = set(speakers[n_train + n_val :])

    assert not (train_speakers & val_speakers)
    assert not (train_speakers & test_speakers)
    assert not (val_speakers & test_speakers)

    splits: dict[SplitName, list[dict]] = {"train": [], "val": [], "test": []}
    for speaker_id, speaker_records in by_speaker.items():
        if speaker_id in train_speakers:
            split: SplitName = "train"
        elif speaker_id in val_speakers:
            split = "val"
        else:
            split = "test"
        for record in speaker_records:
            row = dict(record)
            row["split"] = split
            splits[split].append(row)

    return splits


def utterance_random_split(
    records: list[dict],
    *,
    seed: int = DEFAULT_SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[SplitName, list[dict]]:
    """"Leaky" split: shuffle utterances directly, ignoring speaker identity.

    Same size/ratios as speaker_disjoint_split, but a speaker's utterances can
    land in multiple splits. Used only as a Phase 9 ablation to measure how
    much speaker leakage inflates apparent performance on this dataset.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits: dict[SplitName, list[dict]] = {"train": [], "val": [], "test": []}
    for i, record in enumerate(shuffled):
        row = dict(record)
        if i < n_train:
            row["split"] = "train"
            splits["train"].append(row)
        elif i < n_train + n_val:
            row["split"] = "val"
            splits["val"].append(row)
        else:
            row["split"] = "test"
            splits["test"].append(row)
    return splits


def count_speaker_overlap(splits: dict[SplitName, list[dict]]) -> int:
    """Return number of speakers appearing in more than one split (expect 0)."""
    speaker_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for row in rows:
            speaker_to_splits[row["speaker_id"]].add(split_name)
    return sum(1 for splits_seen in speaker_to_splits.values() if len(splits_seen) > 1)
```

## `src/wer_metrics.py`

```python
"""Minimal, model-agnostic WER/CER computation (jiwer-based)."""
from __future__ import annotations

import jiwer


def compute_wer_cer(references: list[str], hypotheses: list[str]) -> tuple[float, float]:
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return wer, cer
```

## `src/xlsr_baseline.py`

```python
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


def write_openslr43_manifest(rows: list[dict], path: Path | str) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "hf_dataset", "hf_index", "text", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)
```

## `src/xlsr_training.py`

```python
"""XLS-R (wav2vec2) CTC fine-tuning on the speaker-disjoint OpenSLR-54 subset.

Phase 6: fine-tune the audited checkpoint (gagan3012/wav2vec2-xlsr-nepali)
on speaker-diverse data to close its demonstrated generalization gap.
Keeps the checkpoint's own tokenizer/vocab (we repair the same model, not
train a new one). Experiment tracking: MLflow (report_to=["mlflow"]).

Collator adapted from the Hugging Face wav2vec2 fine-tuning blog
(https://huggingface.co/blog/fine-tune-wav2vec2-english) -- cite in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    set_seed,
)


class MPSCacheClearCallback(TrainerCallback):
    """Long MPS training runs accumulate allocator fragmentation until
    torch.autograd's backward pass OOMs (observed: crashed ~1580 steps in,
    17GB allocated, right after an eval pass). Periodic empty_cache() keeps
    the allocator from growing unbounded. No-op on CUDA/CPU."""

    def _clear(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            self._clear()

    def on_evaluate(self, args, state, control, **kwargs):
        self._clear()

from src.manifests import read_jsonl_manifest
from src.wer_metrics import compute_wer_cer
from src.xlsr_baseline import XLSR_MODEL_ID, clean_transcript

DEFAULT_SEED = 42
TARGET_SAMPLE_RATE = 16_000
# The checkpoint is already Nepali fine-tuned; a low LR adapts it to diverse
# speakers without erasing what it knows (Phase 7 checks forgetting anyway).
DEFAULT_LEARNING_RATE = 3e-5
MAX_EPOCHS = 5


def load_model_and_processor(
    model_id: str = XLSR_MODEL_ID,
) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    # torch's scaled_dot_product_attention raises NotImplementedError on Apple
    # MPS when dropout is active (i.e. in training mode); fall back to eager
    # attention off-CUDA so local smoke tests run. CUDA (Colab T4) keeps SDPA.
    attn = "sdpa" if torch.cuda.is_available() else "eager"
    model = Wav2Vec2ForCTC.from_pretrained(model_id, attn_implementation=attn)
    # Standard wav2vec2 fine-tuning practice: the convolutional feature
    # encoder was trained on far more audio than we have -- freeze it.
    model.freeze_feature_encoder()
    return model, processor


def manifest_to_dataset(rows: list[dict], project_root: Path) -> Dataset:
    def _generator() -> Any:
        for row in rows:
            yield {
                "audio": str((project_root / row["audio_path"]).resolve()),
                "text": clean_transcript(row["transcript"]),
                "utterance_id": row["utterance_id"],
            }

    dataset = Dataset.from_generator(_generator)
    return dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))


def load_datasets(
    project_root: Path,
    *,
    manifest_dir: Path | None = None,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> tuple[Dataset, Dataset]:
    manifest_dir = manifest_dir or (project_root / "data" / "manifests")
    train_rows = read_jsonl_manifest(manifest_dir / "train.jsonl")
    val_rows = read_jsonl_manifest(manifest_dir / "val.jsonl")
    if max_train is not None:
        train_rows = train_rows[:max_train]
    if max_eval is not None:
        val_rows = val_rows[:max_eval]
    return (
        manifest_to_dataset(train_rows, project_root),
        manifest_to_dataset(val_rows, project_root),
    )


def build_prepare_fn(processor: Wav2Vec2Processor):
    def prepare(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return prepare


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: Wav2Vec2Processor):
    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(
            pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id
        )
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer, cer = compute_wer_cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


def build_training_arguments(
    output_dir: Path,
    *,
    smoke_test: bool = False,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> TrainingArguments:
    fp16 = torch.cuda.is_available()
    if smoke_test:
        return TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=0,
            max_steps=6,
            eval_strategy="steps",
            eval_steps=3,
            logging_steps=1,
            save_steps=3,
            save_total_limit=2,
            fp16=fp16,
            report_to=["mlflow"],
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=False,
        )

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=500,
        num_train_epochs=MAX_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=3,
        fp16=fp16,
        report_to=["mlflow"],
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )


def create_trainer(
    project_root: Path,
    output_dir: Path,
    *,
    smoke_test: bool = False,
    seed: int = DEFAULT_SEED,
    manifest_dir: Path | None = None,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> tuple[Trainer, Wav2Vec2Processor]:
    set_seed(seed)
    model, processor = load_model_and_processor()

    if smoke_test and max_train is None:
        max_train, max_eval = 32, 8
    train_ds, eval_ds = load_datasets(
        project_root, manifest_dir=manifest_dir, max_train=max_train, max_eval=max_eval
    )

    prepare = build_prepare_fn(processor)
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names)

    args = build_training_arguments(output_dir, smoke_test=smoke_test)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )
    trainer.add_callback(MPSCacheClearCallback())
    if not smoke_test:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    return trainer, processor


def train_and_save(
    project_root: Path,
    output_dir: Path,
    *,
    smoke_test: bool = False,
    resume_from_checkpoint: str | bool | None = None,
    seed: int = DEFAULT_SEED,
    manifest_dir: Path | None = None,
    max_train: int | None = None,
    max_eval: int | None = None,
) -> dict:
    trainer, processor = create_trainer(
        project_root,
        output_dir,
        smoke_test=smoke_test,
        seed=seed,
        manifest_dir=manifest_dir,
        max_train=max_train,
        max_eval=max_eval,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    return {
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_metrics,
        "checkpoint_dir": str(output_dir),
        "global_step": trainer.state.global_step,
    }
```

## `scripts/make_leaky_manifests.py`

```python
"""Phase 9 ablation: build a leaky (utterance-random, non-speaker-disjoint)
split from the same processed records as the real manifests, for a
matched-training-budget comparison against the speaker-disjoint split."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.manifests import read_jsonl_manifest, write_split_manifests
from src.splits import count_speaker_overlap, utterance_random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    all_rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "all.jsonl")
    splits = utterance_random_split(all_rows, seed=42)
    manifest_dir = PROJECT_ROOT / "data" / "manifests_leaky"
    write_split_manifests(splits, manifest_dir)
    overlap = count_speaker_overlap(splits)
    print(f"Leaky split written to {manifest_dir}")
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"speaker_overlap between splits: {overlap} (expected > 0 -- this is the leaky ablation)")


if __name__ == "__main__":
    main()
```

## `scripts/run_audit.py`

```python
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
```

## `scripts/run_efficiency_benchmark.py`

```python
"""Phase 8 — Efficiency / Deployment Benchmark.

Benchmarks the fine-tuned checkpoint (models/xlsr-ft): model size, CPU
inference latency, and the FP32 vs. dynamic int8 quantization trade-off
(latency speedup vs. WER/CER cost). Logged to MLflow (experiment
"phase8-efficiency").
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Apple Silicon (and most non-x86 ARM) only ships the qnnpack quantized
# backend, not fbgemm (torch's x86 default) -- without this, quantize_dynamic
# raises "RuntimeError: Didn't find engine for operation ... NoQEngine".
if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"

from src.manifests import read_jsonl_manifest
from src.wer_metrics import compute_wer_cer
from src.xlsr_baseline import resample_if_needed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "xlsr-ft"
N_SAMPLES = 50
N_LATENCY_RUNS = 20


def model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 * 1024)


def transcribe(model, processor, speech, device: str) -> str:
    inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def benchmark_latency(model, processor, samples: list, device: str) -> dict:
    # Warm-up (first call pays one-time graph/cache setup cost)
    transcribe(model, processor, samples[0], device)
    times = []
    for speech in samples[:N_LATENCY_RUNS]:
        start = time.perf_counter()
        transcribe(model, processor, speech, device)
        times.append(time.perf_counter() - start)
    return {
        "mean_latency_s": sum(times) / len(times),
        "min_latency_s": min(times),
        "max_latency_s": max(times),
        "num_runs": len(times),
    }


def evaluate_wer(model, processor, rows: list, device: str) -> dict:
    refs, hyps = [], []
    for row in rows:
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        speech = resample_if_needed(audio, sr)
        hyps.append(transcribe(model, processor, speech, device))
        refs.append(row["transcript"])
    wer, cer = compute_wer_cer(refs, hyps)
    return {"wer": wer, "cer": cer, "num_utterances": len(rows)}


def main() -> None:
    device = "cpu"  # deployment-realistic target; dynamic quantization is CPU-only anyway
    processor = Wav2Vec2Processor.from_pretrained(CHECKPOINT_DIR)

    rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    sample_rows = shuffled[:N_SAMPLES]
    samples = []
    for row in sample_rows:
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        samples.append(resample_if_needed(audio, sr))

    print("=== FP32 ===")
    model_fp32 = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_DIR)
    model_fp32.to(device).eval()
    fp32_size = model_size_mb(model_fp32)
    fp32_latency = benchmark_latency(model_fp32, processor, samples, device)
    fp32_wer = evaluate_wer(model_fp32, processor, sample_rows, device)
    print(f"size={fp32_size:.1f}MB latency={fp32_latency['mean_latency_s']*1000:.1f}ms wer={fp32_wer['wer']:.4f}")

    print("=== Dynamic INT8 (torch.quantization.quantize_dynamic on Linear layers) ===")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    int8_size = model_size_mb(model_int8)
    int8_latency = benchmark_latency(model_int8, processor, samples, device)
    int8_wer = evaluate_wer(model_int8, processor, sample_rows, device)
    print(f"size={int8_size:.1f}MB latency={int8_latency['mean_latency_s']*1000:.1f}ms wer={int8_wer['wer']:.4f}")

    results = {
        "checkpoint": str(CHECKPOINT_DIR),
        "n_samples": N_SAMPLES,
        "n_latency_runs": N_LATENCY_RUNS,
        "device": "cpu",
        "fp32": {"model_size_mb": fp32_size, **fp32_latency, **fp32_wer},
        "int8_dynamic": {"model_size_mb": int8_size, **int8_latency, **int8_wer},
        "speedup_x": fp32_latency["mean_latency_s"] / int8_latency["mean_latency_s"],
        "size_reduction_pct": (1 - int8_size / fp32_size) * 100,
        "wer_delta": int8_wer["wer"] - fp32_wer["wer"],
    }

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("phase8-efficiency")
    with mlflow.start_run(run_name="fp32_vs_int8"):
        mlflow.log_params({"checkpoint": str(CHECKPOINT_DIR), "n_samples": N_SAMPLES})
        mlflow.log_metrics(
            {
                "fp32_size_mb": fp32_size,
                "fp32_latency_ms": fp32_latency["mean_latency_s"] * 1000,
                "fp32_wer": fp32_wer["wer"],
                "int8_size_mb": int8_size,
                "int8_latency_ms": int8_latency["mean_latency_s"] * 1000,
                "int8_wer": int8_wer["wer"],
                "speedup_x": results["speedup_x"],
            }
        )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "phase8_efficiency_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSpeedup: {results['speedup_x']:.2f}x, size reduction: {results['size_reduction_pct']:.1f}%, WER delta: {results['wer_delta']*100:+.2f}pp")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
```

## `scripts/run_error_analysis.py`

```python
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
```

## `scripts/run_generalization_eval.py`

```python
"""Phase 7 — Generalization Re-Evaluation.

Evaluates a checkpoint (original audited model, OR the Phase 6 fine-tuned
one) on BOTH test sets, producing the four-cell comparison table:
    {original, fine-tuned} x {in-domain OpenSLR-43, out-of-domain OpenSLR-54}

Run once per checkpoint, then diff the two resulting JSON files:
    python scripts/run_generalization_eval.py --checkpoint gagan3012/wav2vec2-xlsr-nepali --tag original
    python scripts/run_generalization_eval.py --checkpoint models/xlsr-ft --tag finetuned

All runs logged to MLflow (experiment "phase7-generalization").
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.manifests import read_jsonl_manifest
from src.wer_metrics import compute_wer_cer
from src.xlsr_baseline import (
    build_openslr43_manifest_rows,
    load_openslr43_test_dataset,
    resample_if_needed,
    transcribe_ctc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def eval_slr43(model, processor, device: str, n_samples: int) -> dict:
    dataset = load_openslr43_test_dataset(max_samples=n_samples)
    rows = build_openslr43_manifest_rows(dataset)
    refs, hyps = [], []
    for i, row in enumerate(dataset):
        audio = row["audio"]
        speech = resample_if_needed(audio["array"], audio["sampling_rate"])
        hyps.append(transcribe_ctc(model, processor, speech, device))
        refs.append(rows[i]["text"])
        if (i + 1) % 25 == 0:
            print(f"  [SLR43] {i + 1}/{len(dataset)}")
    wer, cer = compute_wer_cer(refs, hyps)
    return {"dataset": "OpenSLR-43 (in-domain)", "num_utterances": len(dataset), "wer": wer, "cer": cer}


def eval_slr54(model, processor, device: str, n_samples: int, seed: int) -> dict:
    all_rows = read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
    shuffled = list(all_rows)
    random.Random(seed).shuffle(shuffled)
    rows = shuffled[:n_samples] if n_samples else shuffled

    refs, hyps = [], []
    for i, row in enumerate(rows):
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        speech = resample_if_needed(audio, sr)
        hyps.append(transcribe_ctc(model, processor, speech, device))
        refs.append(row["transcript"])
        if (i + 1) % 25 == 0:
            print(f"  [SLR54] {i + 1}/{len(rows)}")
    wer, cer = compute_wer_cer(refs, hyps)
    return {"dataset": "OpenSLR-54 speaker-disjoint test (out-of-domain)", "num_utterances": len(rows), "wer": wer, "cer": cer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="HF model id or local path")
    parser.add_argument("--tag", required=True, help="'original' or 'finetuned' (used in filenames/MLflow run names)")
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint} ({args.tag})...")
    processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
    model = Wav2Vec2ForCTC.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment("phase7-generalization")

    slr43 = eval_slr43(model, processor, device, args.n_samples)
    slr54 = eval_slr54(model, processor, device, args.n_samples, args.seed)
    print(f"[{args.tag}] SLR43 in-domain:     WER={slr43['wer']:.4f} CER={slr43['cer']:.4f}")
    print(f"[{args.tag}] SLR54 out-of-domain: WER={slr54['wer']:.4f} CER={slr54['cer']:.4f}")

    with mlflow.start_run(run_name=f"{args.tag}_slr43"):
        mlflow.log_params({"checkpoint": args.checkpoint, "tag": args.tag, "dataset": "slr43"})
        mlflow.log_metrics({"wer": slr43["wer"], "cer": slr43["cer"]})
    with mlflow.start_run(run_name=f"{args.tag}_slr54"):
        mlflow.log_params({"checkpoint": args.checkpoint, "tag": args.tag, "dataset": "slr54"})
        mlflow.log_metrics({"wer": slr54["wer"], "cer": slr54["cer"]})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"phase7_{args.tag}_results.json"
    out.write_text(
        json.dumps({"checkpoint": args.checkpoint, "tag": args.tag, "in_domain_openslr43": slr43, "out_of_domain_openslr54": slr54}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {out}")

    other_tag = "finetuned" if args.tag == "original" else "original"
    other_path = REPORTS_DIR / f"phase7_{other_tag}_results.json"
    if other_path.exists():
        other = json.loads(other_path.read_text())
        print("\n=== Four-cell comparison ===")
        print(f"{'':12} {'in-domain (SLR43)':>20} {'out-of-domain (SLR54)':>24}")
        a, b = (out, other_path) if args.tag == "original" else (other_path, out)
        a_data, b_data = json.loads(a.read_text()), json.loads(b.read_text())
        print(f"{'original':12} {a_data['in_domain_openslr43']['wer']*100:>18.2f}% {a_data['out_of_domain_openslr54']['wer']*100:>22.2f}%")
        print(f"{'finetuned':12} {b_data['in_domain_openslr43']['wer']*100:>18.2f}% {b_data['out_of_domain_openslr54']['wer']*100:>22.2f}%")


if __name__ == "__main__":
    main()
```

## `scripts/run_xlsr_baseline.py`

```python
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
```

## `scripts/run_xlsr_train.py`

```python
"""Phase 6: fine-tune gagan3012/wav2vec2-xlsr-nepali on the speaker-disjoint
OpenSLR-54 train split.

Usage:
    python scripts/run_xlsr_train.py --smoke                 # local sanity check
    python scripts/run_xlsr_train.py                          # full run (Colab T4)
    python scripts/run_xlsr_train.py --resume true            # after a disconnect

MLflow tracking goes to ./mlruns (experiment "phase6-finetune").
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# aten::_ctc_loss is not implemented on Apple MPS; this makes torch fall back
# to CPU for that single op (must be set before torch initializes). No effect
# on CUDA runs. Ref: the NotImplementedError raised by torch.ctc_loss on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import torch

from src.xlsr_training import train_and_save

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None, help="'true' or a checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest-dir", default=None, help="defaults to data/manifests; use data/manifests_leaky for the Phase 9 ablation"
    )
    parser.add_argument("--mlflow-experiment", default="phase6-finetune")
    parser.add_argument("--summary-tag", default=None, help="filename tag for reports/phase6_train_<tag>_<ts>.json")
    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
        or (PROJECT_ROOT / ("models/xlsr-ft-smoke" if args.smoke else "models/xlsr-ft"))
    )
    resume = (
        True if args.resume == "true" else args.resume if args.resume else None
    )

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print("Device:", json.dumps(info, indent=2))
    print("Output:", output_dir)
    print("Mode:", "smoke" if args.smoke else "full")
    if not args.smoke and not torch.cuda.is_available():
        print(
            "WARNING: full training without CUDA will be extremely slow -- "
            "this mode is intended for a Colab T4 GPU."
        )

    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
    mlflow.set_experiment(args.mlflow_experiment)

    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else None
    result = train_and_save(
        PROJECT_ROOT,
        output_dir,
        smoke_test=args.smoke,
        resume_from_checkpoint=resume,
        seed=args.seed,
        manifest_dir=manifest_dir,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {"created_at": stamp, "device": info, "smoke_test": args.smoke, "manifest_dir": str(manifest_dir) if manifest_dir else None, **result}
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = args.summary_tag or ("smoke" if args.smoke else "full")
    out = reports_dir / f"phase6_train_{tag}_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print("train_loss:", result["train_loss"])
    print("eval_metrics:", result["eval_metrics"])
    print("global_step:", result["global_step"])
    print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
```


\newpage

# Appendix C — Evidence and Reproducibility Artifacts

This section indexes the reproducibility evidence for this project. Screenshots referenced below are inserted as images where available; all underlying raw data is additionally available as tracked files in the project repository (`coursework-10phase` branch) for direct inspection, per the reproducibility requirement.

## C.1 Environment

```
=== Environment Info ===
Platform: macOS-26.5.2-arm64-arm-64bit
Python: 3.11.15 | packaged by conda-forge | (main, Jun 11 2026, 03:29:05) [Clang 19.1.7 ]
torch: 2.13.0
CUDA available: False
MPS available: True
transformers: 4.49.0
datasets: 3.2.0
mlflow: 2.20.1
```

## C.2 MLflow experiment tracking

Six MLflow experiments were logged over the course of this project (local `mlruns/` store): `phase4-audit`, `phase6-finetune`, `phase7-generalization`, `phase8-efficiency`, `phase9-leaky-ablation`, plus the default experiment used during early smoke testing. Each run logs its parameters, metrics, and (where applicable) example artifacts. [SCREENSHOT: MLflow experiments list] [SCREENSHOT: phase6-finetune run overview, Duration 7.7h, Status Finished] [SCREENSHOT: eval_wer metric curve across epochs]

## C.3 Key result files

### `reports/phase4_audit_results.json`

```json
{
  "model": "gagan3012/wav2vec2-xlsr-nepali",
  "self_reported_wer": 0.0597,
  "device": {
    "platform": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
    "python": "3.13.13",
    "torch": "2.13.0",
    "cuda_available": false,
    "mps_available": true
  },
  "in_domain_openslr43": {
    "dataset": "OpenSLR-43 (in-domain)",
    "num_utterances": 150,
    "wer": 0.04913494809688582,
    "cer": 0.008724212254952273
  },
  "out_of_domain_openslr54": {
    "dataset": "OpenSLR-54 speaker-disjoint test (out-of-domain)",
    "num_utterances": 150,
    "wer": 0.6229508196721312,
    "cer": 0.17383314957736126
  },
  "corpus_separation_note": "OpenSLR-43 and OpenSLR-54 results are never merged into a single number.",
  "slr54_gender_note": "SLR54 rows carry an F0-pitch gender pseudo-label (165Hz threshold), not verified metadata -- used only for Phase 9 per-group analysis."
}
```

### `reports/phase7_original_results.json`

```json
{
  "checkpoint": "gagan3012/wav2vec2-xlsr-nepali",
  "tag": "original",
  "in_domain_openslr43": {
    "dataset": "OpenSLR-43 (in-domain)",
    "num_utterances": 150,
    "wer": 0.04913494809688582,
    "cer": 0.008724212254952273
  },
  "out_of_domain_openslr54": {
    "dataset": "OpenSLR-54 speaker-disjoint test (out-of-domain)",
    "num_utterances": 150,
    "wer": 0.6229508196721312,
    "cer": 0.17383314957736126
  }
}
```

### `reports/phase7_finetuned_results.json`

```json
{
  "checkpoint": "models/xlsr-ft",
  "tag": "finetuned",
  "in_domain_openslr43": {
    "dataset": "OpenSLR-43 (in-domain)",
    "num_utterances": 150,
    "wer": 0.16401384083044981,
    "cer": 0.029764959458072462
  },
  "out_of_domain_openslr54": {
    "dataset": "OpenSLR-54 speaker-disjoint test (out-of-domain)",
    "num_utterances": 150,
    "wer": 0.38173302107728335,
    "cer": 0.09886071297317163
  }
}
```

### `reports/phase7_leaky_results.json`

```json
{
  "checkpoint": "models/xlsr-ft-leaky",
  "tag": "leaky",
  "in_domain_openslr43": {
    "dataset": "OpenSLR-43 (in-domain)",
    "num_utterances": 150,
    "wer": 0.1695501730103806,
    "cer": 0.03048342399671559
  },
  "out_of_domain_openslr54": {
    "dataset": "OpenSLR-54 speaker-disjoint test (out-of-domain)",
    "num_utterances": 150,
    "wer": 0.3255269320843091,
    "cer": 0.08195516354281514
  }
}
```

### `reports/phase8_efficiency_results.json`

```json
{
  "checkpoint": "/Users/bimochankunwar/Desktop/NSTT-Lite/models/xlsr-ft",
  "n_samples": 50,
  "n_latency_runs": 20,
  "device": "cpu",
  "fp32": {
    "model_size_mb": 1203.5573768615723,
    "mean_latency_s": 0.20469055199064315,
    "min_latency_s": 0.14858783304225653,
    "max_latency_s": 0.3167113340459764,
    "num_runs": 20,
    "wer": 0.3581081081081081,
    "cer": 0.07165775401069518,
    "num_utterances": 50
  },
  "int8_dynamic": {
    "model_size_mb": 48.45556640625,
    "mean_latency_s": 0.33524910402484237,
    "min_latency_s": 0.20782845804933459,
    "max_latency_s": 0.5563514999812469,
    "num_runs": 20,
    "wer": 0.36486486486486486,
    "cer": 0.0748663101604278,
    "num_utterances": 50
  },
  "speedup_x": 0.6105625623848806,
  "size_reduction_pct": 95.97397121750821,
  "wer_delta": 0.006756756756756743
}
```

### `reports/phase9_error_analysis_results.json`

```json
{
  "checkpoint": "/Users/bimochankunwar/Desktop/NSTT-Lite/models/xlsr-ft",
  "num_utterances": 150,
  "overall_wer": 0.38173302107728335,
  "overall_cer": 0.09886071297317163,
  "error_category_counts": {
    "other": 59,
    "oov_rare_vocabulary": 90,
    "phonetic_confusion": 64,
    "noise_degradation": 22
  },
  "error_category_note": "Heuristic, regex/jiwer-alignment-based tagging -- categories can overlap per utterance and are not mutually exclusive.",
  "gender_pseudo_label_breakdown": {
    "male": {
      "wer": 0.37668161434977576,
      "cer": 0.10310734463276836,
      "num_utterances": 77
    },
    "female": {
      "wer": 0.3872549019607843,
      "cer": 0.09425287356321839,
      "num_utterances": 73
    }
  },
  "gender_pseudo_label_note": "gender_pseudo_label is an F0-pitch threshold heuristic (165Hz), NOT verified ground truth -- this breakdown measures WER by acoustic pitch-threshold group, not by true gender.",
  "speaker_wer_spread": {
    "min": [
      "76dab",
      {
        "mean_wer": 0.125,
        "n": 12
      }
    ],
    "max": [
      "70e10",
      {
        "mean_wer": 0.6363636363636364,
        "n": 11
      }
    ],
    "num_speakers": 16
  },
  "worst_10_examples": [
    {
      "utterance_id": "8c693c9a97",
      "speaker_id": "7f9c6",
      "gender_pseudo_label": "female",
      "reference": "मान्छे चटकारे",
      "hypothesis": "मान्छि चट कार्य",
      "wer": 1.5,
      "cer": 0.3076923076923077,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "d443c84109",
      "speaker_id": "70e10",
      "gender_pseudo_label": "female",
      "reference": "क्रान्तिकारी वाममोर्चाको",
      "hypothesis": "तन्थिकारी बाम वर्षको",
      "wer": 1.5,
      "cer": 0.4166666666666667,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "7da0e9a7b7",
      "speaker_id": "99866",
      "gender_pseudo_label": "female",
      "reference": "राज्यमाथि हासिल गर्‍यो",
      "hypothesis": "राज्यमाति हासेल गर‍यो",
      "wer": 1.0,
      "cer": 0.13636363636363635,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "8ccd499b56",
      "speaker_id": "da0cf",
      "gender_pseudo_label": "male",
      "reference": "बनाउनबाट जोगाउँदछ",
      "hypothesis": "बनाउनबाटा जोगआउँदछ",
      "wer": 1.0,
      "cer": 0.11764705882352941,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "0cf3ebaf89",
      "speaker_id": "07179",
      "gender_pseudo_label": "female",
      "reference": "उद्देश्यले ५ अप्रिल",
      "hypothesis": "उद्येश्यले पच अपरेल",
      "wer": 1.0,
      "cer": 0.2631578947368421,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "3cd1567940",
      "speaker_id": "e8f6f",
      "gender_pseudo_label": "male",
      "reference": "ब्याज तिर्नुपर्ने हुन्छ",
      "hypothesis": "व्यहास दिन्नु पर्ने हुन्छ",
      "wer": 1.0,
      "cer": 0.2608695652173913,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "0950a3e9a4",
      "speaker_id": "99866",
      "gender_pseudo_label": "female",
      "reference": "रन्मामैकोट तकसेरा हुकाम",
      "hypothesis": "रणमा मैकोट तक्सेरा हुकाम",
      "wer": 1.0,
      "cer": 0.17391304347826086,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "7b1f434978",
      "speaker_id": "99866",
      "gender_pseudo_label": "female",
      "reference": "आइल्यान्ड तथा बेलायतदेखि",
      "hypothesis": "आइल्यान् तथा बेलाय देखि",
      "wer": 1.0,
      "cer": 0.08333333333333333,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "34760118f4",
      "speaker_id": "9d08c",
      "gender_pseudo_label": "male",
      "reference": "वृक्ष यही हो",
      "hypothesis": "बृच्ष यहियो",
      "wer": 1.0,
      "cer": 0.4166666666666667,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    },
    {
      "utterance_id": "dde677080a",
      "speaker_id": "70e10",
      "gender_pseudo_label": "female",
      "reference": "सूर्यले पृथ्वीको वरिपरि",
      "hypothesis": "सवर्यलले फृथ्ष गरीपुरि",
      "wer": 1.0,
      "cer": 0.43478260869565216,
      "categories": [
        "noise_degradation",
        "oov_rare_vocabulary",
        "phonetic_confusion"
      ]
    }
  ]
}
```
