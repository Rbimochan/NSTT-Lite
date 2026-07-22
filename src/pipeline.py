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

    SLR54 ships no gender metadata; this is a heuristic proxy, not ground truth.
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
