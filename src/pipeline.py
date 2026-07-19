"""End-to-end SLR54 data preparation pipeline."""

from __future__ import annotations

from pathlib import Path

from src.manifests import write_split_manifests
from src.preprocessing import process_utterance
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


def run_slr54_pipeline(
    project_root: Path,
    *,
    corpus_dir: Path | None = None,
    zip_names: list[str] | None = None,
    seed: int = DEFAULT_SEED,
    max_utterances: int | None = None,
    skip_download: bool = True,
) -> dict:
    """Preprocess, split, and write manifests for SLR54.

    corpus_dir points at an already-downloaded/extracted corpus
    (tsv + data/<xx>/*.flac directly underneath it, no download step).
    Set skip_download=False to fall back to the original download-then-extract
    flow into data/raw/slr54 instead.
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

    processed_records: list[dict] = []
    skipped_no_audio = 0
    skipped_duration = 0

    for utt in utterances:
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
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "speaker_overlap": overlap,
            "seed": seed,
        },
    }
