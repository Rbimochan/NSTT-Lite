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


def count_speaker_overlap(splits: dict[SplitName, list[dict]]) -> int:
    """Return number of speakers appearing in more than one split (expect 0)."""
    speaker_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for row in rows:
            speaker_to_splits[row["speaker_id"]].add(split_name)
    return sum(1 for splits_seen in speaker_to_splits.values() if len(splits_seen) > 1)
