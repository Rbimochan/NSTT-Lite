"""Phase 3: combine pilot_seed + batch2_seed + batch3_seed into a single
speaker-disjoint train/val/test manifest for fine-tuning continuation.

Only 5 speakers total (bimochan M, binita F, yoyal M, rita F, bijay M) --
too few for a statistically meaningful split, but enough for a first small
continuation fine-tune while more batches (per medical_corpus_README.md)
are collected. Held-out speakers are deliberately one male (bijay -> val)
and one female (rita -> test) so both splits see a female voice at least
once across val+test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifests import read_jsonl_manifest, write_split_manifests  # noqa: E402

MANIFESTS_DIR = ROOT / "data" / "manifests_medical"
OUT_DIR = ROOT / "data" / "manifests_medical_combined"

TRAIN_SPEAKERS = {"pilot_f01", "pilot_m01", "batch2_yoyal"}
VAL_SPEAKERS = {"batch3_bijay"}
TEST_SPEAKERS = {"batch2_rita"}


def main() -> None:
    all_rows = []
    for name in ["pilot_seed", "batch2_seed", "batch3_seed"]:
        path = MANIFESTS_DIR / f"{name}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} missing, skipping")
            continue
        all_rows.extend(read_jsonl_manifest(path))

    splits = {"train": [], "val": [], "test": []}
    unmatched = []
    for row in all_rows:
        speaker = row["speaker_id"]
        row = {**row}
        if speaker in TRAIN_SPEAKERS:
            row["split"] = "train"
            splits["train"].append(row)
        elif speaker in VAL_SPEAKERS:
            row["split"] = "val"
            splits["val"].append(row)
        elif speaker in TEST_SPEAKERS:
            row["split"] = "test"
            splits["test"].append(row)
        else:
            unmatched.append(speaker)

    if unmatched:
        print(f"WARNING: unmatched speakers, excluded: {sorted(set(unmatched))}")

    paths = write_split_manifests(splits, OUT_DIR)

    for split_name, rows in splits.items():
        speakers = sorted({r["speaker_id"] for r in rows})
        duration = sum(r["duration_s"] for r in rows)
        print(f"{split_name}: {len(rows)} clips, speakers={speakers}, {duration:.1f}s")
    print(f"Manifests written to {OUT_DIR}")


if __name__ == "__main__":
    main()
