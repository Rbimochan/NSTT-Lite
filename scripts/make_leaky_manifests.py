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
