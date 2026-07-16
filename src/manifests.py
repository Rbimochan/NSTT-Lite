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
