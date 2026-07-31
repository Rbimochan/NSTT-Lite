"""Plan 02: build the manifest for Batch 3 (Bijay male, #1-25 so far --
Ranjita female, #26-50, not yet recorded). Same schema as pilot_seed /
batch2_seed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifests import write_csv_manifest, write_jsonl_manifest  # noqa: E402

SCRIPTS_MD = ROOT / "data" / "recording_scripts_batch3.md"
AUDIO_DIRS = {
    "male": (ROOT / "data" / "medical_pilot_wav_batch3" / "bijay", "bijay"),
    # "female": (ROOT / "data" / "medical_pilot_wav_batch3" / "ranjita", "ranjita"),  # not yet recorded
}
OUT_DIR = ROOT / "data" / "manifests_medical"


def parse_scripts(md_path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in md_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or line.startswith("| #") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        _, filename, script = cells
        if filename.endswith(".wav"):
            rows[filename] = script
    return rows


def main() -> None:
    script_map = parse_scripts(SCRIPTS_MD)
    rows = []

    for gender, (audio_dir, speaker_id) in AUDIO_DIRS.items():
        for wav_path in sorted(audio_dir.glob("*.wav")):
            filename = wav_path.name
            transcript = script_map.get(filename)
            if transcript is None:
                print(f"WARNING: no script for {filename}, skipping")
                continue
            info = sf.info(wav_path)
            duration_s = round(info.frames / info.samplerate, 3)
            rows.append(
                {
                    "utterance_id": f"medbatch3_{filename.replace('.wav', '')}",
                    "audio_path": str(wav_path.relative_to(ROOT)),
                    "transcript": transcript,
                    "speaker_id": f"batch3_{speaker_id}",
                    "gender": gender,
                    "duration_s": duration_s,
                    "split": "seed",
                }
            )

    write_jsonl_manifest(rows, OUT_DIR / "batch3_seed.jsonl")
    write_csv_manifest(rows, OUT_DIR / "batch3_seed.csv")

    total_duration = sum(r["duration_s"] for r in rows)
    print(f"Clips: {len(rows)}")
    print(f"Speakers: {sorted({r['speaker_id'] for r in rows})}")
    print(f"Total duration: {total_duration:.1f}s (~{total_duration/60:.1f} min)")
    print("NOTE: Ranjita (#26-50) not yet recorded -- re-run once her audio lands in "
          "data/medical_pilot_wav_batch3/ranjita/")


if __name__ == "__main__":
    main()
