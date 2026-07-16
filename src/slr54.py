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
