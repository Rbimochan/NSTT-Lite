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
