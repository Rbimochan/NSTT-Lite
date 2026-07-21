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
# heuristic cutoff. SLR54 ships no gender metadata, so this acoustic proxy is
# used as a pseudo-label, not ground truth — must be reported as such.
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
