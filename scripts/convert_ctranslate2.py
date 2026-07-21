"""Step 13: convert the fine-tuned checkpoint to CTranslate2 and benchmark latency."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "finetuned"
CT2_DIR = PROJECT_ROOT / "models" / "finetuned_ct2"
SAMPLE_WAVS = [
    "/tmp/own_voice_wav/Timro naam k ho.wav",
    "/tmp/own_voice_wav/Maile kaam paina.wav",
    "/tmp/own_voice_wav/Voli chutti ho.wav",
]


def main() -> None:
    if not CT2_DIR.exists():
        subprocess.run(
            [
                "ct2-transformers-converter",
                "--model", str(CHECKPOINT_DIR),
                "--output_dir", str(CT2_DIR),
            ],
            check=True,
        )
        print(f"Converted to CTranslate2 at {CT2_DIR}")
    else:
        print(f"CT2 model already exists at {CT2_DIR}")

    model = WhisperModel(str(CT2_DIR), device="cpu", compute_type="int8")

    print("\nBenchmarking inference latency (CPU, int8):")
    for wav in SAMPLE_WAVS:
        if not Path(wav).exists():
            continue
        start = time.time()
        segments, info = model.transcribe(wav, language="ne")
        text = " ".join(seg.text for seg in segments)
        elapsed = time.time() - start
        print(f"  {Path(wav).stem}: {elapsed:.2f}s -> {text.strip()!r}")


if __name__ == "__main__":
    main()
