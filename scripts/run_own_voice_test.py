"""Step 11: transcribe real self-recorded Nepali clips with the fine-tuned checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.evaluation import compute_wer_cer
from src.training import LANGUAGE, TASK

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "finetuned"
WAV_DIR = Path("/tmp/own_voice_wav")
TXT_DIR = PROJECT_ROOT / "data" / "iphone_recordings"


def main() -> None:
    device = "cpu"
    processor = WhisperProcessor.from_pretrained(CHECKPOINT_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(CHECKPOINT_DIR)
    model.to(device)
    model.eval()

    results = []
    for wav_path in sorted(WAV_DIR.glob("*.wav")):
        txt_path = TXT_DIR / f"{wav_path.stem}.txt"
        if not txt_path.exists():
            continue
        reference = txt_path.read_text(encoding="utf-8").strip()

        audio, sr = sf.read(wav_path)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pred_ids = model.generate(**inputs, language=LANGUAGE, task=TASK)
        hypothesis = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()

        wer, cer = compute_wer_cer([reference], [hypothesis])
        results.append(
            {
                "clip": wav_path.stem,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
            }
        )
        print(f"[{wav_path.stem}]")
        print(f"  ref: {reference}")
        print(f"  hyp: {hypothesis}")
        print(f"  wer={wer:.2f} cer={cer:.2f}")

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "own_voice_test.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(results)} results to reports/own_voice_test.json")


if __name__ == "__main__":
    main()
