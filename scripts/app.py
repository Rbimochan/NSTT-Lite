"""Step 13: minimal Streamlit demo -- transcription + gender prediction +
WER/CER summary + inference latency, on the fine-tuned CTranslate2 model."""
from __future__ import annotations

import os

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from faster_whisper import WhisperModel
from sklearn.linear_model import LogisticRegression
from transformers import WhisperModel as HFWhisperEncoder
from transformers import WhisperProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CT2_DIR = PROJECT_ROOT / "models" / "finetuned_ct2"
HF_CHECKPOINT_DIR = Path(os.environ.get("NSTT_CHECKPOINT_DIR", PROJECT_ROOT / "models" / "finetuned"))
REPORTS_DIR = PROJECT_ROOT / "reports"


@st.cache_resource
def load_asr_model():
    return WhisperModel(str(CT2_DIR), device="cpu", compute_type="int8")


@st.cache_resource
def load_gender_stack():
    """Reproduces the Step 10 classifier at demo time (train once, cache)."""
    from src.manifests import read_jsonl_manifest

    processor = WhisperProcessor.from_pretrained(HF_CHECKPOINT_DIR)
    encoder = HFWhisperEncoder.from_pretrained(HF_CHECKPOINT_DIR)
    encoder.eval()

    rows = [
        r
        for r in read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "train.jsonl")
        if r["gender"] in ("male", "female")
    ][:200]  # small cache-friendly subset for the live demo

    X, y = [], []
    for row in rows:
        audio, sr = sf.read(PROJECT_ROOT / row["audio_path"])
        inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder.encoder(inputs["input_features"]).last_hidden_state
        X.append(hidden.mean(dim=1).squeeze(0).numpy())
        y.append(row["gender"])

    clf = LogisticRegression(max_iter=1000).fit(np.stack(X), y)
    return processor, encoder, clf


def main() -> None:
    st.title("NSTT-Lite: Nepali ASR + Gender Classification Demo")
    st.caption(
        "Fine-tuned Whisper-small (300 steps, 15hr SLR54 subset). "
        "Gender label is an F0-threshold pseudo-label, not verified ground truth."
    )

    with st.expander("Baseline vs. fine-tuned metrics (real numbers from this project)"):
        wer_cer = (REPORTS_DIR / "wer_cer_results.md").read_text() if (REPORTS_DIR / "wer_cer_results.md").exists() else "Not yet computed."
        st.markdown(wer_cer)
        if (REPORTS_DIR / "gender_classifier_results.json").exists():
            st.json(json.loads((REPORTS_DIR / "gender_classifier_results.json").read_text()))

    uploaded = st.file_uploader("Upload a Nepali audio clip (wav/m4a/flac)", type=["wav", "m4a", "flac"])
    if uploaded is None:
        st.info("Upload a clip to transcribe.")
        return

    tmp_path = PROJECT_ROOT / "reports" / "_demo_upload.wav"
    audio, sr = sf.read(uploaded)
    sf.write(tmp_path, audio, sr)

    asr_model = load_asr_model()
    start = time.time()
    segments, info = asr_model.transcribe(str(tmp_path), language="ne")
    transcript = " ".join(seg.text for seg in segments).strip()
    asr_latency = time.time() - start

    processor, encoder, clf = load_gender_stack()
    inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder.encoder(inputs["input_features"]).last_hidden_state
    embedding = hidden.mean(dim=1).squeeze(0).numpy().reshape(1, -1)
    gender_pred = clf.predict(embedding)[0]

    st.subheader("Transcription")
    st.write(transcript)
    st.subheader("Gender prediction (pseudo-label classifier)")
    st.write(gender_pred)
    st.subheader("Inference time")
    st.write(f"{asr_latency:.2f}s (CTranslate2 int8, CPU)")


if __name__ == "__main__":
    main()
