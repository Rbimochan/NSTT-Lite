"""Step 10: gender classification (second task).

Extracts mean-pooled Whisper encoder embeddings from the fine-tuned checkpoint,
trains a logistic-regression head on the F0 pseudo-labels, and evaluates
against the Step 5 majority-class baseline.

CAVEAT: the gender field is an acoustic pseudo-label (F0 threshold, 165Hz),
not verified ground truth -- SLR54 ships no gender metadata. Reported accuracy
reflects agreement with the F0 heuristic, not necessarily true gender.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import WhisperModel, WhisperProcessor

from src.manifests import read_jsonl_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "finetuned"
REPORTS_DIR = PROJECT_ROOT / "reports"
LABELS = ["male", "female"]


def extract_embeddings(rows: list[dict], model, processor, device: str) -> np.ndarray:
    embeddings = []
    for row in rows:
        audio_path = PROJECT_ROOT / row["audio_path"]
        audio, sr = sf.read(audio_path)
        inputs = processor.feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model.encoder(inputs["input_features"]).last_hidden_state
        pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(pooled)
    return np.stack(embeddings)


def main() -> None:
    device = "cpu"
    processor = WhisperProcessor.from_pretrained(CHECKPOINT_DIR)
    model = WhisperModel.from_pretrained(CHECKPOINT_DIR)
    model.to(device)
    model.eval()

    train_rows = [
        r
        for r in read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "train.jsonl")
        if r["gender"] in LABELS
    ][:800]
    test_rows = [
        r
        for r in read_jsonl_manifest(PROJECT_ROOT / "data" / "manifests" / "test.jsonl")
        if r["gender"] in LABELS
    ]

    print(f"Extracting embeddings: {len(train_rows)} train, {len(test_rows)} test...")
    X_train = extract_embeddings(train_rows, model, processor, device)
    y_train = [r["gender"] for r in train_rows]
    X_test = extract_embeddings(test_rows, model, processor, device)
    y_test = [r["gender"] for r in test_rows]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="female")
    rec = recall_score(y_test, y_pred, pos_label="female")
    f1 = f1_score(y_test, y_pred, pos_label="female")
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    baseline = json.loads((REPORTS_DIR / "baseline_gender.json").read_text())

    results = {
        "caveat": (
            "gender is an F0-threshold (165Hz) pseudo-label, not verified ground truth; "
            "this accuracy measures agreement with the heuristic, not true gender"
        ),
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "accuracy": round(acc, 4),
        "precision_female": round(prec, 4),
        "recall_female": round(rec, 4),
        "f1_female": round(f1, 4),
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "majority_class_baseline_accuracy": baseline["majority_class_accuracy"],
        "improvement_over_baseline": round(acc - baseline["majority_class_accuracy"], 4),
    }

    (REPORTS_DIR / "gender_classifier_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
