"""Whisper fine-tuning helpers for NSTT.

Adapted from the Hugging Face Whisper fine-tuning tutorial:
https://huggingface.co/blog/fine-tune-whisper
(Cite in academic work per start.md §7.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from src.manifests import read_jsonl_manifest

WHISPER_MODEL_ID = "openai/whisper-small"
DEFAULT_SEED = 42
TARGET_SAMPLE_RATE = 16_000
LANGUAGE = "nepali"
TASK = "transcribe"


def load_whisper_model_and_processor(
    model_id: str = WHISPER_MODEL_ID,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    processor = WhisperProcessor.from_pretrained(model_id, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    # Whisper multilingual fine-tuning: do not force English-only decoder tokens during training.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return model, processor


def manifest_to_dataset(rows: list[dict], project_root: Path) -> Dataset:
    """Build a Hugging Face Dataset from manifest rows."""

    def _generator() -> Any:
        for row in rows:
            audio_path = project_root / row["audio_path"]
            yield {
                "audio": str(audio_path.resolve()),
                "sentence": row["transcript"],
                "utterance_id": row["utterance_id"],
            }

    dataset = Dataset.from_generator(_generator)
    return dataset.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))


def load_manifest_datasets(
    project_root: Path,
    manifest_dir: Path | None = None,
) -> tuple[Dataset, Dataset]:
    manifest_dir = manifest_dir or (project_root / "data" / "manifests")
    train_rows = read_jsonl_manifest(manifest_dir / "train.jsonl")
    val_rows = read_jsonl_manifest(manifest_dir / "val.jsonl")
    return (
        manifest_to_dataset(train_rows, project_root),
        manifest_to_dataset(val_rows, project_root),
    )


def build_prepare_fn(processor: WhisperProcessor) -> Callable[[dict], dict]:
    def prepare_dataset(batch: dict) -> dict:
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    return prepare_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad input features and token labels for Whisper Seq2Seq training."""

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def build_compute_metrics(processor: WhisperProcessor) -> Callable:
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    return compute_metrics


def build_training_arguments(
    output_dir: Path,
    *,
    smoke_test: bool = False,
    fp16: bool = True,
) -> Seq2SeqTrainingArguments:
    """Build Seq2SeqTrainingArguments for Colab T4 (or CPU smoke test)."""
    if smoke_test:
        return Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=0,
            max_steps=6,
            eval_strategy="steps",
            eval_steps=3,
            logging_steps=1,
            save_steps=3,
            save_total_limit=2,
            predict_with_generate=True,
            generation_max_length=128,
            fp16=fp16 and torch.cuda.is_available(),
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=False,
        )

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=2000,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=100,
        save_steps=500,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=225,
        fp16=fp16 and torch.cuda.is_available(),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=False,
    )


def create_trainer(
    project_root: Path,
    output_dir: Path,
    *,
    smoke_test: bool = False,
    seed: int = DEFAULT_SEED,
    manifest_dir: Path | None = None,
) -> tuple[Seq2SeqTrainer, WhisperProcessor]:
    set_seed(seed)
    model, processor = load_whisper_model_and_processor()
    train_ds, eval_ds = load_manifest_datasets(project_root, manifest_dir=manifest_dir)

    prepare_fn = build_prepare_fn(processor)
    train_ds = train_ds.map(prepare_fn, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(prepare_fn, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = build_training_arguments(output_dir, smoke_test=smoke_test)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor),
        processing_class=processor.feature_extractor,
    )
    return trainer, processor


def train_and_save(
    project_root: Path,
    output_dir: Path,
    *,
    smoke_test: bool = False,
    resume_from_checkpoint: str | bool | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run training and return metrics + checkpoint path."""
    trainer, processor = create_trainer(
        project_root,
        output_dir,
        smoke_test=smoke_test,
        seed=seed,
    )
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    return {
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_metrics,
        "checkpoint_dir": str(output_dir),
    }
