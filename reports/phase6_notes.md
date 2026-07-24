# Phase 6 — Task 1: ASR Fine-Tuning

**Rubric target:** Algorithm Modifications (5) + Difficulty (15).

## Training configuration (locked in `src/training.py`)

| Setting | Value | Why |
|---|---|---|
| Model | `openai/whisper-small` | Phase 5 selection; matches Phase 4 baseline |
| Optimizer | AdamW (Hugging Face Trainer default) | Standard Whisper fine-tuning recipe |
| Precision | FP16 when CUDA available; FP32 on CPU | T4 Colab speed/memory; CPU smoke stays stable |
| Batch size | 2 | Fits T4 with Whisper-small + generate-for-eval |
| Gradient accumulation | 4 | Effective batch ≈ 8 without OOM |
| Epochs | 5 ceiling | Roadmap limit; early stop may finish sooner |
| Early stopping | patience 2 on validation **WER** | Stop when val WER stops improving; select best checkpoint |
| LR | 1e-4, warmup 500 (full) / 0 (smoke) | Common Whisper fine-tune LR; short smoke needs no warmup |
| Logging | TensorBoard under `{output_dir}/runs` | Curves for loss / eval WER |
| Checkpoints | save each epoch (full) or every 3 steps (smoke); `save_total_limit=3` | Resumable across Colab disconnects |
| Language / task | `nepali` / `transcribe` | Multilingual Whisper conditioning for Nepali ASR |
| Forced decoder IDs | cleared for training | Required for multilingual Whisper fine-tuning |

## Modifications vs vanilla HF Whisper tutorial

1. **Early stopping on WER** (`EarlyStoppingCallback`, patience 2) with
   `metric_for_best_model="wer"`, `greater_is_better=False`,
   `load_best_model_at_end=True` — tutorial often stops on loss or fixed steps;
   WER is the metric we report.
2. **Epoch ceiling (5) instead of a large `max_steps`** for the full run —
   clearer for coursework reporting and pairs cleanly with epoch-level eval/save.
3. **Smoke mode** — 32 train / 8 val utterances, `max_steps=6`, still wires
   TensorBoard + checkpointing so the loop is verified before burning GPU quota.
4. **Resume** — `scripts/run_train.py --resume true` (or a checkpoint path)
   for Colab session limits.

## How to run

```bash
# Local / CI smoke (CPU OK; not for reported WER numbers)
python scripts/run_train.py --smoke

# Full fine-tune (use Colab T4 GPU; multi-hour)
python scripts/run_train.py --output-dir models/whisper-small-ft
# After disconnect:
python scripts/run_train.py --output-dir models/whisper-small-ft --resume true
```

## Status

- [x] Training script + early stopping + TensorBoard wired
- [x] Smoke test executed end-to-end (2026-07-24, MPS, 6 steps)
- [ ] Full Colab T4 run to early-stop or 5-epoch ceiling
- [ ] Best checkpoint selected by validation WER; TensorBoard screenshots

### Smoke-test evidence (pipeline sanity only — not reported WER)

| Field | Value |
|---|---|
| Device | macOS arm64, torch 2.13.0, CUDA=false, MPS=true |
| Steps | 6 (`max_steps`), 32 train / 8 val utterances |
| Train loss | 1.976 |
| Eval WER (tiny subset) | 0.92 (92%) — not comparable to Phase 4's 150-utt baseline |
| Checkpoints | `models/whisper-small-ft-smoke/checkpoint-{3,6}` (+ final save) |
| TensorBoard | `models/whisper-small-ft-smoke/runs/` populated |
| Summary JSON | `reports/phase6_train_smoke_20260724T065232Z.json` |

Final reported WER/CER must come from the **full GPU run**, not the smoke test.
