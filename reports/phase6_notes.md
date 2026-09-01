# Phase 6 — XLS-R Fine-Tuning on Speaker-Diverse Data

**Rubric target:** Algorithm Modifications (5) + Difficulty (15).

## Training configuration (locked in `src/xlsr_training.py`)

| Setting | Value | Why |
|---|---|---|
| Base checkpoint | `gagan3012/wav2vec2-xlsr-nepali` | The model under audit (Phase 4: 4.91% in-domain / 62.30% out-of-domain WER) |
| Objective | CTC, unchanged head + vocab | Repair the same model — Phase 7 comparability |
| Feature encoder | Frozen (`freeze_feature_encoder()`) | Conv encoder pretrained on far more audio than our 15hrs |
| LR | 3e-5, warmup 500 | Low LR for continued fine-tuning of an already-Nepali checkpoint |
| Batch / grad-accum | 2 / 4 (effective 8) | Fits Colab T4 with FP16 |
| Epochs | 5 ceiling, eval/save per epoch | Coursework budget; resumable across disconnects |
| Early stopping | patience 2 on validation WER | Stop on the reported metric; best checkpoint kept (`load_best_model_at_end`) |
| Precision | FP16 on CUDA / FP32 otherwise | T4 speed; local stability |
| Tracking | MLflow, experiment `phase6-finetune` | Project decision: MLflow, not TensorBoard |

## Modifications vs. the vanilla HF wav2vec2 recipe

1. **Continued fine-tuning of an already-fine-tuned checkpoint** (not from
   `facebook/wav2vec2-xls-r-*`): vocab/tokenizer kept from the audited model,
   LR lowered to 3e-5 (vs the recipe's ~1e-4 from-scratch-head setting).
2. **Early stopping on WER** with best-checkpoint selection — the recipe uses
   fixed epochs.
3. **MLflow tracking** via `report_to=["mlflow"]`.
4. **Apple-silicon workarounds** (local smoke only; no effect on CUDA):
   - SDPA attention raises `NotImplementedError` on MPS when dropout is active
     → model loads with `attn_implementation="eager"` when CUDA is unavailable.
   - `aten::_ctc_loss` is not implemented on MPS at all
     → `PYTORCH_ENABLE_MPS_FALLBACK=1` set in `scripts/run_xlsr_train.py`
     before torch loads (CPU fallback for that single op).

## How to run

```bash
# Local smoke (CPU/MPS OK; sanity only, not reported numbers)
python scripts/run_xlsr_train.py --smoke

# Full fine-tune (Colab T4; multi-hour)
python scripts/run_xlsr_train.py --output-dir models/xlsr-ft
# After a disconnect:
python scripts/run_xlsr_train.py --output-dir models/xlsr-ft --resume true
```

Colab notebook: `notebooks/phase6_xlsr_colab.ipynb` (clones this branch,
checks GPU + data, runs, serves the MLflow UI for screenshots).

## Status

- [x] Training module + runner written; MLflow wired
- [x] Smoke test end-to-end (2026-07-26, MPS+CPU-fallback, 6 steps):
      train_loss 2.72, eval WER 0.68 / CER 0.26 on the 8-utterance smoke
      subset — `reports/phase6_train_smoke_20260726T055255Z.json`,
      checkpoints at `models/xlsr-ft-smoke/checkpoint-{3,6}`
- [ ] **Full Colab T4 run** (the remaining step — needs a human Colab session)
- [ ] Best checkpoint by val WER + MLflow UI screenshots

Note the smoke eval WER (68% on 8 utterances after 6 steps) is a pipeline
sanity number only — but it is already consistent with the Phase 4 zero-shot
out-of-domain figure (62.30%), as expected for a barely-trained run starting
from the audited checkpoint.

## Prerequisite for the Colab run

Training reads `data/processed/*.wav` (Phase 3 output, gitignored, ~1.6GB).
Sync it to Drive under `MyDrive/NSTT-Lite/data/processed/`, or rebuild on
Colab from the raw corpus — the notebook's data-check cell covers both paths.
