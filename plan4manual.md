# Nepali ASR (Whisper Fine-tuning) — Plan 4, Manual Mode

For running Plan 4 yourself in Colab, step by step, without Cursor/Claude in the loop.
Uses `notebooks/plan4_colab.ipynb` and the `scripts/run_colab_*.py` files already in
the repo — this file just tells you which cell/command to run at each step and what
"done" looks like. One step at a time; don't start the next until the current
checkpoint is true.

---

**STEP 1: Open the notebook on the right branch**
Go to:
```
https://colab.research.google.com/github/Rbimochan/NSTT-Lite/blob/feature/plan2-plan3-execution/notebooks/plan4_colab.ipynb
```
Checkpoint: the notebook opens in Colab and you can see the 22 cells.

**STEP 2: Set the runtime to GPU**
`Runtime → Change runtime type → T4 GPU`, then `Runtime → Restart session`.
Checkpoint: no error; runtime shows "T4 GPU" in the top-right corner.

**STEP 3: Mount Drive and sync the repo**
Run the first code cell (mounts Drive, clones/pulls into `/content/drive/MyDrive/NSTT-Lite`).
Checkpoint: Drive is mounted and `%cd` printed the NSTT-Lite path with no error.

**STEP 4: Install dependencies and verify GPU**
Run the `pip install -r requirements.txt` cell, then the GPU-check cell.
Checkpoint: output shows `CUDA available: True` and a real device name (e.g. "Tesla T4").
**Screenshot this cell's output now** — it's your Plan 4 Step 6 evidence for GPU/library versions.
If it says `False`, stop and fix Step 2 before continuing.

**STEP 5: Confirm the real corpus/manifests are on Drive**
Run the data-check cell (`wc -l data/manifests/*.jsonl`).
Checkpoint: you see real counts (train≈12,300 / val≈1,484 / test≈1,675), not zero/missing files.
If missing, copy `data/manifests/` and `data/openslr54_ne/` from your local machine to
`MyDrive/NSTT-Lite/` before continuing.

**STEP 6: Start TensorBoard, then launch training**
Run the `%tensorboard --logdir models/finetuned_full/runs` cell first, then
`!python scripts/run_colab_training.py`.
Checkpoint: training starts printing step progress; TensorBoard panel is open and will
start showing a loss curve within the first few hundred steps. This is the
multi-hour step — leave it running.

**STEP 7: Handle a disconnect, if it happens**
If Colab disconnects mid-run: reconnect, re-run Steps 3–4 (mount + install), then
re-run the training cell — it resumes from the last checkpoint automatically
(`run_colab_training.py` and the underlying `Seq2SeqTrainer` pick up from
`models/finetuned_full/checkpoint-*` on Drive rather than starting over).
Checkpoint: training continues from a step number greater than 0, not from step 0 again.

**STEP 8: Confirm training finished and capture the numbers**
When the training cell finishes, it prints `train_loss`, `eval_metrics`, and the
checkpoint path. **Screenshot the TensorBoard loss/WER curves now.**
Then read the final numbers directly:
```python
import json
state = json.load(open("models/finetuned_full/trainer_state.json"))
print("global_step:", state["global_step"])
print("final eval_wer:", [x for x in state["log_history"] if "eval_wer" in x][-1]["eval_wer"])
```
Checkpoint: `models/finetuned_full/model.safetensors` exists on Drive; you have a
`global_step` number and an `eval_wer` number written down.

**STEP 9: Re-evaluate on the full test split**
Run the two full-eval cells (fine-tuned checkpoint, then zero-shot baseline) —
these transcribe all 1,675 test utterances, not the 150-sample local subset.
Checkpoint: `reports/eval_finetuned_full.json` and `reports/eval_baseline_full.json`
both exist with real WER/CER numbers, not the old 150-sample ones.

**STEP 10: Run the matched-step ablation**
Run the `make_leaky_manifests.py` + `run_colab_ablation_training.py` cells, then the
matching full-eval cell for the leaky checkpoint.
Checkpoint: `models/finetuned_leaky_full/` exists, trained for the **same** step
count as Step 8's `global_step` (the script reads it automatically from
`trainer_state.json`) — this is what fixes Plan 3's confounded 300-vs-100-step
comparison. Compute the WER gap: `eval_finetuned_full.json` vs the new leaky-full result.

---

After Step 10, come back and tell me: the `global_step`, `eval_wer` from Step 8, and
the full-test-split WER/CER from Steps 9–10. I'll take it from there — Steps 3–5 of
`plan4.md` (gender classifier, error analysis, redeployment on the full-scale
checkpoint, and the report update) don't need Colab and I can run them locally once
the checkpoint is on your machine (or synced back from Drive).
