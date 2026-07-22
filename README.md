# NSTT-Lite — Nepali Speech-to-Text (Rescoped)

A leaner rebuild of the original [NSTT](https://github.com/Rbimochan/NSTT) project,
restructured around a stricter 100-mark academic rubric. Fine-tunes Whisper-Small
for Nepali ASR, adds a second task (speaker gender classification), and benchmarks
production-style deployment (CTranslate2/Faster-Whisper).

## Where this stands right now

**Plans 1–3 are complete and verified on disk** (branch `feature/plan2-plan3-execution`,
[PR #2](https://github.com/Rbimochan/NSTT-Lite/pull/2)) — real corpus, real
preprocessing, a working fine-tuning/eval/deployment pipeline, and a report draft.
All of it ran on a local machine (CPU/Apple Silicon MPS, no CUDA) at **reduced
scale** to stay tractable without a GPU.

**Plan 4 is in progress**: closing that scale gap with a real Colab T4 GPU run.
This is the current bottleneck — see [Plan 4](#plan-4--closing-the-scale-gap-in-progress) below.

**Separately, T-011 (XLS-R comparison baseline) is done.** As a one-time sanity
check — not a second fine-tuning track — `gagan3012/wav2vec2-xlsr-nepali`'s
published 5.97% WER was audited: **4.91% WER on OpenSLR-43** (its own in-domain
training/self-eval corpus, single female speaker) vs. **65.28% WER on
OpenSLR-54** (this repo's existing, multi-speaker test manifest — a completely
separate corpus, never merged with OpenSLR-43 numbers). See
[reports/xlsr_baseline_results.json](reports/xlsr_baseline_results.json).

| | Local (done) | Full-scale (Plan 4, pending) |
|---|---|---|
| Training steps | 300 (~16% of 1 epoch) | 2000 / 5 epochs, early-stopped |
| Zero-shot baseline | WER 221.8%, CER 147.4% | to be re-measured on full test split |
| Fine-tuned | WER 70.1%, CER 22.0% | pending Colab run |
| Ablation (speaker-disjoint vs. leaky split) | confounded — unequal step counts (300 vs 100) | pending matched-step re-run |
| Gender classifier | 44.8% acc. (below 75.2% majority baseline) | pending re-run on stronger encoder |

All numbers above are 150-test-utterance subsets (CPU tractability), not the full
1,675-utterance test split — Plan 4 fixes that too.

## Key facts worth knowing before touching this repo

- **The corpus has no gender metadata at all.** OpenSLR54 ships only
  `utt_spk_text.tsv` (utterance/speaker/transcript) — verified against both the
  downloaded files and the live openslr.org/54 listing. The `gender` field used
  throughout this project is an **acoustic pseudo-label** derived from mean
  pitch (F0, 165Hz threshold via `librosa.pyin`), not ground truth. Every
  gender-related number in this repo should be read as "agreement with the F0
  heuristic," not "true gender."
- **Only a partial corpus is downloaded locally.** 55,980 of 157,905 total
  utterances (96 of 256 two-character speaker-hash subdirectories) — this is
  intentional, sized to build the roadmap's 10–20 hour subset (currently 15.03hrs,
  15,459 utterances, 146 speakers), not a failed download.
- **The gender classifier underperforms its own baseline** (44.8% vs. 75.2%
  majority-class). This is reported as a genuine negative result, not hidden —
  see `reports/gender_classifier_results.json` and the report's Discussion section
  for why (likely an undertrained encoder + noisy pseudo-labels, not a bug).

## Repo map

```
NSTT-Lite_10_Plan_Roadmap.md   canonical 10-plan spec (source of truth for scope)
plan4.md / plan4manual.md      current work: closing the local→full-scale gap
start.md                        Colab/Drive-I/O landmines already solved
.duo/                            Claude & Cursor Duo Framework task tracker
  project-state.json             live status, T-001…T-010 mapped to Plan 1…10
  dearcursor.md / dearclaude.md  handoff protocol between Claude and Cursor
src/                             pipeline code: slr54.py, preprocessing.py,
                                  manifests.py, splits.py, pipeline.py,
                                  training.py, evaluation.py, error_analysis.py
scripts/                         runnable entry points (baseline, training,
                                  ablation, gender classifier, error analysis,
                                  CTranslate2 conversion, Streamlit app)
notebooks/plan4_colab.ipynb      ready-to-run Colab notebook for the full-scale
                                  training run (Plan 4 Step 1)
data/manifests/                  real train/val/test manifests (tracked in git;
                                  raw audio/corpus is gitignored, too large)
reports/                         every experimental result as a real file — WER/CER,
                                  baselines, ablation, gender classifier, error
                                  analysis, own-voice test, deployment benchmark,
                                  and report_draft.md
models/                          checkpoints (gitignored — local/Drive only)
```

## Setup

```bash
git clone https://github.com/Rbimochan/NSTT-Lite.git
cd NSTT-Lite
git checkout feature/plan2-plan3-execution   # current work-in-progress branch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` pins `ctranslate2==4.5.0` for Colab compatibility — that
version has no Python 3.13 wheel, so a local install on 3.13 needs
`pip install "ctranslate2>=4.6"` instead. This doesn't affect Colab, which runs an
older Python.

For the Colab GPU training run specifically, see `plan4manual.md` (10 click-by-click
steps) or open `notebooks/plan4_colab.ipynb` directly:
```
https://colab.research.google.com/github/Rbimochan/NSTT-Lite/blob/feature/plan2-plan3-execution/notebooks/plan4_colab.ipynb
```

## Running things locally

```bash
python scripts/run_baseline.py          # zero-shot whisper-small baseline
python scripts/run_smoke_test.py        # fast loop-correctness check (6 steps)
python scripts/run_full_training.py     # reduced-scale local fine-tuning (300 steps)
python scripts/run_finetuned_eval.py    # WER/CER on the fine-tuned checkpoint
python scripts/run_gender_classifier.py # gender classifier on encoder embeddings
python scripts/run_error_analysis.py    # error category breakdown
python scripts/run_own_voice_test.py    # test against your own recorded voice
python scripts/convert_ctranslate2.py   # CTranslate2 conversion + latency benchmark
streamlit run scripts/app.py            # demo: transcription + gender + latency
```

Most of these accept `NSTT_CHECKPOINT_DIR=<path>` to point at a different
checkpoint (e.g. a full-scale one synced back from Colab) instead of the default
`models/finetuned`.

## Status tracking

`.duo/project-state.json` is the live source of truth for task status (T-001…T-010,
mapped 1:1 to Plan 1…10). As of this writing: T-001–T-004 done, T-005 (full-scale
Colab training) is `Ready for Implementation` — the current blocker, since it needs
an actual GPU session that can't be run from an agent environment.

## Plan 4 — closing the scale gap (in progress)

Everything in Plans 1–3 proved the pipeline works end-to-end, but ran at reduced
scale to stay feasible on a laptop without a GPU. Plan 4 (`plan4.md`, or
`plan4manual.md` for the manual click-through version) re-runs the expensive steps
at their real intended scale:

1. Full-scale training on Colab T4 (2000 steps / 5 epochs, early stopping) — **the
   current blocker, requires a human to run the Colab session**
2. Re-evaluate baseline vs. fine-tuned on the *full* 1,675-utterance test split
3. Re-run the speaker-leakage ablation at *matched* step counts (fixing Plan 3's
   300-vs-100-step confound)
4. Re-run the gender classifier on the stronger encoder
5. Refresh error analysis and the own-voice test
6. Redeploy (CTranslate2 + Streamlit) against the full-scale checkpoint
7. Capture the screenshots the coursework rubric requires (GPU runtime,
   TensorBoard, running Streamlit app — all things only possible from a live
   Colab/browser session)
8. Expand the report toward the 4,500-word submission limit (currently 2,792)
9. Assemble the appendix (screenshots + code + the verbatim 1-page proposal)
10. Final proofread and submission — a human-owned step, not automatable

See `NSTT-Lite_10_Plan_Roadmap.md` for the original 10-plan spec this all traces
back to, and `start.md` for Colab-specific gotchas (dependency pinning, Drive I/O
speed, resumability) already solved from the original NSTT project.
