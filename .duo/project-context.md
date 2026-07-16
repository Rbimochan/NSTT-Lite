# NSTT-Lite — Project Context
> Authoritative narrative knowledge base. Owned by Claude. Machine state lives in project-state.json.

## Vision
A leaner, re-scoped rebuild of NSTT (Nepali Speech-to-Text) that satisfies a stricter
100-mark academic rubric. Source of truth for scope/sequencing is
`../NSTT-Lite_10_Plan_Roadmap.md` (10 sequential plans) — this file exists to give
Claude/Cursor day-to-day working context; the roadmap is the canonical spec.

## Goals & Target Users
- Primary user: a single student/researcher, same context as the original NSTT project
  (ST7088CEM Artificial Neural Networks, Softwarica College / Coventry University).
- Goal: fine-tune Whisper-Small for Nepali ASR (as before), **plus** a second, distinct
  task — speaker gender classification from Whisper encoder embeddings — required by
  this rubric's "Difficulty" criterion (more than one task).
- Goal: production-style deployment via CTranslate2/Faster-Whisper (not just raw
  HuggingFace inference), benchmarked for latency.

## Scope & MVP
**In scope:**
- Everything the original NSTT project already built (SLR54 download/preprocess/split,
  Whisper-Small fine-tuning, WER/CER evaluation, error analysis) — reused, not rebuilt.
- NEW: speaker gender labels/classification (Plan 6).
- NEW: zero-shot baseline numbers for both tasks, logged before fine-tuning (Plan 3).
- NEW: TensorBoard logging + early stopping (patience 2 on val WER) in the training loop (Plan 4-5).
- NEW: CTranslate2 conversion + Faster-Whisper latency benchmark (Plan 8).
- NEW: Streamlit app extended to show both tasks + inference time (Plan 8).
- NEW: reproducibility evidence package — screenshots, appendix-ready code, proposal
  reproduction (Plan 9).

**Out of scope:** anything beyond the 10 plans in the roadmap; no new tasks invented
without updating the roadmap file first.

## Functional Requirements
See `../NSTT-Lite_10_Plan_Roadmap.md` Plans 1–10 for the authoritative, itemized list.
Summary:
1. Env + data acquisition (Colab T4, SLR54 download, screenshots).
2. Preprocessing + speaker-disjoint 10-20hr subset + gender labels.
3. Zero-shot baselines (ASR WER/CER, gender-classification trivial baseline).
4. Fine-tuning script (Seq2SeqTrainer, FP16, batch 2, grad_accum 4, 5 epochs, early stop).
5. Full fine-tuning execution + TensorBoard monitoring + best-checkpoint selection.
6. Gender classifier on Whisper encoder embeddings (logistic regression / small MLP).
7. Final evaluation + ASR error analysis + classification error analysis.
8. CTranslate2 conversion, Faster-Whisper benchmark, extended Streamlit app.
9. Evidence/appendix assembly (screenshots, code, proposal cross-check).
10. Report writing + submission (4500 words, Campus 4.0).

## Non-Functional Requirements
- Must run within Google Colab free-tier constraints (T4 GPU, session timeouts) —
  same operational lessons as the original NSTT project apply (Drive I/O is slow for
  many small files; process locally then batch-copy to Drive; make long-running loops
  resumable/checkpointed, since disconnects/quota limits are routine, not exceptional).
- Reproducible: pinned dependency versions, seeded splits, every experiment
  screenshotted with visible device/software info (rubric requires this explicitly).
- Academic integrity: any code adapted from external sources must be clearly attributed.
- No external links in the final report except the dataset link; the 1-page proposal
  must be reproduced verbatim in the appendix.

## Architecture
Same offline batch pipeline as NSTT, extended with a second branch after the shared
Whisper encoder:

```
[SLR54 corpus] → [Preprocess: resample, NFC-normalize, filter, gender labels, split]
        │
        ▼
[Whisper-Small fine-tuning] ──────────────┐
        │                                  │
        ▼                                  ▼
[ASR eval: WER/CER,          [Gender classifier: mean-pooled encoder
 error analysis]              embeddings → logistic regression/MLP,
        │                     accuracy/precision/recall/F1/confusion matrix]
        │                                  │
        └──────────────┬───────────────────┘
                        ▼
        [CTranslate2 conversion + Faster-Whisper benchmark]
                        │
                        ▼
        [Streamlit app: both tasks + timing + WER/CER display]
```

## Technology Stack
Everything NSTT already used (`transformers==4.49.0`, `huggingface_hub==0.27.1`,
`datasets`, `accelerate`, `evaluate`, `jiwer`, `librosa`, `soundfile`, `streamlit`,
`pandas`) plus, new for NSTT-Lite: `tensorboard`, `ctranslate2`, `faster-whisper`,
`scikit-learn` (gender classifier head).

## Folder Structure
```
NSTT-LITE/
├── data/                     # raw + processed datasets (gitignored)
├── notebooks/                # Colab notebooks, one family per Plan
├── src/                      # reusable Python modules (copied from NSTT, extended)
├── models/                   # checkpoints (gitignored, synced to Drive)
├── reports/                  # WER/CER, gender-classification metrics, error analysis
├── dashboard/                # Streamlit app (extended for both tasks)
└── .duo/                     # Duo framework workspace (this scaffold)
```

## Coding Standards & Conventions
Same as NSTT: PEP 8, type hints in `src/`, notebooks are thin orchestration layers,
all randomness seeded, external code attributed with citation comments.

## API Conventions
Not applicable — offline ML experimentation, no API surface.

## Data / Database Strategy
- Same SLR54 corpus as NSTT. **Do not re-download from scratch** — the original NSTT
  project's Colab/Drive already has the full ~8GB corpus downloaded and extracted;
  point NSTT-Lite's data prep at that same Drive location (or re-run the download step,
  which already skips files that exist) rather than pulling fresh.
- NEW: derive/extract speaker gender labels from SLR54 metadata for the classification
  task (roadmap Plan 2) — this is new work, not present in the original NSTT pipeline.
- NEW: the roadmap calls for a speaker-disjoint **10-20 hour subset**, not the full
  ~165 hours — smaller than original NSTT's approach, chosen for faster iteration
  given the added scope (two tasks, more deliverables) within the same time budget.

## Constraints
Same as NSTT: Colab free-tier session/GPU limits, low-resource language tooling gaps,
academic integrity requirements. Additionally: this project has a strict 4500-word
report limit and a 100-mark rubric with itemized criteria (see roadmap file) — every
plan's output should be traceable to specific marks it protects.

## Roadmap
Canonical roadmap lives in `../NSTT-Lite_10_Plan_Roadmap.md`. Duo task IDs T-001
through T-010 map 1:1 to Plan 1 through Plan 10 in that file — see project-state.json.
