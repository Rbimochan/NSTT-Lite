# Plan 1 (Amended) — Post-Migration State

## What changed from the original Plan 1

The original NSTT-Lite Plan 1 (environment setup) and the 10-plan roadmap remain
the source of truth for the Whisper-Small ASR + gender-classification project.
This amendment records what was added *alongside* it via the XLS-R migration —
it does not replace, restructure, or invalidate any prior plan.

**Nothing was deleted.** `main`, PR #1 (`fix/pipeline-corpus-path`), and PR #2
(`feature/plan2-plan3-execution`, containing all of Plans 1–4's completed work)
are untouched. A new branch `plan4-continuation` was cut from PR #2's tip for
future work, currently empty (no diff, no PR yet).

## What was added: T-011 — XLS-R Baseline Audit

**Purpose.** Before resuming Plan 4 (full-scale Colab training), we ran an
independent sanity check on `gagan3012/wav2vec2-xlsr-nepali`'s published 5.97%
WER claim — verifying it wasn't an artifact of a narrow, single-speaker,
self-reported eval, and checking for gender bias.

**Scope.** One-time comparison baseline only (confirmed via elicitation). No
XLS-R fine-tuning track, no XLS-R deployment benchmark. This sits alongside the
Whisper-Small pipeline, not in competition with it as a build target.

**Files added:**
- `src/xlsr_baseline.py` — dataset loading, model loading, preprocessing
  (ported from the verified Colab notebook, cells 1–4)
- `scripts/run_xlsr_baseline.py` — runnable entry point, mirrors the convention
  of `scripts/run_baseline.py`
- `data/manifests/xlsr_openslr43_test_manifest.csv` — manifest for the audit run
- `reports/xlsr_baseline_results.json` — results, kept in two clearly separated
  blocks (see below)
- `.duo/project-state.json` — T-011 added (v4), independent of T-001…T-010

## Findings (for the record — these drive next steps, not just this plan)

Two corpora, **never merged**, per the report's own `corpus_separation_note`:

| Corpus | Utterances | WER | Gender breakdown |
|---|---|---|---|
| `openslr43_in_domain` (OpenSLR-43, `gauravparajuli/slr43`) | 150 | **4.91%** | None — single-speaker female corpus, no metadata. Confirms our own run beats the published 5.97% on the same narrow distribution; no dataset swap or cherry-picking found. |
| `openslr54_diversity_check` (OpenSLR-54, existing NSTT-Lite test manifest) | 150 (108 male / 42 female, shuffled seed=42 — not the full 1,675-row test split) | **65.28%** | 70.16% (pseudo-labeled "male") / 52.14% (pseudo-labeled "female") — F0-pitch heuristic, **not verified ground truth** (same method NSTT-Lite's own gender classifier already found unreliable, 44.8% acc.) |

**The headline finding:** the model's real-world generalization gap is large
(4.91% → 65.28% WER moving from narrow to diverse speech). The apparent
male/female WER gap inside OpenSLR-54 is suggestive but should be described as
"WER by pitch-threshold group," not "WER by gender," until backed by
self-reported demographic labels.

## Open items carried forward

- Full-scale Whisper-Small Colab T4 run (Plan 4 / T-005) is still the primary
  blocker — this audit does not change that priority.
- If male/female WER gap is worth investigating further, it needs a dataset
  with *real* self-reported gender metadata (e.g. Common Voice contributor
  fields) before drawing conclusions — not the F0 pseudo-label.
- `plan4-continuation` branch exists but is empty; no action needed until the
  Colab run produces results to build on.
