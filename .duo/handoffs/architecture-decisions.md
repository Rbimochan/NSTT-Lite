# NSTT-Lite — Architecture Decision Log

## ADR-001 — Migrate from NSTT to NSTT-Lite, reusing existing src/ modules
**Date:** 2026-07-16
**Decision:** Scaffold NSTT-Lite as a brownfield migration that copies the original
NSTT project's already-tested `src/` modules (`slr54.py`, `preprocessing.py`,
`manifests.py`, `splits.py`, `evaluation.py`, `error_analysis.py`, `training.py`,
`pipeline.py`) rather than rewriting the SLR54 download/preprocess/split/eval logic
from scratch.
**Alternatives considered:**
- Fresh scaffold, data-only reuse: would have meant re-deriving already-solved
  preprocessing/splitting logic for no benefit.
- Scaffold-only, no code yet: would defer known-safe reuse to a later plan for no
  reason, given the code is already tested and working.
**Rationale:** NSTT-Lite's roadmap (Plans 1-3) covers ground the original NSTT
project already solved (download, resample, NFC-normalize, speaker-disjoint split,
WER/CER evaluation, error categorization). Rebuilding it would waste effort the
100-mark rubric doesn't reward, and risks reintroducing bugs (dependency pinning,
Drive I/O bottlenecks, resumability) that were already debugged in the original
project. New work is isolated to genuinely new scope: gender labels/classification
(Plan 2, 6), TensorBoard + early stopping (Plan 4-5), CTranslate2/Faster-Whisper
deployment (Plan 8).
**Trade-offs:** The reused modules will need light adaptation (e.g. `pipeline.py`
currently drives the original NSTT's full-corpus/no-gender-label flow; NSTT-Lite
needs a 10-20hr subset with gender labels, so this file will need Plan-2-specific
extension, not a straight port).
**Impact:** T-001 marked `Ready for Implementation` (dependency/download plumbing
already exists); T-002 onward remain `Planned` pending the new gender-label and
subset-selection logic these files don't yet have.
