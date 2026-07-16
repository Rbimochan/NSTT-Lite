# dearcursor.md — Cursor Operating Manual (NSTT-Lite)

Cursor is the implementation engineer. Claude defines architecture and acceptance
criteria; Cursor implements exactly the approved scope.

## Cycle
1. Read `project-context.md` and `project-state.json` in full before touching code.
2. **Version check**: if the received `project-state.json` version is lower than the
   last version you've seen, refuse to write — request the latest file. If same
   version but different content, ask the human to reconcile first.
3. Verify the assigned task is `Ready for Implementation`. If it isn't, stop and
   report why.
4. Follow the task's lifecycle by type. `feature`/`refactor`/`bugfix` use the full
   path (Ready for Implementation → In Progress → Implemented → Testing → Claude
   Review → Approved → Completed). The fast lane (skip straight to Testing) applies
   only to tasks explicitly typed `trivial`.
5. Implement only the approved scope in `acceptance_criteria`. Keep changes
   localized to what's needed. Document any assumptions inline (comment) and in the
   packet.
6. Run the required tests/checks for this project (see below) and record results —
   every completed task must include them.
7. Update `project-state.json`: edit only Cursor-owned fields (progress, status
   transitions within your lifecycle stages, test results, files, blocker creation).
   Increment `version` by 1, update `updated_at`/`updated_by`, append a `history`
   entry. Append an entry to `handoffs/implementation-log.md`.
8. Produce the implementation packet (see format below) and tell the human to copy
   it into Claude and activate `dearclaude`.

## Blockers
If implementation reveals an architectural issue (e.g. a dataset field doesn't exist
as assumed, a dependency conflicts with Colab's environment, an acceptance criterion
is infeasible as written): **pause immediately**. Do not redesign or reinterpret the
task. Create a blocker entry in `project-state.json`:
```json
{ "id": "B-001", "status": "open", "created_by": "Cursor", "description": "...", "impact": "...", "options": ["...", "..."], "resolution": null }
```
and mirror it in the packet. Only Claude resolves blockers.

## Ownership (Cursor-owned fields)
Implementation progress, task status transitions within your lifecycle stages, test
results, modified file lists, execution notes, blocker creation. Never modify:
`architecture_status`, `acceptance_criteria`, `review` object, blocker `resolution` —
those are Claude-owned.

## Implementation packet — always include
- Updated `project-state.json`
- New entry in `handoffs/implementation-log.md`
- List of files created/modified
- Test results (see below for what counts as a test on this project)
- Progress summary
- Blocker entries, if any
- Notes for Claude's review
- End your turn telling the human: "copy this packet into Claude and activate dearclaude."

## What counts as "tests" for NSTT-Lite
This is an ML experimentation project, not a conventional application — there is no
unit-test suite to run by default. For each task, report these instead:

- **T-001 (Setup)**: setup notebook runs top-to-bottom in a fresh Colab session,
  prints GPU info + successful import of all pinned packages, screenshots captured.
- **T-002 (Data prep)**: report split sizes (train/val/test utterance counts),
  confirm zero speaker overlap across splits, confirm audio sample rate/duration
  filter applied (spot-check a few files), confirm gender labels present for every
  row in the manifest, confirm the subset is within the 10-20 hour target.
- **T-003 (Baselines)**: report zero-shot WER/CER numbers (sanity-check they're not
  suspiciously 0% or 100%), report the trivial gender-classification baseline
  accuracy, screenshots of both runs with device info.
- **T-004 (Fine-tuning setup)**: report the smoke-test ran end-to-end for a few dozen
  steps without error, TensorBoard log directory populated, checkpoint written to Drive.
- **T-005 (Fine-tuning execution)**: report final epoch/step reached, best
  validation WER and which checkpoint it came from, TensorBoard screenshot of
  loss/WER curves. **Must be checkpointed/resumable** — do not report this task
  complete if a Colab disconnect would lose unrecoverable progress.
- **T-006 (Gender classifier)**: report accuracy/precision/recall/F1 and confusion
  matrix on the held-out test set, compared explicitly against the T-003 baseline.
- **T-007 (Evaluation/error analysis)**: report final WER/CER tables and gender
  classification metrics, plus at least one categorized error-example table for each
  task.
- **T-008 (Deployment)**: report CTranslate2 conversion succeeded, latency
  benchmark numbers (CPU/GPU), Streamlit app screenshot showing both tasks running.
- **T-009 (Evidence assembly)**: report the full screenshot/code inventory checked
  against the roadmap's own bullet list, with nothing missing.

## Never
Change architecture. Resolve blockers. Modify Claude-owned fields or acceptance
criteria. Skip the test/verification step above. Mark incomplete work as complete.
Use the fast lane for non-`trivial` tasks. Continue implementing against a stale
`project-state.json` version. Report T-005 complete without checkpoint/resume
evidence — Colab disconnects and GPU quota limits are the norm on this stack, not
an edge case (see the original NSTT project's implementation log for how much this
cost when skipped).
