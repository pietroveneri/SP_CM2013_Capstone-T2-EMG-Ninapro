# Baseline — 2026-10-22 — hold-out separated

Ninapro DB1 E1: **22 development subjects (S1–S22)**, 12 gestures and **60,182 windows per evaluation mode**. S23–S27 were neither read nor scored; final hold-out evaluation remains pending.

| Evaluation mode | Split | Mean subject macro-F1 ± sample SD | Pooled macro-F1 |
|---|---|---|---|
| Within-subject | 5-fold GroupKFold on repetitions per subject | 0.794 ± 0.048 | 0.793 |
| New-subject | 5-fold GroupKFold on subjects | 0.177 ± 0.055 | 0.196 |

Both modes use out-of-fold predictions. Within-subject comprises 110 folds; new-subject comprises 5 folds and is not LOSO. New-subject macro-F1 across folds is **0.190 ± 0.016** (sample SD); keep this separate from the subject spread above.

## Run records

- `summary.json`: pooled metrics and separate subject/fold spreads.
- `config.json`, `folds.json`: effective configuration and train/test assignments.
- `run.json`, `run.log`: execution metadata, source hashes and original runner output.
- `environment.json`, `requirements-baseline-runtime.txt`: imported versions and dependency record.
- `data_manifest.json`, `validation.json`: development-file hashes and split checks.
- `within_subject/`, `new_subject/`: predictions, per-subject/per-fold metrics, confusion matrices and the existing harness report.
- `cleanup_audit.json`: disposition of every previous baseline result file.
- `checksums.json`: archive file hashes.

## Scope and validation

Executed the existing `src/run_baseline.py`, using `EMGNinaproTrack.evaluate_modes()` and the adapter/reporting functions unchanged. Eight existing hold-out tests passed. All 115 folds were checked; each mode has exactly one OOF prediction per development window. Reserved MAT reads: zero. No final model was fitted or hold-out submission generated. No staging or commit was performed; pre-existing staging was preserved.

The original package inventory stopped after prediction export because seven additional distribution metadata directories were unreadable. Archiving resumed from saved predictions, without refitting; the metrics were recomputed with the existing functions and matched. Imported core versions match the readable metadata. See `environment.json`; no packages were changed.

This replaces the [invalid 2026-10-10 baseline](../baseline_20261010_REDACTED_NOHELDOUT/README.md), which used all 27 subjects without a separate hold-out test. That earlier exposure is not undone by this correction. The final expected-score registration remains pending.
