# Verified baseline — 2026-10-10

Baseline evaluation on Ninapro DB1, exercise E1: 27 subjects, 12 gestures and 75,197 windows per evaluation mode.

| Evaluation mode | Split | Mean subject macro-F1 ± sample SD | Pooled macro-F1 |
|---|---|---|---|
| Within-subject | 5-fold GroupKFold on repetitions per subject | 0.798 ± 0.046 | 0.796 |
| New-subject | 5-fold GroupKFold on subjects | 0.205 ± 0.073 | 0.221 |

Both results use out-of-fold predictions. The within-subject evaluation comprises 135 folds in total: five per subject. The new-subject evaluation is **not LOSO**.

## Run records

- `summary.json`: pooled metrics and per-subject spread.
- `config.json`: effective run configuration.
- `run.json`: execution metadata and source provenance.
- `environment.json`: execution environment and package versions.
- `folds.json`: train/test fold assignments.

The complete local archive also contains per-mode outputs, the execution log and a source snapshot. Retain these alongside the files committed here.

## Scope

These results establish the reference for subsequent experiments. Reuse the recorded folds for controlled comparisons. Out-of-fold evaluation does not constitute an untouched final test set.

Files supplied by the course framework remain unchanged; project-specific validation and archiving belong in separate team-owned modules.
Run metadata and summaries are versioned here; complete predictions and per-subject outputs are retained in the local archive.