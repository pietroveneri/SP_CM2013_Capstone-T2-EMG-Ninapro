# Historical baseline — 2026-10-10 — invalid: no hold-out separation

**Superseded.** S23–S27 were included in training and CV. These results had no separate hold-out test and are invalid under the current protocol. Earlier exposure is not undone by the corrected run.

Use the [S1–S22 replacement](../baseline_20261022_holdout_separated/README.md) for development comparisons. The original numbers below are retained only for provenance.

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

Obsolete local predictions, logs and diagnosis artifacts were removed during the correction. The metadata in this folder records the invalid historical run.

## Scope

Do not reuse these folds or results for current comparisons. Out-of-fold evaluation on all 27 subjects did not provide a separate final test set.

Historical metadata is retained here; current results and complete predictions are in the replacement folder.
