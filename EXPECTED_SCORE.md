# Expected hold-out score — EMG Ninapro DB1

**Status: DRAFT — NOT REGISTERED.** No final model is frozen. The corrected S1–S22 baseline is a development reference, not a final score registration. No staging or commit was performed for this run.

| Required field | Value |
| --- | --- |
| Evaluation mode | new-subject |
| Development subjects | S1–S22 only |
| Hold-out subjects | S23–S27 |
| Metric | macro-F1 |
| CV mean ± sample SD across folds | PENDING — register the final frozen run |
| Split unit | subject |
| Number of CV folds | PENDING — record the final run |
| Frozen final-model commit | PENDING |

Baseline reference (2026-10-22): new-subject macro-F1 **0.190 ± 0.016 across 5 subject folds**; subject spread **0.177 ± 0.055**. See [the run records](docs/Baseline/baseline_20261022_holdout_separated/README.md). Register the final frozen pipeline separately.

Complete and commit this file after freezing the final pipeline and before final hold-out access. Keep the existing per-subject spread separate from the required across-fold statistic.

Prior exposure: the historical 27-subject baseline included S23–S27 before their hold-out designation was published; see `docs/RESULTS.md`. This template does not retroactively establish an untouched hold-out. If hold-out data already appears in repository history, clarify the registration chronology with the instructor; do not rewrite history.

After registration, download into `data/holdout`, load with `include_holdout=True`, and write `predictions.csv` once using the frozen model. Do not score the hold-out or update the model afterwards.
