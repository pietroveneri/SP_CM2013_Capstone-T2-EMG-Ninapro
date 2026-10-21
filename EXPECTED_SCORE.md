# Expected hold-out score — EMG Ninapro DB1

**Status: DRAFT — NOT REGISTERED.** No final model is frozen and no valid replacement CV score has been produced. Committing this template does not register an expected score.

| Required field | Value |
| --- | --- |
| Evaluation mode | new-subject |
| Development subjects | S1–S22 only |
| Hold-out subjects | S23–S27 |
| Metric | macro-F1 |
| CV mean ± sample SD across folds | PENDING — compute on S1–S22 |
| Split unit | subject |
| Number of CV folds | PENDING — record the final run |
| Frozen final-model commit | PENDING |

Complete and commit this file after freezing the final pipeline and before final hold-out access. Keep the existing per-subject spread separate from the required across-fold statistic.

Prior exposure: the historical 27-subject baseline included S23–S27 before their hold-out designation was published; see `docs/RESULTS.md`. This template does not retroactively establish an untouched hold-out. If hold-out data already appears in repository history, clarify the registration chronology with the instructor; do not rewrite history.

After registration, download into `data/holdout`, load with `include_holdout=True`, and write `predictions.csv` once using the frozen model. Do not score the hold-out or update the model afterwards.
