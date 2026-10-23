# CM2013 Capstone — EMG Gesture Recognition with Ninapro DB1

Team project for CM2013 Biomedical Signal Processing & Data Analytics.

The project investigates hand-gesture classification using the Ninapro DB1 surface-EMG dataset, with particular attention to the difference between within-subject and new-subject generalization.

## Development baseline — hold-out separated

The supplied pipeline was rerun unchanged on **S1–S22** on 2026-10-22: 60,182 windows per mode. **S23–S27 were neither read nor scored** and remain reserved for final inference.

| Evaluation mode | Mean subject macro-F1 ± sample SD | Pooled macro-F1 |
| --- | --- | --- |
| Within-subject | 0.794 ± 0.048 | 0.793 |
| New-subject | 0.177 ± 0.055 | 0.196 |

New-subject CV uses five subject folds; macro-F1 across folds is 0.190 ± 0.016. The [run archive](docs/Baseline/baseline_20261022_holdout_separated/README.md) contains predictions, fold assignments and checks. Full results and decisions are in [docs/RESULTS.md](docs/RESULTS.md).

The historical all-27 baseline had no separate hold-out test and is invalid under the current protocol. Earlier exposure to S23–S27 remains disclosed in the results log. The final expected-score registration is still pending.

## Final evaluation

Follow the [course hold-out protocol](https://github.com/farhad-abtahi/CM2013/blob/main/tracks/HOLDOUT_EVALUATION.md):
1. Develop and cross-validate on **S1–S22 only**. `download()` and `load()` exclude S23–S27 by default; evaluation and final training call `assert_no_holdout()`.
2. Report both evaluation modes; the final hold-out uses **new-subject macro-F1**, with subject-level splits.
3. Freeze the final pipeline and commit a completed `EXPECTED_SCORE.md` with mean ± SD **across CV folds**, the split unit, fold count and final-model commit. The current file is an unregistered draft, not a score registration.
4. Only then call `download_holdout("data/holdout")`, load it with `include_holdout=True`, and generate `predictions.csv` once with `write_submission()` and the frozen model. Do not score the hold-out or change the model afterwards; the instructor scores the submission.

## Repository structure

```text
src/
├── adapter.py          # supplied evaluation/pipeline framework
├── emg_ninapro.py      # EMG track integration
├── report.py           # supplied reporting utilities
├── run_baseline.py     # supplied baseline runner
├── preprocessing.py    # team preprocessing
├── features.py         # team feature engineering
├── models.py           # team ML models
└── experiments.py      # experiment configurations/runners

docs/                   # track documentation and RESULTS.md
Notes/                  # internal notes
Reference/              # literature/background material
NinaPro_Mat/
├── s1/ ... s22/         # development subjects; default baseline cache
data/
├── holdout/
│   └── s23/ ... s27/    # reserved for final inference after model freeze
└── raw/                # original ZIP/CSV archives, not CV inputs
```

The real-data loader recursively reads Exercise E1 `.mat` files from the chosen
cache. Use `NinaPro_Mat` for development; `load()` excludes S23–S27 by default even
if they are present in a cache. Subject IDs in `.mat` filenames become uppercase
`Recording.group` values such as `S23`, matching the adapter's `holdout_ids` and
`assert_no_holdout()` checks. See [data/README.md](data/README.md) for the data layout.

Each experiment or isolated change is developed on a short-lived branch:
```
git switch main
git pull
git switch -c experiment/<name>
```
After completing the experiment:
```
git add .
git commit -m "Add per-channel normalization"
git push -u origin experiment/normalization
```
Then open a Pull Request into main.

## Experiments
For every experimental iteration:
1. Start from the current accepted pipeline.
2. Change one clearly defined component when possible.
3. Keep the prescribed group-aware validation.
4. Evaluate both within-subject and new-subject performance.
5. Report the metric together with its spread.
6. Update docs/RESULTS.md.
7. Record negative or rejected experiments as well as successful ones.
