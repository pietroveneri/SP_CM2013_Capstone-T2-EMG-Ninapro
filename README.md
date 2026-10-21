# CM2013 Capstone — EMG Gesture Recognition with Ninapro DB1

Team project for CM2013 Biomedical Signal Processing & Data Analytics.

The project investigates hand-gesture classification using the Ninapro DB1 surface-EMG dataset, with particular attention to the difference between within-subject and new-subject generalization.

## Historical baseline — invalid under the current hold-out protocol

The supplied pipeline was evaluated on all 27 subjects using `cfg={"seed": 0}`, including the now-reserved S23–S27. The results below are retained only as historical records and are invalid for the current hold-out protocol. A replacement baseline on S1–S22 is pending.

| Evaluation mode | Macro-F1 |
| --- | --- |
| Within-subject | 0.798 ± 0.049 |
| New-subject | 0.204 ± 0.074 |

Full experiment history is recorded in `docs/RESULTS.md`.

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
data/                   # local working data
```

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
