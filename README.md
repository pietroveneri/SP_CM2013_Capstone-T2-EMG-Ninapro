# CM2013 Capstone — EMG Gesture Recognition with Ninapro DB1

Team project for CM2013 Biomedical Signal Processing & Data Analytics.

The project investigates hand-gesture classification using the Ninapro DB1 surface-EMG dataset, with particular attention to the difference between within-subject and new-subject generalization.

## Setup and data

The archived baseline used **CPython 3.13.2, 64-bit**, on Windows. From the
repository root, create a fresh virtual environment and install the archived
baseline packages together with the feature-comparison dependency:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt -r requirements.txt -r docs/Baseline/baseline_20261022_holdout_separated/requirements-baseline-runtime.txt
python -m pip check
```

`requirements-lock.txt` pins the baseline's core packages; `requirements.txt`
adds pandas for the feature comparison. The archived runtime file pins the
other packages recorded for the baseline. Packages required only by pandas are
resolved by pip.

To download the development data, run from the repository root:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from emg_ninapro import EMGNinaproTrack; EMGNinaproTrack().download('NinaPro_Mat', subset=range(1, 23))"
```

This downloads and extracts the [NinaPro DB1](https://ninapro.hevs.ch/data/DB1.html)
archives for S1–S22 into `NinaPro_Mat/s1/` through `s22/`; existing E1 files are
reused. Alternatively, place the corresponding `S*_A1_E1.mat` files in that
directory. The loader reads them recursively, uses the `restimulus` and
`rerepetition` labels, excludes rest and creates 200 ms windows with a 150 ms
step. Keep the reserved S23–S27 data out of development and do not commit the
dataset files. See [data/README.md](data/README.md).

## Run the pipeline

With the environment activated, run from the repository root:

```powershell
python src/emg_ninapro.py
python src/run_baseline.py
```

The first command runs `smoke()` on synthetic data without downloading the
dataset. Its numbers are plumbing checks. The second runs the supplied
50-feature baseline on S1–S22, downloading missing development files if needed,
and reports both within-subject and new-subject results.

Run `python src/run_mixed_5050.py` to evaluate the 50/50 ensemble; it requires the development data to be prepared beforehand.

To repeat the development feature-set comparison:

```powershell
python src/compare_feature_sets.py
```

The comparison uses the five archived new-subject folds, no additional
preprocessing and the same baseline classifier for the 70-, 60-, 40- and
20-feature sets. It requires all 22 development subjects and also computes
permutation importance, so it can take considerable time. Fold scores,
subject scores, feature importance and the summary are written to
`outputs/feature_selection/`. This comparison does not change the features
used by `run_baseline.py`. The `full_60` set excludes RMS; the comparison
results are recorded in [docs/RESULTS.md](docs/RESULTS.md).

Install the optional envelope CNN dependencies and run the experiment from the
repository root:

```shell
.venv/bin/python -m pip install -r requirements-cnn.txt
.venv/bin/python src/envelope_cnn/run_envelope_cnn.py
```

On Intel macOS, where newer PyTorch wheels are unavailable, this installs
PyTorch 2.2.2 and uses NumPy 1.x with compatible pandas, SciPy, and ContourPy
versions. Other platforms use PyTorch 2.4 or newer with NumPy 2.

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
