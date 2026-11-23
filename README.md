# CM2013 Capstone — EMG Gesture Recognition with Ninapro DB1

Team project for CM2013 Biomedical Signal Processing & Data Analytics.

The project investigates hand-gesture classification using the Ninapro DB1 surface-EMG dataset, with particular attention to the difference between within-subject and new-subject generalization.

## Run a Random Forest with Savitzky–Golay smoothing

From the repository root, after setup, run

```shell
python src/classifiers.py --model rf --features time_40 --preprocessing median-savgol
```

This command takes the 20-sample windows from S1–S22, applies a median filter of
length 3 and a Savitzky–Golay filter of length 5 and order 2 to each channel,
extracts 40 time-domain features and trains the supplied 200-tree Random Forest
inside each fold. The filters are centred, so the decision uses samples up to
the end of the window. The new-subject setting is **No calibration**.

## Setup and data

The archived development runs used 64-bit CPython 3.13.2 on Windows. In PowerShell, run

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python src/emg_ninapro.py --download-development
```

The download reuses cached E1 files and fills in missing development subjects
under `NinaPro_Mat/`. You can also supply the `S*_A1_E1.mat` files there yourself.
Run `python src/emg_ninapro.py` for an offline synthetic smoke test. See
[data layout](data/README.md).

## Classical experiments

Use `python src/classifiers.py` for RF, shrinkage LDA, RBF SVM or 5-NN. Select
one model (`rf`, `lda`, `svm`, `knn`), one feature set and, optionally, a filter.
The feature set applies to all ten channels.

| `--features` | Columns | Per-channel features |
| --- | ---: | --- |
| `baseline` | 50 | MAV, RMS, waveform length, variance and rFFT centroid |
| `full_70` | 70 | MAV, waveform length, variance, relative MAV, mean frequency, median frequency and RMS |
| `full_60` | 60 | Expanded set with RMS removed |
| `time_40` | 40 | MAV, waveform length, variance and relative MAV |
| `amplitude_20` | 20 | MAV and relative MAV |

To reproduce the supplied baseline and the November smoothing experiment on the
same test windows, run

```shell
python src/classifiers.py --model rf --features baseline
python src/classifiers.py --model rf --features baseline --preprocessing median-savgol
```

The default preprocessing is `none`. All runs use the archived five subject folds
and five repetition folds per subject. Feature scaling is fitted on the training
windows of each fold. Each run writes indexed predictions, per-subject and
per-fold scores, plots and run metadata to
`outputs/<model>_<features>_<preprocessing>/`. To repeat a run, supply a fresh
`--output-dir`. For the full option list, use `python src/classifiers.py --help`.

The filter recipes are in [preprocessing.py](src/preprocessing.py), the feature
definitions are in [features.py](src/features.py), and model defaults are in
[classifiers.py](src/classifiers.py). The [preprocessing notes](docs/FOLD_SAFE_PREPROCESSING.md)
map the old option numbers to the CLI and document the historical normalization runs.

### Reproduce a specific analysis

| Task | Command from the repository root |
| --- | --- |
| Compare RF feature sets on archived new-subject folds and compute permutation importance | `python src/compare_feature_sets.py` |
| Compare a baseline feature ablation against the saved subject scores | `python src/compare_baseline_ablation.py noRMS` or `python src/compare_baseline_ablation.py norFFT` |
| Reproduce the 50/50 RF-time and LDA-baseline ensemble | `python src/run_mixed_5050.py` |
| Inspect feature distributions before and after smoothing | `python src/compare_preprocessing.py` |
| Recompute the expanded feature figures | `python src/visualize_features.py` |
| Verify the archived classifier, ensemble, MLP and smoothing predictions | `python src/audit/verify_replay.py` |
| Verify the two archived fold-fitted z-score runs | `python src/audit/verify_preprocessing.py` |

The [audit guide](docs/audit_20261119/README.md) includes commands for replaying
the full historical classifier grid and checking its outputs.

### Envelope CNN and feature MLP

Install the optional envelope CNN dependencies and run the experiment from the
repository root:

```shell
.venv/bin/python -m pip install -r requirements-cnn.txt
.venv/bin/python src/MLP/run_envelope_cnn.py
```

On Windows PowerShell, use `.venv\Scripts\python.exe` in place of
`.venv/bin/python` in these commands.

Use `--input 60` for the 60-feature MLP (or `40` or `20` for the other
feature sets). Each run writes to a fresh directory under `outputs/`, such as
`outputs/mlp_60`. Use `--output-dir PATH` to choose another run directory.
Generate figures from a run with, for example:

```shell
.venv/bin/python src/MLP/plot_envelope_cnn.py outputs/mlp_60
```

On Intel macOS, where newer PyTorch wheels are unavailable, this installs
PyTorch 2.2.2 and uses NumPy 1.x with compatible pandas, SciPy, and ContourPy
versions. Other platforms use PyTorch 2.15.0 with NumPy 2.

## Development baseline — hold-out separated

The supplied pipeline was rerun unchanged on **S1–S22** on 2026-10-22: 60,182 windows per mode. **S23–S27 were neither read nor scored** and remain reserved for final inference.

| Evaluation mode | Mean subject macro-F1 ± sample SD | Pooled macro-F1 |
| --- | --- | --- |
| Within-subject | 0.794 ± 0.048 | 0.793 |
| New-subject · No calibration | 0.177 ± 0.055 | 0.196 |

New-subject · No calibration uses five subject folds. Macro-F1 across folds is 0.190 ± 0.016. The [run archive](docs/Baseline/baseline_20261022_holdout_separated/README.md) contains predictions, fold assignments and checks. Full results and decisions are in [docs/RESULTS.md](docs/RESULTS.md).

The historical all-27 baseline had no separate hold-out test and is invalid under the current protocol. Earlier exposure to S23–S27 remains disclosed in the results log. The final expected-score registration is still pending.

## Final evaluation

Follow the [course hold-out protocol](https://github.com/farhad-abtahi/CM2013/blob/main/tracks/HOLDOUT_EVALUATION.md):
1. Develop and cross-validate on **S1–S22 only**. `download()` and `load()` exclude S23–S27 by default; evaluation and final training call `assert_no_holdout()`.
2. Report both evaluation modes; the final hold-out uses **new-subject macro-F1**, with subject-level splits.
3. Freeze the final pipeline and commit a completed `EXPECTED_SCORE.md` with mean ± SD **across CV folds**, the split unit, fold count and final-model commit. The current file is an unregistered draft, not a score registration.
4. Only then call `download_holdout("data/holdout")`, load it with `include_holdout=True`, and generate `predictions.csv` once with `write_submission()` and the frozen model. Do not score the hold-out or change the model afterwards; the instructor scores the submission.

## Repository structure

- `src/` contains the executable workflows and their implementations. The course adapter and DB1 loader live in `src/adapter.py` and `src/emg_ninapro.py`.
- `docs/` contains the results log, original classifier logs and figures, archived predictions, and fold definitions. See [results](docs/RESULTS.md).
- `outputs/` holds local reproductions. `NinaPro_Mat/` supplies the development recordings. `data/holdout/` is reserved for final inference.
- [Notes](Notes/) and [Reference](Reference/) contain course and background material. The [original code guide](Notes/Capstone%20Project%20-%20Quick%20Guide%20to%20all%20Code.pdf) describes an earlier repository layout. For current paths and commands, see [Classical experiments](#classical-experiments).

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
