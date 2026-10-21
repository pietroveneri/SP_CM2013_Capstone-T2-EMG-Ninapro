# Data

The project uses Ninapro DB1 Exercise E1: 12 finger movements, 10 surface-EMG
channels and a 100 Hz sampling rate. The 27 subjects are separated locally:

```text
NinaPro_Mat/             # relative to the repository root
├── s1/ ... s22/         # development: training, tuning and cross-validation
data/
├── holdout/
│   └── s23/ ... s27/    # final inference only
└── raw/
    ├── NinaproDB1.zip   # original source archive
    └── *.csv           # original source exports, when present
```

Use `NinaPro_Mat` as the baseline cache. `EMGNinaproTrack.load(cache_dir,
exercise="E1")` searches recursively for E1 `.mat` files, so the per-subject
folders need not be flattened. `Recording.group` uses uppercase subject IDs
(`S1`–`S27`); the reserved IDs `S23`–`S27` match `adapter.py`'s hold-out checks.

`download()` and `load()` exclude the reserved subjects by default. Develop and
evaluate both modes on S1–S22 only. Do not load the original ZIP/CSV archives for
cross-validation: they may contain all 27 subjects and are retained only as
source archives.

After freezing the final model and registering `EXPECTED_SCORE.md`, use
`download_holdout("data/holdout")` if the reserved files are not already available,
then `load("data/holdout", exercise="E1", include_holdout=True)` and
`write_submission()` with the frozen model. Do not score the hold-out or use it
for model selection; the instructor scores `predictions.csv`.

Dataset files are local and must not be committed to Git. See
[`src/emg_ninapro.py`](../src/emg_ninapro.py) for loading and
[`src/adapter.py`](../src/adapter.py) for evaluation and submission guards.
