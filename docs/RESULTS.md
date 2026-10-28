# Results log — `EMG NinaPro`, `T2`

## Hold-out correction — 2026-10-21; replacement run — 2026-10-22

**Previously registered 27-subject results had no separate hold-out test:** S23–S27 were included in training and CV before their hold-out designation was published. They are invalid under the current protocol and retained only as [historical metadata](Baseline/baseline_20261010_REDACTED_NOHELDOUT/README.md). Earlier exposure is not undone by the correction.

The replacement baseline (and all further results) uses **S1–S22 only**. **S23–S27 remain reserved for final new-subject evaluation** and were neither read nor scored in this run. The existing loader and adapter guards were verified; no source code was changed.

**Track:** `emg_ninapro` · **Subjects:** 22 · **Classes:** g01–g12; rest excluded · **Windows:** 60,182 per mode.

**Evaluation modes:** within-subject + new-subject · **Split unit:** repetition within each subject; subject for new-subject · **CV:** 5 repetition folds per subject (110 total); 5 subject folds for new-subject, not LOSO.

**Primary metric:** macro-F1 · **Secondary metrics:** Cohen's κ, balanced accuracy, accuracy · **Headline spread:** mean, sample SD and min–max across 22 subjects. New-subject macro-F1 across the 5 folds is **0.190 ± 0.016**; this is a separate development reference for the future expected-score registration.

**Detailed results and decision log:** [BASELINE.md](Baseline/BASELINE.md) · **Complete outputs:** [2026-10-22 baseline](Baseline/baseline_20261022_holdout_separated/README.md).

```python
cfg = {
    "seed": 0,
    "select": "none",
    "imbalance": "balanced",
    "cv_max_splits": 5,
    "loso_max_groups": 12
}
```

## Iteration log

**2026-10-28 preprocessing audit:** iterations 3–5 are invalid as inductive
cross-validation estimates. Their per-subject normalization was fitted on all
windows before the subject/repetition splits, including the evaluated windows.
The values below are retained only as historical, transductive results; they do
not establish performance or a significant difference. The identity-preprocessing
baseline in iteration 2 is not affected by this specific leak. Corrected runs
must fit every learned preprocessing parameter on each training fold only.

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 | 2026-10-08 | Historical supplied baseline; verification on 2026-10-10 used all 27 subjects without a separate hold-out test | Invalid under current protocol; see historical archive | — | Provenance only; superseded | `379f1e6` |
| 2 | 2026-10-22 | Rerun unchanged pipeline on S1–S22; reserve S23–S27 | within_subject mean macro_f1 0.794 (sd 0.048, range 0.706-0.880 across 22 subjects); new_subject mean macro_f1 0.177 (sd 0.055, range 0.064-0.276 across 22 subjects) | Not comparable: different cohort and folds | Correct subject scope; no tuning or configuration search | ` 8ed9ab8` |
| 3 | 2026-10-24 | Per-subject z-score, no filter | **Invalid: leakage.** Historical within 0.773 ± 0.053; new 0.242 ± 0.063 (subject SD) | Not assessed | Historical only | `ce1db46` |
| 4 | 2026-10-24 | Peak scaling, no centering or filter (guard 0.05) | **Invalid: leakage.** Historical within 0.795 ± 0.051; new 0.195 ± 0.056 (subject SD) | Not assessed | Historical only | `ce1db46` |
| 5 | 2026-10-24 | Median (3) + Savitzky-Golay (5, 2) + z-score | **Invalid: leakage.** Historical within 0.763 ± 0.054; new 0.244 ± 0.064 (subject SD) | Not assessed | Historical only | `ce1db46` |

### Current preprocessing without signal normalization — 2026-10-28

Options 2 and 5 are now identity: their former z-score or peak scaling has
been removed. They correspond to the identity-preprocessing baseline in
iteration 2. Option 3 retains median (3) and Savitzky–Golay (5, order 2),
without signal normalization. The feature scaler remains fitted on the training
fold as part of the classifier. Results are development estimates; SD is the
sample standard deviation.

| Recipe | Within-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across 5 folds | Archive |
|---|---:|---:|---:|---|
| Identity (options 2 and 5) | 0.794 ± 0.048 | 0.177 ± 0.055 | 0.190 ± 0.016 | [Baseline](Baseline/baseline_20261022_holdout_separated/README.md) |
| Median (3) + Savitzky–Golay (5, 2), no normalization | 0.793 ± 0.049 | 0.177 ± 0.053 | 0.191 ± 0.012 |  |

Each smoothing evaluation has one out-of-fold prediction for each of 60,182
development windows. The split assignments and input/source hashes were checked;
hold-out reads were not instrumented and no hold-out evaluation was performed.
These point estimates do not establish a statistically significant difference.

### Historical fold-fitted normalization experiments — 2026-10-28

These experiments remain valid for their stated protocol, but are **superseded**
by the current preprocessing design, which does not normalize the signal.
They do not describe the current code or its expected score.

Normalization is fitted separately on training windows in every fold. For
new-subject folds, channel statistics are pooled over training subjects and
applied to the unseen subjects without fitting on their recordings. Within-subject
folds use only that subject's training repetitions. The baseline features and
classifier are unchanged. These are development results, not final hold-out
scores. Standard deviations below are sample SDs.

| Recipe | Within-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across 5 folds | Archive |
|---|---:|---:|---:|---|
| Z-score; no filter | 0.771 ± 0.052 | 0.175 ± 0.052 | 0.188 ± 0.019 | [Run and validation](Preprocessing/fold_safe_20261028/) |
| Peak scaling; guard 0.05; no filter | 0.794 ± 0.048 | 0.176 ± 0.055 | 0.190 ± 0.015 | [Run and validation](Preprocessing/peak_20261028/) |
| Median (3) + Savitzky–Golay (5, 2) + z-score | 0.761 ± 0.053 | 0.174 ± 0.050 | 0.186 ± 0.017 | [Run and validation](Preprocessing/smooth_20261028/) |

The z-score archive is marked `partial` because the original three-option run
was stopped after this recipe completed, to avoid repeating an independently
running peak-scaling experiment. Its two evaluation modes and validation are
complete. These point estimates alone do not establish statistical significance.
See [preprocessing protocol](Preprocessing/README.md) for filtering and
deployment limitations.

### Baseline provenance

- Existing runner: `src/run_baseline.py`.
- One configuration evaluated; 115 fits; 8 existing hold-out tests passed; zero reserved MAT reads. Complete fold and prediction checks passed.
- [Run metadata](Baseline/baseline_20261022_holdout_separated/run.json), [validation](Baseline/baseline_20261022_holdout_separated/validation.json), [environment](Baseline/baseline_20261022_holdout_separated/environment.json) and [cleanup audit](Baseline/baseline_20261022_holdout_separated/cleanup_audit.json).
- Historical registration: `379f1e6`; historical verification: `0b85ef5`. 


## Decision log — the choices behind the numbers

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | DB1 E1; S1–S22; restimulus/rerepetition; rest excluded | Historical all-27 scope (invalid) | Reserve S23–S27 for final new-subject inference | 2 | Yes: corrected subject scope |
| 2. Preprocessing |  |  |  |  |  |
| 3. Feature extraction |  |  |  | |  |
| 4. Feature selection |  |  |  |  |  |
| 5. Classification, incl. `imbalance` | seed 0 |  |  | |  |
| 6. Inference |  | |  |  | |
| 7. Reporting |  | |  |  |  |

## Revisions — the ones that went backwards

| # | The downstream result that triggered it | Which earlier decision it indicted | What you changed | What happened to the metric |
|---|---|---|---|---|
| 1 | Hold-out protocol revealed S23–S27 in historical training/CV | All-27 baseline scope | Reran the supplied pipeline on S1–S22 and regenerated development folds | New-subject subject mean 0.177; not directly comparable with the invalid all-27 result |

## Who did what

| Iteration | Who | Modules / tasks owned | Reviewed by |
|---|---|---|---|
| 1 | Pietro | Ran the supplied baseline and recorded its results | Invalid |
| 2 | Pietro | Executed the existing runner; verified hold-out guards and artifacts; updated results and baseline decisions | Pending team review |

## Final numbers

| | Value | Under what split |
|---|---|---|
| Primary metric, development baseline (mean + spread) |  | 5-fold GroupKFold on S1–S22 subjects |
| Secondary metrics, development baseline | | Same subject folds |
| Held-out set — scored once, never tuned against |  |  |
| Supplied baseline, same harness | Within-subject 0.794 ± 0.048; new-subject 0.177 ± 0.055 (subject spread) | Corrected S1–S22 scope |
| Configurations evaluated in the replacement run |  |  |
| How option comparison was kept separate from final reporting |  |  |
