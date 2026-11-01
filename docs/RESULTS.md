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
| 3 | 2026-10-24 | Per-subject z-score, no filter | **Invalid: leakage.** Historical within 0.773 (sd 0.053, range 0.675–0.875); new 0.242 (sd 0.063, range 0.058–0.368) | Not assessed | Historical only | `ce1db46` |
| 4 | 2026-10-24 | Peak scaling, no centering or filter (guard 0.05) | **Invalid: leakage.** Historical within 0.795 (sd 0.051, range 0.697–0.879); new 0.195 (sd 0.056, range 0.068–0.316) | Not assessed | Historical only | `ce1db46` |
| 5 | 2026-10-24 | Median (3) + Savitzky-Golay (5, 2) + z-score | **Invalid: leakage.** Historical within 0.763 (sd 0.054, range 0.663–0.866); new 0.244 (sd 0.064, range 0.065–0.388) | Not assessed | Historical only | `ce1db46` |
| 6 | 2026-10-25 | Ablation on the baseline: remove RMS from each channel | within_subject 0.788 (sd 0.051, range 0.692–0.877); new_subject 0.178 (sd 0.054, range 0.079–0.271) across 22 subjects | vs baseline: within_subject −0.006736; new_subject +0.001654 mean subject macro_f1 | comparison of RMS contribution | `9d3dcd9` |
| 7 | 2026-10-25 | Ablation on the baseline: remove rFFT mean frequency from each channel | within_subject 0.797 (sd 0.048, range 0.703–0.881); new_subject 0.178 (sd 0.052, range 0.065–0.261) across 22 subjects | vs baseline: within_subject +0.002661; new_subject +0.000990 mean subject macro_f1 | comparison of mean frequency contribution | `9d3dcd9` |
| 8 |  |  | | | | `` |
| 9 |  |  |  |    |  | `` |
| 10 |  |  |  |  |  | `` |
| 11 | 2026-11-01 | Random Forest with full feature set | Within 0.809 (SD 0.047, range 0.727–0.888); new 0.176 (SD 0.051, range 0.075–0.273) | vs #2: within +0.015; new −0.001 | Feature-set comparison | `e6800f3` |
| 12 | 2026-11-01 | Random Forest with time feature set | Within 0.812 (SD 0.046, range 0.727–0.885); new 0.172 (SD 0.052, range 0.071–0.270) | vs #11: within +0.003; new −0.004 | Feature-set comparison | `e6800f3` |
| 13 | 2026-11-01 | Random Forest with amplitude feature set | Within 0.807 (SD 0.044, range 0.728–0.878); new 0.170 (SD 0.048, range 0.075–0.241) | vs #12: within −0.005; new −0.002 | Feature-set comparison | `e6800f3` |
| 14 | 2026-11-01 | LDA with baseline feature set | Within 0.716 (SD 0.047, range 0.595–0.783); new 0.190 (SD 0.053, range 0.109–0.274) | vs #2: within −0.078; new +0.013 | Classifier comparison | `e6800f3` |
| 15 | 2026-11-01 | LDA with full feature set | Within 0.742 (SD 0.051, range 0.607–0.806); new 0.180 (SD 0.063, range 0.075–0.315) | vs #14: within +0.026; new −0.010 | Feature-set comparison | `e6800f3` |
| 16 | 2026-11-01 | LDA with time feature set | Within 0.739 (SD 0.051, range 0.593–0.802); new 0.179 (SD 0.065, range 0.071–0.315) | vs #15: within −0.003; new −0.001 | Feature-set comparison | `e6800f3` |
| 17 | 2026-11-01 | LDA with amplitude feature set | Within 0.720 (SD 0.052, range 0.573–0.780); new 0.169 (SD 0.066, range 0.060–0.278) | vs #16: within −0.019; new −0.010 | Feature-set comparison | `e6800f3` |
| 18 | 2026-11-01 | SVM with baseline feature set | Within 0.757 (SD 0.053, range 0.639–0.836); new 0.181 (SD 0.060, range 0.060–0.285) | vs #2: within −0.037; new +0.004 | Classifier comparison | `e6800f3` |
| 19 | 2026-11-01 | SVM with full feature set | Within 0.777 (SD 0.050, range 0.670–0.851); new 0.179 (SD 0.061, range 0.067–0.306) | vs #18: within +0.020; new −0.002 | Feature-set comparison | `e6800f3` |
| 20 | 2026-11-01 | SVM with time feature set | Within 0.792 (SD 0.047, range 0.696–0.872); new 0.175 (SD 0.060, range 0.069–0.309) | vs #19: within +0.015; new −0.004 | Feature-set comparison | `e6800f3` |
| 21 | 2026-11-01 | SVM with amplitude feature set | Within 0.798 (SD 0.046, range 0.704–0.875); new 0.162 (SD 0.058, range 0.067–0.294) | vs #20: within +0.006; new −0.013 | Feature-set comparison | `e6800f3` |
| 22 | 2026-11-01 | k-NN (n=5) with baseline feature set | Within 0.675 (SD 0.062, range 0.543–0.754); new 0.150 (SD 0.039, range 0.077–0.219) | vs #2: within −0.119; new −0.027 | Classifier comparison | `e6800f3` |
| 23 | 2026-11-01 | k-NN (n=5) with full feature set | Within 0.688 (SD 0.055, range 0.561–0.780); new 0.148 (SD 0.036, range 0.082–0.208) | vs #22: within +0.013; new −0.002 | Feature-set comparison | `e6800f3` |
| 24 | 2026-11-01 | k-NN (n=5) with time feature set | Within 0.763 (SD 0.050, range 0.673–0.859); new 0.149 (SD 0.043, range 0.071–0.224) | vs #23: within +0.075; new +0.001 | Feature-set comparison | `e6800f3` |
| 25 | 2026-11-01 | k-NN (n=5) with amplitude feature set | Within 0.775 (SD 0.051, range 0.686–0.867); new 0.142 (SD 0.043, range 0.077–0.216) | vs #24: within +0.012; new −0.007 | Feature-set comparison | `e6800f3` |

Iterations 11–25 report mean macro-F1 and spread across 22 subjects. Differences use the rounded means above; they do not establish statistical significance. Iterations 14, 18 and 22 also use #2 as the reference for the baseline feature set; the other comparisons use the preceding row. Results are retained for comparison, without implying a final selection. 


### Current preprocessing without signal normalization — 2026-10-28

Option 3 retains median (3) and Savitzky–Golay (5, order 2), without signal
normalization. The feature scaler remains fitted on the training fold as part
of the classifier. Results are development estimates; SD is the sample standard
deviation.

| Recipe | Within-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across 5 folds | Archive |
|---|---:|---:|---:|---|
| Median (3) + Savitzky–Golay (5, 2), no normalization | 0.793 ± 0.049 | 0.177 ± 0.053 | 0.191 ± 0.012 | Not available |

The smoothing run reported one out-of-fold prediction per development window,
with split assignments and input/source hashes checked. Hold-out reads were not
instrumented and no hold-out evaluation was performed. 

### Historical fold-fitted normalization experiments — 2026-10-28

These development experiments use a superseded preprocessing design. The
reported protocol fitted normalization on training windows in each fold. The
original implementation and run metadata are unavailable, so that fitting
procedure cannot be verified from the saved predictions. These results do not
describe the current preprocessing or a final hold-out score. SDs are sample
standard deviations.

| Recipe | Within-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across subjects | New-subject macro-F1, mean ± SD across 5 folds | Archive |
|---|---:|---:|---:|---|
| Z-score; no filter | 0.771 ± 0.052 | 0.175 ± 0.052 | 0.188 ± 0.019 | [Saved outputs](Preprocessing/fold_safe_20261028/zscore/) |
| Peak scaling; guard 0.05; no filter | 0.794 ± 0.048 | 0.176 ± 0.055 | 0.190 ± 0.015 | Not available |
| Median (3) + Savitzky–Golay (5, 2) + z-score | 0.761 ± 0.053 | 0.174 ± 0.050 | 0.186 ± 0.017 | [Saved outputs](Preprocessing/smooth_20261028/median_savgol_zscore/) |

The z-score and median + z-score predictions and test folds can be checked with
`src/preprocessing_noleakage_pietro/verify_saved_predictions.py`. The point
estimates do not establish statistical significance. See the
[preprocessing protocol](FOLD_SAFE_PREPROCESSING.md) for the current filtering
and deployment limitations.

### Baseline provenance

- Existing runner: `src/run_baseline.py`; source revision `a534402e790e014265de3837accf4b97b6d45f28`; executed sources unchanged and hashed in `run.json`.
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
| 3 | Hannah | Added pre-processing methods | Invalid due to leakage |
| 4 | Hannah | Added pre-processing methods | Invalid due to leakage |
| 5 | Hannah | Added pre-processing methods | Invalid due to leakage |
| 6 | Pietro | Ablation study without RMS  | Pending team review |
| 7 | Pietro | Ablation study without rFFT  | Pending team review |
| 8->10 | Thomas | Feature extraction  | Pending team review |
| 11->20 | Pietro | Explored different classifiers | Pending team review |





## Final numbers

| | Value | Under what split |
|---|---|---|
| Primary metric, development baseline (mean + spread) |  | 5-fold GroupKFold on S1–S22 subjects |
| Secondary metrics, development baseline | | Same subject folds |
| Held-out set — scored once, never tuned against |  |  |
| Supplied baseline, same harness | Within-subject 0.794 ± 0.048; new-subject 0.177 ± 0.055 (subject spread) | Corrected S1–S22 scope |
| Configurations evaluated in the replacement run |  |  |
| How option comparison was kept separate from final reporting |  |  |
