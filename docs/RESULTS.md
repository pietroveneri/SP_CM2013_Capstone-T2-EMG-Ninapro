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

| # | Date | What changed & why (one line) | Primary metric **with spread** | Better than previous? | If not — why it was kept | Commit |
|---|---|---|---|---|---|---|
| 1 | 2026-10-08 | Historical supplied baseline; verification on 2026-10-10 used all 27 subjects without a separate hold-out test | Invalid under current protocol; see historical archive | — | Provenance only; superseded | `379f1e6` |
| 2 | 2026-10-22 | Rerun unchanged pipeline on S1–S22; reserve S23–S27 | within_subject mean macro_f1 0.794 (sd 0.048, range 0.706-0.880 across 22 subjects); new_subject mean macro_f1 0.177 (sd 0.055, range 0.064-0.276 across 22 subjects) | Not comparable: different cohort and folds | Correct subject scope; no tuning or configuration search | ` 8ed9ab8` |
| 3 | 2026-10-24 | Run of selected preprocessing option without editing baseline features/classifiers: per-subject z-score only, no filter | within_subject 0.773 (sd 0.053, range 0.675–0.875); new_subject 0.242 (sd 0.063, range 0.058–0.368) | no significant difference | potential to improve with adjusted feature extraction / classifiers | `ce1db46` |
| 4 | 2026-10-24 | Run of selected preprocessing option without editing baseline features/classifiers: peak scaling, no centering, no filter (guard 0.05) (Wei et al. (2022)) | within_subject 0.795 (sd 0.051, range 0.697–0.879); new_subject 0.195 (sd 0.056, range 0.068–0.316) | no significant difference | potential to improve with adjusted feature extraction / classifiers | `ce1db46` |
| 5 | 2026-10-24 | Run of selected preprocessing option without editing baseline features/classifiers: median (3) + Savitzky-Golay (5, 2) + z-score | within_subject 0.763 (sd 0.054, range 0.663–0.866); new_subject 0.244 (sd 0.064, range 0.065–0.388) | no significant difference | worth passing on to test whether light smoothing helps once features change | `ce1db46` |

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

## Final numbers

| | Value | Under what split |
|---|---|---|
| Primary metric, development baseline (mean + spread) |  | 5-fold GroupKFold on S1–S22 subjects |
| Secondary metrics, development baseline | | Same subject folds |
| Held-out set — scored once, never tuned against |  |  |
| Supplied baseline, same harness | Within-subject 0.794 ± 0.048; new-subject 0.177 ± 0.055 (subject spread) | Corrected S1–S22 scope |
| Configurations evaluated in the replacement run |  |  |
| How option comparison was kept separate from final reporting |  |  |
