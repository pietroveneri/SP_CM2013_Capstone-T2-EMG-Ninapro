## Baseline configuration — 2026-10-22

DB1 E1: **S1–S22**, 12 gestures, **60,182 windows per mode**. S23–S27 are reserved for the final hold-out test and were neither read nor scored in this run. The earlier 27-subject baseline had no separate hold-out test and is invalid under the current protocol; its prior exposure remains disclosed.

[Complete run records](baseline_20261022_holdout_separated/README.md) · [Invalid historical metadata](baseline_20261010_REDACTED_NOHELDOUT/README.md)
| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | DB1 E1; S1–S22; restimulus/rerepetition; rest excluded | Historical all-27 scope (invalid) | Reserve S23–S27 for final new-subject inference | 2 | Yes: corrected subject scope |
| 2. Preprocessing | Identity; 20-sample windows, 15-sample step at 100 Hz | No alternatives evaluated | Preserve the supplied reference on the stored EMG envelope | 2 | No |
| 3. Feature extraction | 50 features: MAV, RMS, WL, variance and magnitude-spectrum frequency centroid per channel | No alternatives evaluated | Preserve the supplied representation for controlled comparisons | 2 | No |
| 4. Feature selection | `select="none"` | No alternatives evaluated | Keep all 50 features in the reference run | 2 | No |
| 5. Classification, incl. `imbalance` | StandardScaler -> RF; 200 trees; `class_weight="balanced"`; seed 0 | No alternatives evaluated | Fit scaling and the supplied classifier inside each training fold | 2 | No |
| 6. Inference | OOF; 5 repetition folds per subject; 5 subject folds for new-subject | Historical folds containing S23–S27 (invalid) | Keep evaluation groups separate from training in each mode | 2 | Yes: development folds regenerated |
| 7. Reporting | Both modes; per-subject spread; pooled metrics, confusion matrices and separate per-fold spread | Historical all-27 reports (invalid) | Distinguish subject variability from the across-fold expected-score statistic | 2 | Yes: corrected scope and provenance |

within_subject mean macro_f1 0.794 (sd 0.048, range 0.706-0.880 across 22 subjects)

### Results — split unit: repetition (within subject) (n=22)

**Confusion matrix** (rows = true, columns = predicted, row-normalised)

| true \ pred | g01 | g02 | g03 | g04 | g05 | g06 | g07 | g08 | g09 | g10 | g11 | g12 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **g01** | 0.84 | 0.04 | 0.04 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 5303 |
| **g02** | 0.04 | 0.85 | 0.03 | 0.02 | 0.00 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.02 | 5149 |
| **g03** | 0.03 | 0.03 | 0.79 | 0.03 | 0.04 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 5632 |
| **g04** | 0.02 | 0.03 | 0.05 | 0.80 | 0.03 | 0.03 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 4723 |
| **g05** | 0.01 | 0.01 | 0.04 | 0.03 | 0.83 | 0.02 | 0.02 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 4918 |
| **g06** | 0.01 | 0.01 | 0.03 | 0.03 | 0.04 | 0.77 | 0.04 | 0.05 | 0.01 | 0.01 | 0.00 | 0.01 | 5005 |
| **g07** | 0.02 | 0.00 | 0.02 | 0.02 | 0.02 | 0.04 | 0.80 | 0.05 | 0.01 | 0.00 | 0.00 | 0.01 | 5217 |
| **g08** | 0.01 | 0.00 | 0.03 | 0.01 | 0.02 | 0.05 | 0.06 | 0.77 | 0.02 | 0.01 | 0.01 | 0.01 | 5270 |
| **g09** | 0.01 | 0.02 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | 0.69 | 0.06 | 0.10 | 0.05 | 4753 |
| **g10** | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.01 | 0.06 | 0.80 | 0.05 | 0.04 | 4617 |
| **g11** | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.08 | 0.06 | 0.77 | 0.03 | 4527 |
| **g12** | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.05 | 0.04 | 0.04 | 0.80 | 5068 |

**macro_f1** — mean macro_f1 0.794 (sd 0.048, range 0.706-0.880 across 22 subjects); pooled 0.793
- cohens_kappa: pooled 0.775, mean over subjects 0.776
- balanced_accuracy: pooled 0.793, mean over subjects 0.793
- accuracy: pooled 0.794, mean over subjects 0.795
- worst subject: S14 (macro_f1 0.706, n=2906) — read this one's errors before the mean's

*Now read it:* which classes trade places in the matrix, is the spread driven by one hard group, and what would you change next?

new_subject mean macro_f1 0.177 (sd 0.055, range 0.064-0.276 across 22 subjects)

### Results — split unit: subject (n=22)

**Confusion matrix** (rows = true, columns = predicted, row-normalised)

| true \ pred | g01 | g02 | g03 | g04 | g05 | g06 | g07 | g08 | g09 | g10 | g11 | g12 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **g01** | 0.19 | 0.12 | 0.08 | 0.14 | 0.03 | 0.02 | 0.07 | 0.03 | 0.04 | 0.09 | 0.07 | 0.13 | 5303 |
| **g02** | 0.06 | 0.42 | 0.05 | 0.02 | 0.03 | 0.02 | 0.05 | 0.03 | 0.03 | 0.09 | 0.11 | 0.09 | 5149 |
| **g03** | 0.06 | 0.10 | 0.14 | 0.10 | 0.05 | 0.02 | 0.13 | 0.10 | 0.04 | 0.08 | 0.10 | 0.08 | 5632 |
| **g04** | 0.14 | 0.04 | 0.07 | 0.23 | 0.04 | 0.05 | 0.10 | 0.08 | 0.05 | 0.08 | 0.03 | 0.07 | 4723 |
| **g05** | 0.06 | 0.07 | 0.11 | 0.07 | 0.19 | 0.01 | 0.11 | 0.12 | 0.06 | 0.04 | 0.07 | 0.10 | 4918 |
| **g06** | 0.03 | 0.05 | 0.06 | 0.20 | 0.03 | 0.14 | 0.10 | 0.10 | 0.10 | 0.05 | 0.08 | 0.04 | 5005 |
| **g07** | 0.06 | 0.07 | 0.08 | 0.11 | 0.08 | 0.05 | 0.18 | 0.12 | 0.04 | 0.03 | 0.09 | 0.09 | 5217 |
| **g08** | 0.02 | 0.06 | 0.08 | 0.06 | 0.07 | 0.04 | 0.08 | 0.27 | 0.06 | 0.07 | 0.09 | 0.10 | 5270 |
| **g09** | 0.04 | 0.08 | 0.06 | 0.05 | 0.09 | 0.05 | 0.04 | 0.06 | 0.11 | 0.16 | 0.12 | 0.14 | 4753 |
| **g10** | 0.09 | 0.12 | 0.06 | 0.05 | 0.04 | 0.01 | 0.02 | 0.08 | 0.06 | 0.20 | 0.15 | 0.11 | 4617 |
| **g11** | 0.05 | 0.15 | 0.08 | 0.02 | 0.05 | 0.01 | 0.03 | 0.07 | 0.07 | 0.15 | 0.21 | 0.09 | 4527 |
| **g12** | 0.09 | 0.14 | 0.06 | 0.02 | 0.08 | 0.00 | 0.04 | 0.10 | 0.10 | 0.15 | 0.11 | 0.12 | 5068 |

**macro_f1** — mean macro_f1 0.177 (sd 0.055, range 0.064-0.276 across 22 subjects); pooled 0.196
- cohens_kappa: pooled 0.127, mean over subjects 0.127
- balanced_accuracy: pooled 0.199, mean over subjects 0.201
- accuracy: pooled 0.199, mean over subjects 0.200
- worst subject: S2 (macro_f1 0.064, n=2637) — read this one's errors before the mean's

*Now read it:* which classes trade places in the matrix, is the spread driven by one hard group, and what would you change next?

New-subject macro-F1 across 5 subject folds: **0.190 ± 0.016** (sample SD). This is a development baseline reference; the final model is not frozen and no hold-out score is registered.

### Follow-up — not applied

Within-subject CV currently uses `min(5, n_rep)` and ignores `cv_max_splits`; expose that setting consistently before comparing other fold counts. This does not affect the present five-fold run. The environment also contains seven unreadable additional distribution metadata directories; inspect them before future package updates. No source or package changes were made.
