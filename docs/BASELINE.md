## Baseline configuration

| Pipeline module | Option chosen | Alternative(s) considered | Why this one (one sentence) | Iteration | Revised later? |
|---|---|---|---|---|---|
| 1. Data loading | DB1 E1; all 27 subjects; restimulus/rerepetition; rest excluded | Not evaluated in the baseline run | Use the prescribed dataset scope and the supplied correct labels | 1 | No |
| 2. Preprocessing | Identity signal processing | Not evaluated in the baseline run | Preserve the supplied reference pipeline; DB1 already contains an EMG envelope | 1 | No |
| 3. Feature extraction | 50 features: MAV, RMS, WL, variance and magnitude-spectrum frequency centroid per channel | Not evaluated in the baseline run | Establish the supplied feature representation as the reference for controlled comparisons | 1 | No |
| 4. Feature selection | `select="none"` | Not evaluated in the baseline run | Keep all 50 features in the reference run and evaluate selection separately later |  1|  No|
| 5. Classification | StandardScaler -> RF; 200 trees; `class_weight="balanced"`; `seed=0` | Not evaluated in the baseline run | Preserve the supplied classifier and its train-fold scaling and class weighting  | 1 | No |
| 6. Inference | Out-of-fold predictions using each fold's fitted selector and classifier pipeline |  Not evaluated in the baseline run| Score windows using models trained on the corresponding training groups only  | 1 | No |
| 7. Reporting | Both modes; per-subject macro-F1 spread; pooled secondary metrics and confusion matrices | Not evaluated in the baseline run | Expose the within/new-subject gap and subject variability  |  1|  No|

within_subject mean macro_f1 0.798 (sd 0.049, range 0.697-0.879 across 27 subjects)


### Results — split unit: repetition (within subject) (n=27)

**Confusion matrix** (rows = true, columns = predicted, row-normalised)

| true \ pred | g01 | g02 | g03 | g04 | g05 | g06 | g07 | g08 | g09 | g10 | g11 | g12 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **g01** | 0.84 | 0.04 | 0.04 | 0.02 | 0.01 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 | 6515 |
| **g02** | 0.04 | 0.86 | 0.03 | 0.01 | 0.00 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 6421 |
| **g03** | 0.03 | 0.03 | 0.79 | 0.03 | 0.03 | 0.02 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 7245 |
| **g04** | 0.02 | 0.03 | 0.04 | 0.81 | 0.02 | 0.03 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 5992 |
| **g05** | 0.01 | 0.01 | 0.04 | 0.03 | 0.84 | 0.02 | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 6008 |
| **g06** | 0.01 | 0.01 | 0.03 | 0.03 | 0.03 | 0.79 | 0.04 | 0.04 | 0.01 | 0.01 | 0.00 | 0.01 | 6283 |
| **g07** | 0.02 | 0.00 | 0.02 | 0.02 | 0.02 | 0.04 | 0.80 | 0.05 | 0.01 | 0.00 | 0.00 | 0.01 | 6382 |
| **g08** | 0.01 | 0.01 | 0.03 | 0.01 | 0.02 | 0.04 | 0.05 | 0.78 | 0.02 | 0.01 | 0.01 | 0.01 | 6618 |
| **g09** | 0.01 | 0.02 | 0.01 | 0.00 | 0.00 | 0.01 | 0.01 | 0.03 | 0.67 | 0.07 | 0.12 | 0.04 | 6057 |
| **g10** | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.07 | 0.80 | 0.05 | 0.04 | 5817 |
| **g11** | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.10 | 0.06 | 0.78 | 0.03 | 5581 |
| **g12** | 0.01 | 0.01 | 0.02 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.05 | 0.05 | 0.03 | 0.81 | 6278 |

**macro_f1** mean macro_f1 0.798 (sd 0.046, range 0.706-0.880 across 27 subjects); pooled 0.796
- cohens_kappa: pooled 0.778, mean over subjects 0.779
- balanced_accuracy: pooled 0.796, mean over subjects 0.797
- accuracy: pooled 0.797, mean over subjects 0.798
- worst subject: S14 (macro_f1 0.706, n=2906) � read this one's errors before the mean's

new_subject mean macro_f1 0.205 (sd 0.073, range 0.082-0.380 across 27 subjects)
### Results — split unit: subject (n=27)

**Confusion matrix** (rows = true, columns = predicted, row-normalised)

| true \ pred | g01 | g02 | g03 | g04 | g05 | g06 | g07 | g08 | g09 | g10 | g11 | g12 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **g01** | 0.20 | 0.08 | 0.09 | 0.13 | 0.07 | 0.03 | 0.06 | 0.05 | 0.06 | 0.07 | 0.06 | 0.10 | 6515 |
| **g02** | 0.06 | 0.37 | 0.06 | 0.03 | 0.02 | 0.09 | 0.02 | 0.03 | 0.04 | 0.08 | 0.12 | 0.07 | 6421 |
| **g03** | 0.06 | 0.08 | 0.15 | 0.10 | 0.08 | 0.05 | 0.06 | 0.11 | 0.05 | 0.11 | 0.08 | 0.08 | 7245 |
| **g04** | 0.09 | 0.05 | 0.08 | 0.33 | 0.11 | 0.05 | 0.06 | 0.06 | 0.03 | 0.06 | 0.05 | 0.03 | 5992 |
| **g05** | 0.06 | 0.03 | 0.09 | 0.12 | 0.26 | 0.03 | 0.06 | 0.11 | 0.05 | 0.04 | 0.06 | 0.09 | 6008 |
| **g06** | 0.03 | 0.06 | 0.05 | 0.16 | 0.07 | 0.21 | 0.07 | 0.13 | 0.05 | 0.06 | 0.06 | 0.05 | 6283 |
| **g07** | 0.06 | 0.07 | 0.11 | 0.14 | 0.10 | 0.07 | 0.08 | 0.11 | 0.05 | 0.04 | 0.07 | 0.08 | 6382 |
| **g08** | 0.03 | 0.04 | 0.11 | 0.05 | 0.07 | 0.08 | 0.06 | 0.28 | 0.06 | 0.10 | 0.06 | 0.07 | 6618 |
| **g09** | 0.03 | 0.07 | 0.05 | 0.06 | 0.09 | 0.04 | 0.03 | 0.08 | 0.07 | 0.16 | 0.15 | 0.16 | 6057 |
| **g10** | 0.07 | 0.07 | 0.06 | 0.04 | 0.03 | 0.01 | 0.02 | 0.06 | 0.08 | 0.27 | 0.13 | 0.17 | 5817 |
| **g11** | 0.03 | 0.09 | 0.05 | 0.02 | 0.05 | 0.03 | 0.03 | 0.07 | 0.06 | 0.13 | 0.35 | 0.09 | 5581 |
| **g12** | 0.07 | 0.13 | 0.07 | 0.02 | 0.10 | 0.01 | 0.03 | 0.07 | 0.07 | 0.16 | 0.11 | 0.16 | 6278 |

**macro_f1** mean macro_f1 0.205 (sd 0.073, range 0.082-0.380 across 27 subjects); pooled 0.221
- cohens_kappa: pooled 0.155, mean over subjects 0.156
- balanced_accuracy: pooled 0.227, mean over subjects 0.232
- accuracy: pooled 0.226, mean over subjects 0.227
- worst subject: S2 (macro_f1 0.082, n=2637) read this one's errors before the mean's