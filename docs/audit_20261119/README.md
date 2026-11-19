# Development evidence from the 19 November audit

Each evaluation uses DB1 Exercise E1, S1–S22 and 60,182 windows per mode.
The new-subject setting throughout this archive is **No calibration**. The five
test folds hold out subjects. Within-subject CV holds out repetitions in five
folds per subject (110 folds). Subject summaries give each of the 22 subjects
equal weight. Both subject and fold SDs use the sample standard deviation.

| Iterations | Evidence | Origin |
| --- | --- | --- |
| #8–#10 and their 70-feature reference | [Feature counts, fold scores, subject scores and summary](feature_selection/) | Recovered CSVs from the development feature comparison at `b23e079`. The four new-subject RF prediction files below are audit replays. |
| #11–#25 | [Indexed prediction CSVs](classifiers/) and [recomputed metrics](replay_summary.csv) | Audit replays of the classifier configurations documented with their original [RF](../../src/random_forest/rf_20261101), [LDA](../../src/lda_regularized/lda_20261101), [SVM](../../src/svm_rbf/svm_20261101) and [5-NN](../../src/knn/knn_20261101_5nn) logs at `fa65e4f`. |
| #26 | [Both prediction arrays and summary](mixed_5050/) | Recovered 50/50 ensemble outputs. Each NPZ contains the subject, repetition, fold, true label and prediction in source-window order. |
| #27–#29 | Updated run tables for [MLP-20](../../src/MLP/mlp_20/), [MLP-40](../../src/MLP/mlp_40/) and [MLP-60](../../src/MLP/mlp_60/), with subject and pooled κ | Derived from the six original MLP prediction CSVs. Each score table retains the folds actually used by that historical run. |
| Fixed median and Savitzky–Golay smoothing | [Run metadata, predictions and reports](../Preprocessing/no_normalization_audit_20261119/) | A new run of option 3 without signal normalization. The run metadata includes the SHA-256 of the 22 development MAT files and the executed sources. |

Both smoothing filters are centred within each window. A decision using these
features waits for the remaining samples in that window.

## Feature labels

At `b23e079` the order of the seven names was aligned with the values emitted
by the extractor. RF, LDA, SVM and k-NN use all 70 columns for `full_feature`.
RF `full_60` and MLP-60 explicitly exclude the RMS column of every channel.
The 40-column set retains MAV, WL, VAR and relative MAV. The 20-column set
retains MAV and relative MAV. The baseline classifier input has its own
50-column extractor.

The [four regenerated figures, ranking and split metadata](feature_figures/)
come from the expanded 70-column input. The CSV ranking and the NPZ metadata
contain all ten RMS labels. The permutation plot ranks features on one
development fold. Its reference score comes from the visualiser's own model.
The iteration metrics come from the experiment runners.

Row #27 reports a +0.0065 difference between MLP-60 and RF-60. Using their
new-subject · No calibration mean macro-F1 across 22 subjects, the difference
is 0.1842232623 − 0.1776465092 ≈ 0.006576753, or +0.0066 at four decimals.
Both underlying scores agree with their archived tables.

The new-subject RF replays reproduced all 88 per-subject and 20 per-fold
macro-F1 values in the recovered feature-set CSVs within 1e-12.

## Reproducing and checking the evidence

The classifier replays used Python 3.13.2, NumPy 2.2.6, SciPy 1.15.3,
scikit-learn 1.7.2 and pandas 2.3.3 on Windows. All 22 input files matched
the SHA-256 values in the [baseline data manifest](../Baseline/baseline_20261022_holdout_separated/data_manifest.json).
The replay maps the archived
[folds](../Baseline/baseline_20261022_holdout_separated/folds.json) to each
window and preserves the historical training row order. Each compressed CSV
has one row per window, identified by `source_window`, with subject,
repetition, fold, true label and predicted label. In the feature-set comparison,
`full_70`, `full_60`, `time_40` and `amplitude_20` correspond to the new-subject
files under `rf_full_feature`, `rf_full_60`, `rf_time` and `rf_amplitude`.

From the repository root, use a fresh destination for each replay.

```shell
python src/audit/replay_classifiers.py --jobs 4 --output outputs/classifier_replay
python src/preprocessing_noleakage_pietro/run_no_normalization_smoothing.py --output-dir docs/Preprocessing/smoothing_replay
python src/audit/verify_replay.py --classifiers outputs/classifier_replay --smoothing docs/Preprocessing/smoothing_replay
python src/MLP/score_predictions.py src/MLP/mlp_60 outputs/mlp_60_rescored
```

With no options, the verifier checks the committed archive. It compares the CSVs against the baseline windows and folds,
the original RF scores for all four feature sets by subject and fold, and the
published classifier log precision for macro-F1 and κ. It also checks the
recovered ensemble against its summary and the smoothing predictions against
their reports. For MLP it recomputes the six score tables and checks the historical
window identities and fold assignments. The classifier logs preserve rounded aggregates. The individual
historical classifier predictions were unavailable, so window-by-window
identity with those original predictions cannot be established.

## MLP folds and figures

All three original MLP runs used the five archived new-subject splits.
In within-subject CV, six of 110 original fold IDs differed in their assignment
of equal-sized repetitions. The corrected runner reads both modes from
`folds.json`. For a reproduction of historical scores from the saved predictions,
use the fold IDs in those CSVs. The [original score tables](mlp_legacy_scores/)
preserve the schema and the input files used in the earlier figure audit.

The neural runner enables deterministic PyTorch operations, seeds model
initialisation and batch shuffling per fold, and records the device, build and
training settings in each fresh run directory. The non-Intel-macOS pin is
PyTorch 2.15.0, the build tested during this audit. Intel macOS retains 2.2.2.
Two one-epoch MLP-20 runs on the same CPU produced byte-identical prediction
and score CSVs for all 115 folds. CUDA was unavailable on the audit machine.
Reproducibility across devices and PyTorch builds is unverified.
For report figures, use the [verified MLP plots](../MLP_verified/README.md).
The [nine original PNGs](mlp_legacy_figures/) are preserved for provenance.
