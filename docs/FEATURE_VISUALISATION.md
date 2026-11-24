The script loads local development recordings, extracts expanded features,
and saves figures under `outputs/feature_analysis/`. The [audit figures and
feature ranking](audit_20261119/feature_figures/) use all 22 development subjects.

## Outputs

- Feature correlation hierarchy: clusters features using
  1 - abs(Pearson correlation). Reveals possible redundancy, not importance.
- New-subject LDA: fits on training subjects and projects unseen
  development subjects using the same transformation.
- Within-subject LDA: fits on training repetitions of S1 and projects
  held-out repetitions.
- Permutation importance: measures the decrease in
  development-test macro-F1 when each feature is shuffled.

## Feature set

The visualiser uses the 70-column expanded matrix. For each of the ten
channels the extractor emits MAV, waveform length, variance, relative MAV,
mean frequency, median frequency and RMS, in that order. The hierarchy
omits columns that are constant in its training fold.

The figures, full permutation ranking and split metadata use the suffix
`_expanded_70`. The metadata records the ordered column names. The
`full_60` comparison in `compare_feature_sets.py` drops the ten RMS columns.

## Interpretation

LDA is supervised: training separation is not evidence of generalisation.
Only two discriminant directions are displayed.

Correlated features may substitute for one another, reducing individual
permutation importance. Shuffle SD is not uncertainty across subjects.

Use development subjects S1–S22 only. Keep S23–S27 reserved for final
inference. Fit learned transformations on training folds only.
