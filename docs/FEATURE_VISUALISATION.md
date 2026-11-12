The script loads local development recordings, extracts expanded features,
and saves figures under outputs/feature_analysis/.

Outputs
- Feature correlation hierarchy: clusters features using
  1 - abs(Pearson correlation). Reveals possible redundancy, not importance.
- New-subject LDA: fits on training subjects and projects unseen
  development subjects using the same transformation.
- Within-subject LDA: fits on training repetitions of S1 and projects
  held-out repetitions.
- Permutation importance: if enabled, measures the decrease in
  development-test macro-F1 when each feature is shuffled.

Current feature experiment
RMS is excluded, giving six features per channel and 60 columns:
MAV, waveform length, variance, relative MAV, mean frequency and median
frequency. Feature names follow the same channel-first order.

RMS removal tests possible redundancy with MAV. Its effect on predictive
performance must be checked against the full feature set using both
evaluation modes.

Interpretation
LDA is supervised: training separation is not evidence of generalisation.
Only two discriminant directions are displayed.

Correlated features may substitute for one another, reducing individual
permutation importance. Shuffle SD is not uncertainty across subjects.

Use development subjects S1–S22 only. Keep S23–S27 reserved for final
inference. Fit learned transformations on training folds only.
EOF