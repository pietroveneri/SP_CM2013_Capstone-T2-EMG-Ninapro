# Signal preprocessing without normalization

`src/fold_safe_preprocessing.py` retains the default fixed signal operations
from `origin/experiment/hannah`, but no option applies z-score, peak scaling,
or any other normalization to the signal.

| Option | Signal operation |
|---|---|
| 1 | Median and Savitzky–Golay smoothing, window 7 |
| 2 | Identity |
| 3 | Median window 3, then Savitzky–Golay window 5, order 2 |
| 4 | Median, 0.1 Hz high-pass and Savitzky–Golay, window 7 |
| 5 | Identity |
| 6 | Guarded average reference, without peak scaling |

Option 6 estimates its live-channel mask from the training windows only, then
subtracts the mean of those channels sample by sample from both train and test.
For within-subject evaluation the mask uses only the subject's training
repetitions; for new-subject evaluation it uses pooled training subjects. The
reference is a signal operation, not a fitted amplitude normalization; no test
window participates in selecting channels. Other options do not fit any signal
preprocessing parameters. The model's feature-level `StandardScaler` remains
unchanged and is fitted inside each training fold.

The smoothing filters operate independently within each window; centered median
and Savitzky–Golay filters are non-causal within the window. Option 4's
high-pass is zero-phase and operates on short windows, so it should not be
interpreted as a validated real-time drift-removal procedure.

Use `FoldSafePreprocessingTrack(option=...).evaluate_modes(records, cfg=...)`
for grouped evaluation. The inherited dataset-building, evaluation and final
inference APIs remain disabled because option 6 needs its learned mask preserved.
See `tests/test_fold_safe_preprocessing.py` for the split, held-out perturbation, signal operation and no-normalization checks.
