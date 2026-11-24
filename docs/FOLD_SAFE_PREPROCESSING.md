# DB1 signal preprocessing

Choose the filter with `--preprocessing` on the [classical experiment command](../README.md#classical-experiments).
The implementation is in [preprocessing.py](../src/preprocessing.py). Fixed filters run along
the time axis of each channel inside each 200 ms window. Feature extraction follows filtering.

| Former option | Current CLI value | Signal operation |
| --- | --- | --- |
| 2, 5 | `none` (default) | Use the DB1 envelope as supplied |
| 3 | `median-savgol` | Median length 3, then Savitzky–Golay length 5, order 2 |
| 1 | `smooth-7` | Median length 7, then Savitzky–Golay length 7, order 2 |
| 4 | `highpass-smooth-7` | Median length 7, zero-phase 0.1 Hz high-pass, then Savitzky–Golay length 7, order 2 |
| 6 | `guarded-reference` | Subtract the mean of the live channels at each sample |

Option numbers refer to the signal operations after amplitude normalization was
removed. The guarded reference estimates its live-channel mask from the training
windows in each fold. A channel qualifies when its peak amplitude is at least
5% of the median channel peak. Within-subject training uses the subject's training
repetitions. New-subject training uses the pooled training subjects.
All five CLI choices use **New-subject · No calibration**. The classifier fits
its feature scaler inside the training fold.

Centred median and Savitzky–Golay filters use later samples within the same
window. The high-pass is also zero-phase. These experiments measure offline
classification once the window is complete.

## Historical normalization experiments

Hannah's iterations 3–5 used per-subject signal statistics from the entire
recording. Their new-subject setting is **Whole session**. The within-subject
figures include test repetitions in the normalization fit. The October 28
archives contain separate fold-fitted z-score runs. Use
`python src/audit/verify_preprocessing.py` to recompute their scores from saved
predictions. The November 19 smoothing run uses `--model rf --features baseline
--preprocessing median-savgol` and applies no signal normalization. See the
[results log](RESULTS.md) and [audit commands](audit_20261119/README.md).
