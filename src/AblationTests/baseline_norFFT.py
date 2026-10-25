"""Baseline rFFT ablation: retain MAV, RMS, WL and variance (no MNF)."""

import sys
from pathlib import Path

# Allow direct execution from the AblationTests subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emg_ninapro import EMGNinaproTrack


class NoRFFTTrack(EMGNinaproTrack):
    def extract_features(self, rec, cfg=None):
        X, y, group = super().extract_features(rec, cfg=cfg)
        # Baseline order per channel: [MAV, RMS, WL, VAR, MNF].
        # Remove the rFFT-derived mean frequency: 50 -> 40 features on DB1.
        keep_columns = [i for i in range(X.shape[1]) if i % 5 != 4]
        return X[:, keep_columns], y, group


def main():
    # Same configuration, data and evaluation modes as run_baseline.py.
    cfg = {
        "seed": 0,
        "select": "none",
        "imbalance": "balanced",
        "cv_max_splits": 5,
        "loso_max_groups": 12,
    }

    root = Path(__file__).resolve().parents[2]
    track = NoRFFTTrack()
    track.download(root / "NinaPro_Mat", subset=range(1, 23))
    records = track.load(root / "NinaPro_Mat", exercise="E1")
    track.assert_no_holdout(records)
    if not records:
        raise RuntimeError("No recordings found. Check the data path.")

    results = track.evaluate_modes(records, cfg=cfg)

    for mode, result in results.items():
        print("Baseline without rFFT mean frequency:", mode, result["summary"])
        track.report(result)


if __name__ == "__main__":
    main()
