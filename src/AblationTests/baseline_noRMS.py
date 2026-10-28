"""Evaluate the RMS ablation and export paired subject F1 differences from baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AblationTests.compare_subject_f1 import run_comparison
from emg_ninapro import EMGNinaproTrack


class NoRMSTrack(EMGNinaproTrack):
    def extract_features(self, rec, cfg=None):
        X, y, group = super().extract_features(rec, cfg=cfg)
        # Baseline order per channel: [MAV, RMS, WL, VAR, MNF].
        # Remove RMS for every channel: 50 -> 40 features on DB1.
        keep_columns = [i for i in range(X.shape[1]) if i % 5 != 1]
        return X[:, keep_columns], y, group


def main():
    run_comparison(NoRMSTrack(), "noRMS")


if __name__ == "__main__":
    main()
