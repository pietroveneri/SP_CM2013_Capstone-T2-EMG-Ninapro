"""Evaluate the rFFT/MNF ablation and export paired subject F1 differences."""

import sys
from pathlib import Path

# Allow direct execution from the AblationTests subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AblationTests.compare_subject_f1 import run_comparison
from emg_ninapro import EMGNinaproTrack


class NoRFFTTrack(EMGNinaproTrack):
    def extract_features(self, rec, cfg=None):
        X, y, group = super().extract_features(rec, cfg=cfg)
        # Baseline order per channel: [MAV, RMS, WL, VAR, MNF].
        # Remove the rFFT-derived mean frequency: 50 -> 40 features on DB1.
        keep_columns = [i for i in range(X.shape[1]) if i % 5 != 4]
        return X[:, keep_columns], y, group


def main():
    run_comparison(NoRFFTTrack(), "norFFT")


if __name__ == "__main__":
    main()
