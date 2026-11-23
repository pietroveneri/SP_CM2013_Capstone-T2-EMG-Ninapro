"""DB1 per-window features and the column order used by all expanded feature sets."""
import numpy as np

from adapter import Recording
from emg_ninapro import EMGNinaproTrack, FS


def _expanded_win_features(win, fs=FS):
    """Features from a window shaped [samples, channels]."""
    win = np.asarray(win, dtype=float)
    eps = 1e-12

    # Compute these together so relative MAV uses all channels.
    mav_all = np.mean(np.abs(win), axis=0)
    relative_mav = mav_all / max(mav_all.sum(), eps)

    features = []

    for ch in range(win.shape[1]):
        x = win[:, ch]

        # Time-domain features
        mav = mav_all[ch]
        rms = np.sqrt(np.mean(x ** 2))
        waveform_length = np.sum(np.abs(np.diff(x)))
        variance = np.var(x)

        # Remove the mean so DC amplitude does not dominate.
        centred = x - x.mean()

        # Windowing reduces spectral leakage.
        taper = np.hanning(len(x))
        spectrum = np.fft.rfft(centred * taper)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(len(x), d=1 / fs)

        # Exclude DC explicitly.
        power[0] = 0
        total_power = power.sum()

        if total_power > eps:
            mean_frequency = (
                np.sum(frequencies * power) / total_power
            )

            cumulative_power = np.cumsum(power)
            median_index = np.searchsorted(
                cumulative_power, 0.5 * total_power
            )
            median_frequency = frequencies[median_index]
        else:
            mean_frequency = 0.0
            median_frequency = 0.0

        features.extend([
            mav,
            waveform_length,
            variance,
            relative_mav[ch],
            mean_frequency,
            median_frequency,
            rms,
        ])

    return np.asarray(features)


def extract_window_features(rec: Recording, cfg=None):
    """Return one expanded feature row per window, its gesture and subject."""
    X = np.vstack([_expanded_win_features(window, fs=rec.fs) for window in rec.epochs["emg"]])
    return X, np.asarray(rec.labels), rec.group


FEATURE_TYPES = (
    "MAV",
    "WL",
    "VAR",
    "relative_MAV",
    "mean_frequency",
    "median_frequency",
    "RMS",
)


FEATURE_SETS = {
    "full_70": FEATURE_TYPES,
    "full_60": FEATURE_TYPES[:-1],  # omit RMS
    "time_40": ("MAV", "WL", "VAR", "relative_MAV"),
    "amplitude_20": ("MAV", "relative_MAV"),
}


def expanded_feature_names(n_channels=10):
    return [f"ch{channel:02d}_{feature}" for channel in range(1, n_channels + 1)
            for feature in FEATURE_TYPES]


def feature_columns(feature_set, n_channels=10):
    families = FEATURE_SETS[feature_set]
    return [i for i, name in enumerate(expanded_feature_names(n_channels))
            if name.split("_", 1)[1] in families]


class FeatureTrack(EMGNinaproTrack):
    """Choose the supplied 50-feature extractor or columns of the expanded matrix."""
    def __init__(self, feature_set="full_70", cfg=None):
        super().__init__(cfg=cfg)
        self.feature_set = feature_set
        if feature_set != "baseline" and feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set {feature_set!r}")

    def extract_features(self, rec, cfg=None):
        if self.feature_set == "baseline":
            return super().extract_features(rec, cfg)
        X, y, subject = extract_window_features(rec, cfg)
        columns = feature_columns(self.feature_set, rec.epochs["emg"].shape[-1])
        return X[:, columns], y, subject
