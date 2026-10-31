from adapter import Recording
from emg_ninapro import FS
import numpy as np

# This produces 7 features × 10 channels = 70 features. 
# The power scaling cancels in the mean and median frequency calculations, so an absolute PSD calibration is unnecessary here.
# TJ 18/10
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

#rms removed from features because it is highly correlated with MAV and adds little information. It is still computed for completeness, but not returned. TJ 23/10
# return rms for feature importance analysis, but not for the reduced feature sets? TJ 29/10
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


# Applies the feature function to every window. It should preserve labels and the subject identifier TJ 18/10
def extract_window_features(rec: Recording, cfg=None):
        """Module 3 — per-window, per-channel time-domain descriptors (MAV, RMS,
        waveform length, variance) plus a mean-frequency term. Rectification and
        envelope smoothing, per-channel normalisation, and a wider window are all
        candidate improvements that belong partly in `preprocess()` — the window
        length in particular trades responsiveness against feature stability."""
        windows = rec.epochs["emg"]

        X = np.vstack([
            _expanded_win_features(window, fs=rec.fs)
            for window in windows])

        y = np.asarray(rec.labels)

        return X, y, rec.group
    ## Result has the following structure:

        # | Output      | Shape/content |

        # | `X`         | Number of windows × 70 features |
        # | `y`         | One gesture label per window |
        # | `rec.group` | Subject identifier, such as `"S1"` |

FEATURE_TYPES = (
    "MAV",
    "WL",
    "VAR",
    "relative_MAV",
    "mean_frequency",
    "median_frequency",
    "RMS",
)


def expanded_feature_names(n_channels=10):
    # retained_features = [
    #     feature for feature in FEATURE_TYPES 
    #     if feature != "RMS"
    # ]  # helps reduce feature correlation and dimensionality, while retaining the most informative features. TJ 23/10

    return [
        f"ch{channel:02d}_{feature}"
        for channel in range(1, n_channels + 1)
                # for feature in retained_features
        for feature in FEATURE_TYPES ]
