"""Signal operations on DB1 windows shaped (windows, samples, channels)."""
from dataclasses import dataclass

import numpy as np

from adapter import denoise

PREPROCESSING = ("none", "median-savgol", "smooth-7", "highpass-smooth-7", "guarded-reference")


@dataclass
class GuardedAverageReference:
    guard: float = 0.05

    def fit(self, epochs):
        if not 0 <= self.guard <= 1:
            raise ValueError("guard must be between 0 and 1")
        x = np.asarray(epochs, dtype=float)
        if x.ndim != 3 or not len(x):
            raise ValueError("expected nonempty (windows, samples, channels)")
        flat = x.reshape(-1, x.shape[-1])
        maxima = np.max(np.abs(flat), axis=0)
        ref = np.median(maxima)
        self.live_ = maxima >= self.guard * ref
        return self

    def transform(self, epochs):
        x = np.asarray(epochs, dtype=float)
        return x - x[..., self.live_].mean(axis=-1, keepdims=True)


def filter_windows(rec, recipe):
    """Apply a fixed filter along time, independently for each window and channel."""
    if recipe == "none":
        return rec
    # denoise expects time last, while DB1 stores channels last.
    oriented = rec.replace_epochs({"emg": rec.epochs["emg"].swapaxes(1, 2)})
    if recipe == "median-savgol":
        filtered = denoise(oriented, impulsive="median", win=3)
        filtered = denoise(filtered, broadband="savgol", win=5, savgol_order=2)
    elif recipe in ("smooth-7", "highpass-smooth-7"):
        filtered = denoise(oriented, impulsive="median",
                           baseline="highpass" if recipe == "highpass-smooth-7" else None,
                           broadband="savgol", hp_fc=0.1, win=7, savgol_order=2, causal=False)
    else:
        raise ValueError(f"Unknown fixed window filter {recipe!r}")
    return rec.replace_epochs({"emg": filtered.epochs["emg"].swapaxes(1, 2)})
