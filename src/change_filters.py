from emg_ninapro import EMGNinaproTrack
import numpy as np

# Add or remove optional adapter helper imports here.
from adapter import bandpass_notch, denoise


class FilterTrack(EMGNinaproTrack):
    SUPPORTED_CFG_KEYS = EMGNinaproTrack.SUPPORTED_CFG_KEYS | frozenset({
        "filter_method",
        "impulsive",
        "baseline",
        "powerline",
        "broadband",
        "hp_fc",
        "f0",
        "q",
        "win",
        "causal",
        "order",
        "band",
        "notch",
        "notch_q",
        "normalization",
    })
    # Add or remove custom config keys here.

    # Tune your default parameters and hyperparameters here.
    PREPROCESS_DEFAULTS = {
        "filter_method": "denoise",  # Choose "denoise", "bandpass_notch", or "none".
        "impulsive": "median",       # Example filter choice.
        "win": 3,                    # Example median-filter window in samples.
        "baseline": None,
        "powerline": None,
        "broadband": None,
        "hp_fc": 0.5,                # Example high-pass corner if baseline is enabled.
        "f0": 50.0,
        "q": 30.0,
        "causal": False,
        "order": 4,
        "band": None,
        "notch": None,
        "notch_q": 30.0,
        "normalization": "zscore",
    }

    def __init__(self, cfg=None):
        config = dict(self.PREPROCESS_DEFAULTS)
        config.update(cfg or {})
        super().__init__(cfg=config)

    def preprocess(self, rec, cfg=None):
        cfg = self._cfg(cfg)

        # Step 1: filter the signal; adapter filters expect time on the last axis.
        # Ninapro EMG epochs are (epochs, samples, channels), so orient them first.
        filter_epochs = {
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(values).ndim >= 3 else np.asarray(values)
            for name, values in rec.epochs.items()
        }
        filter_input = rec.replace_epochs(filter_epochs)

        # Insert or swap your preprocessing filter logic here.
        filter_method = cfg.get("filter_method", "denoise")
        if filter_method == "denoise":
            filtered_input = denoise(
                filter_input,
                impulsive=cfg.get("impulsive"),
                baseline=cfg.get("baseline"),
                powerline=cfg.get("powerline"),
                broadband=cfg.get("broadband"),
                hp_fc=cfg.get("hp_fc", 0.5),
                f0=cfg.get("f0", 50.0),
                q=cfg.get("q", 30.0),
                win=cfg.get("win", 3),
                causal=cfg.get("causal", False),
                order=cfg.get("order", 4),
            )
        elif filter_method == "bandpass_notch":
            filtered_input = bandpass_notch(
                filter_input,
                band=cfg.get("band"),
                notch=cfg.get("notch"),
                order=cfg.get("order", 4),
                notch_q=cfg.get("notch_q", 30.0),
                causal=cfg.get("causal", False),
            )
        elif filter_method == "none":
            filtered_input = filter_input
        else:
            raise ValueError(
                "filter_method must be 'denoise', 'bandpass_notch', or 'none'; "
                f"got {filter_method!r}"
            )

        rec_filtered = rec.replace_epochs({
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(rec.epochs[name]).ndim >= 3 else np.asarray(values)
            for name, values in filtered_input.epochs.items()
        })

        # Step 2: normalize each subject/channel independently after filtering.
        # Solid per-subject/per-channel Z-score normalization block (independent of filter choices).
        normalization = cfg.get("normalization", "zscore")
        norm_epochs = {}
        for name, values in rec_filtered.epochs.items():
            arr = np.asarray(values, dtype=float)
            if normalization == "zscore":
                mean = np.mean(arr, axis=(0, 1), keepdims=True)
                std = np.std(arr, axis=(0, 1), keepdims=True) + 1e-8
                norm_epochs[name] = (arr - mean) / std
            elif normalization == "none":
                norm_epochs[name] = arr
            else:
                raise ValueError(
                    "normalization must be 'zscore' or 'none'; "
                    f"got {normalization!r}"
                )

        # Step 3: package the normalized epochs into a new recording.
        return rec_filtered.replace_epochs(norm_epochs)


# External runner wiring:
# src/run_baseline.py now imports and instantiates FilterTrack, so its records
# flow through this class's filtering and normalization before feature extraction.
# The root visualize_emg.py is a standalone raw-MAT preview and does not import
# EMGNinaproTrack; it is not part of the track preprocessing pipeline.
# To run the pure parent baseline instead, restore the import and constructor in
# src/run_baseline.py to `from emg_ninapro import EMGNinaproTrack` and
# `track = EMGNinaproTrack()`.