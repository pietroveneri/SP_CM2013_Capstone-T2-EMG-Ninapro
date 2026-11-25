from emg_ninapro import EMGNinaproTrack
import numpy as np

from adapter import denoise


class FilterOption01(EMGNinaproTrack):
    SUPPORTED_CFG_KEYS = EMGNinaproTrack.SUPPORTED_CFG_KEYS | frozenset({
        "filter_method",
        "impulsive",
        "baseline",
        "powerline",
        "broadband",
        "win",
        "savgol_order",
        "normalization",
    })

    PREPROCESS_DEFAULTS = {
        "filter_method": "denoise",
        "impulsive": "median",
        "baseline": None,
        "powerline": None,
        "broadband": "savgol",
        "win": 7,
        "savgol_order": 2,
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

        filter_method = cfg.get("filter_method", "denoise")
        if filter_method == "denoise":
            # causal is not passed, so adapter default causal=False applies (zero-phase / offline filtering).
            # The median and Savitzky-Golay filters use centered windows and are non-causal regardless of this flag.
            filtered_input = denoise(
                filter_input,
                impulsive=cfg.get("impulsive"),
                baseline=cfg.get("baseline"),
                powerline=cfg.get("powerline"),
                broadband=cfg.get("broadband"),
                win=cfg.get("win", 7),
                savgol_order=cfg.get("savgol_order", 2),
            )
        elif filter_method == "none":
            filtered_input = filter_input
        else:
            raise ValueError(
                "filter_method must be 'denoise' or 'none'; "
                f"got {filter_method!r}"
            )

        rec_filtered = rec.replace_epochs({
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(rec.epochs[name]).ndim >= 3 else np.asarray(values)
            for name, values in filtered_input.epochs.items()
        })

        # Step 2: normalize each subject/channel independently after filtering.
        # Z-score mean/std use the whole subject recording, including CV-held-out windows, without labels or other subjects.
        # This implies unlabeled per-subject calibration before use; it is a known, slightly generous offline assumption.
        # See the preprocess() docstring in emg_ninapro.py and RESULTS.md.
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

        return rec_filtered.replace_epochs(norm_epochs)


if __name__ == "__main__":
    t = FilterOption01()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    from sklearn.metrics import f1_score
    for name, rep in modes.items():
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("    spread:", rep["summary"])
