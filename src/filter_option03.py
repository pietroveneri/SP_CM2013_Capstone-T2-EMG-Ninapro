from emg_ninapro import EMGNinaproTrack
import numpy as np

from adapter import denoise


class FilterOption03(EMGNinaproTrack):
    """Apply the fixed per-window filters from the fold-safe option 3."""

    def preprocess(self, rec, cfg=None):
        oriented = rec.replace_epochs({
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(values).ndim >= 3 else np.asarray(values)
            for name, values in rec.epochs.items()
        })
        filtered = denoise(oriented, impulsive="median", win=3)
        filtered = denoise(filtered, broadband="savgol", win=5, savgol_order=2)
        return rec.replace_epochs({
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(rec.epochs[name]).ndim >= 3 else np.asarray(values)
            for name, values in filtered.epochs.items()
        })


if __name__ == "__main__":
    t = FilterOption03()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    from sklearn.metrics import f1_score
    for name, rep in modes.items():
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("    spread:", rep["summary"])
