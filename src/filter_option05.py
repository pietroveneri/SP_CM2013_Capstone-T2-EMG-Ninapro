from emg_ninapro import EMGNinaproTrack
import numpy as np


def peak_scale(X, guard=0.05):  # X: samples x channels, one subject
    # Wei et al. (2022) motivates peak scaling; per-subject/per-channel scope
    # and the dead-channel guard are custom adaptations for this pipeline.
    scale = np.max(np.abs(X), axis=0)
    ref = np.median(scale)
    scale = np.where(scale < guard * ref, ref, scale)
    return X / scale


# No IIR filter is applied here, so causality / the causal flag does not apply; scaling uses whole-recording statistics.
class FilterOption05(EMGNinaproTrack):
    SUPPORTED_CFG_KEYS = EMGNinaproTrack.SUPPORTED_CFG_KEYS | frozenset({"guard"})
    PREPROCESS_DEFAULTS = {"guard": 0.05}

    def __init__(self, cfg=None):
        config = dict(self.PREPROCESS_DEFAULTS)
        config.update(cfg or {})
        super().__init__(cfg=config)

    def preprocess(self, rec, cfg=None):
        cfg = self._cfg(cfg)
        guard = cfg.get("guard", 0.05)
        scaled_epochs = {}
        for name, values in rec.epochs.items():
            arr = np.asarray(values, dtype=float)
            flat = arr.reshape(-1, arr.shape[-1])
            # Peak scale uses the whole subject recording, including CV-held-out windows, without labels or other subjects.
            # This implies unlabeled per-subject calibration before use; it is a known, slightly generous offline assumption.
            # See the preprocess() docstring in emg_ninapro.py and RESULTS.md.
            scaled_epochs[name] = peak_scale(flat, guard=guard).reshape(arr.shape)
        return rec.replace_epochs(scaled_epochs)


if __name__ == "__main__":
    t = FilterOption05()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    from sklearn.metrics import f1_score
    for name, rep in modes.items():
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("    spread:", rep["summary"])
