from emg_ninapro import EMGNinaproTrack
import numpy as np


def average_reference(X, guard=0.05):
    channel_max = np.max(np.abs(X), axis=0)
    ref = np.median(channel_max)
    live = channel_max >= guard * ref
    baseline = np.mean(X[:, live], axis=1, keepdims=True)
    return X - baseline


def peak_scale(X, guard=0.05):  # X: samples x channels, one subject
    # Wei et al. (2022) motivates peak scaling. Per-subject/per-channel scope,
    # the guarded average reference, and combining both steps are adaptations.
    scale = np.max(np.abs(X), axis=0)
    ref = np.median(scale)
    scale = np.where(scale < guard * ref, ref, scale)
    return X / scale


class FilterTrack(EMGNinaproTrack):
    SUPPORTED_CFG_KEYS = EMGNinaproTrack.SUPPORTED_CFG_KEYS | frozenset({"guard"})
    PREPROCESS_DEFAULTS = {"guard": 0.05}

    def __init__(self, cfg=None):
        config = dict(self.PREPROCESS_DEFAULTS)
        config.update(cfg or {})
        super().__init__(cfg=config)

    def preprocess(self, rec, cfg=None):
        cfg = self._cfg(cfg)
        guard = cfg.get("guard", 0.05)
        processed_epochs = {}
        for name, values in rec.epochs.items():
            arr = np.asarray(values, dtype=float)
            flat = arr.reshape(-1, arr.shape[-1])
            # Kok et al. (2024), GRABMyo: average referencing before scaling.
            # Only channels above the guard threshold contribute to the reference.
            referenced = average_reference(flat, guard=guard)
            processed_epochs[name] = peak_scale(referenced, guard=guard).reshape(arr.shape)
        return rec.replace_epochs(processed_epochs)


if __name__ == "__main__":
    t = FilterTrack()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    from sklearn.metrics import f1_score
    for name, rep in modes.items():
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("    spread:", rep["summary"])
