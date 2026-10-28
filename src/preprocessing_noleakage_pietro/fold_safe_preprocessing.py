"""Hannah's fixed signal operations without signal normalization.

The filters act independently per window. Option 6 retains guarded average
referencing; its live-channel mask is learned from training windows only.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from adapter import denoise
from emg_ninapro import EMGNinaproTrack


@dataclass
class GuardedAverageReference:
    guard: float = 0.05

    def fit(self, epochs):
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


class FoldSafePreprocessingTrack(EMGNinaproTrack):
    """Evaluate options 1–6 without signal normalization.

    Only ``evaluate_modes`` implements this protocol. Inherited feature-matrix
    and final-model APIs are disabled because they cannot fit signal-level
    reference mask within folds or carry the fitted mask into inference.
    """
    def __init__(self, option=1, guard=0.05, cfg=None):
        if option not in range(1, 7):
            raise ValueError("option must be 1 through 6")
        if not 0 <= guard <= 1:
            raise ValueError("guard must be between 0 and 1")
        super().__init__(cfg=cfg)
        self.option, self.guard = option, guard

    def preprocess(self, rec, cfg=None):
        """Apply only the fixed per-window filter from the original option."""
        if self.option in (2, 5, 6):
            return rec
        oriented = rec.replace_epochs({
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(values).ndim >= 3 else np.asarray(values)
            for name, values in rec.epochs.items()
        })
        if self.option == 3:
            filtered = denoise(oriented, impulsive="median", win=3)
            filtered = denoise(filtered, broadband="savgol", win=5, savgol_order=2)
        else:
            filtered = denoise(oriented, impulsive="median",
                               baseline="highpass" if self.option == 4 else None,
                               broadband="savgol", hp_fc=0.1, win=7,
                               savgol_order=2, causal=False)
        return rec.replace_epochs({
            name: np.swapaxes(np.asarray(values), -1, -2)
            if np.asarray(rec.epochs[name]).ndim >= 3 else np.asarray(values)
            for name, values in filtered.epochs.items()
        })

    def _fold_features(self, windows, labels, fs, tr, te):
        from adapter import Recording
        train_windows, test_windows = windows[tr], windows[te]
        if self.option == 6:
            reference = GuardedAverageReference(self.guard).fit(train_windows)
            train_windows = reference.transform(train_windows)
            test_windows = reference.transform(test_windows)
        train = Recording("train", fs, {"emg": train_windows}, labels[tr])
        test = Recording("test", fs, {"emg": test_windows}, labels[te])
        return self.extract_features(train)[0], self.extract_features(test)[0]

    def evaluate_modes(self, recs, cfg=None):
        self.assert_no_holdout(recs)
        cfg = self._cfg(cfg)
        filtered = [self.preprocess(r, cfg) for r in recs]
        if not filtered:
            raise ValueError("at least one recording is required")
        fs = filtered[0].fs
        if any(r.fs != fs for r in filtered):
            raise ValueError("all recordings must have the same sampling rate")
        windows = np.concatenate([r.epochs["emg"] for r in filtered])
        labels = np.concatenate([r.labels for r in filtered])
        groups = np.concatenate([np.repeat(r.group, len(r.labels)) for r in filtered])
        reps = np.concatenate([np.asarray(r.meta["rep"]) for r in filtered])

        def evaluate_folds(folds, split_unit, group_unit):
            yt, yp, gg, ff = [], [], [], []
            for k, (tr, te) in enumerate(folds):
                xtr, xte = self._fold_features(windows, labels, fs, tr, te)
                selector = self.select_features(xtr, labels[tr], cfg)
                model = clone(self._baseline(cfg))
                model.fit(selector.transform(xtr), labels[tr])
                yt.extend(labels[te]); yp.extend(model.predict(selector.transform(xte)))
                gg.extend(groups[te]); ff.extend([k] * len(te))
            return self._make_report(yt, yp, groups=gg, folds=ff,
                                     split_unit=split_unit,
                                     n_groups=len(set(groups.tolist())), group_unit=group_unit)

        n_groups = len(set(groups.tolist()))
        if n_groups < 2:
            raise ValueError("new-subject evaluation needs at least two subjects")
        if n_groups <= int(cfg.get("loso_max_groups", 12)):
            new_folds = LeaveOneGroupOut().split(windows, labels, groups)
        else:
            new_folds = GroupKFold(n_splits=min(int(cfg.get("cv_max_splits", 5)), n_groups)).split(windows, labels, groups)

        def within_folds():
            for subject in sorted(set(groups.tolist())):
                ids = np.flatnonzero(groups == subject)
                rs, ys = reps[ids], labels[ids]
                n_rep = len(set(rs.tolist()))
                if n_rep < 2 or len(set(ys.tolist())) < 2:
                    continue
                for tr, te in GroupKFold(n_splits=min(5, n_rep)).split(ids, ys, rs):
                    yield ids[tr], ids[te]

        return {
            "within_subject": evaluate_folds(within_folds(), "repetition (within subject)", "subject"),
            "new_subject": evaluate_folds(new_folds, "subject", "subject"),
        }

    def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("Use evaluate_modes; this API cannot preserve the fold-fitted reference mask")

    _build = build_dataset = evaluate = train_baseline = infer = write_submission = _unsupported
