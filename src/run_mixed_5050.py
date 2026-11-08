"""One fixed 50/50 RF-time and LDA-baseline development experiment."""
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from random_forest.run_rf import RFTrack, CFG as RF_CFG
from lda_regularized.run_lda_regularized import LDATrack, CFG as LDA_CFG

ROOT = Path(__file__).resolve().parents[1]


def main():
    rf_track, lda_track = RFTrack("time"), LDATrack("baseline")
    records = rf_track.load(ROOT / "NinaPro_Mat", exercise="E1")
    rf_track.assert_no_holdout(records)
    if {r.group for r in records} != {f"S{i}" for i in range(1, 23)} or len(records) != 22:
        raise ValueError("Expected exactly the 22 development recordings")
    xr, yr, subjects, repetitions = rf_track._build(records, RF_CFG)
    xl, yl, subjects_l, repetitions_l = lda_track._build(records, LDA_CFG)
    for a, b in ((yr, yl), (subjects, subjects_l), (repetitions, repetitions_l)):
        if not np.array_equal(a, b):
            raise ValueError("Feature pipelines disagree on window metadata")
    if len(xr) != len(xl) or xr.shape[1] != 40 or xl.shape[1] != 50:
        raise ValueError("Unexpected window count or feature dimensions")
    labels = np.asarray(rf_track.meta.classes)

    def run_fold(train, test):
        models = []
        for track, cfg, X in ((rf_track, RF_CFG, xr), (lda_track, LDA_CFG, xl)):
            selector = track.select_features(X[train], yr[train], cfg)
            model = clone(track._baseline(cfg)).fit(selector.transform(X[train]), yr[train])
            if set(model.classes_) != set(labels):
                raise ValueError("Training fold does not contain all expected classes")
            probabilities = model.predict_proba(selector.transform(X[test]))
            columns = [np.flatnonzero(model.classes_ == label)[0] for label in labels]
            models.append(probabilities[:, columns])
        return labels[np.argmax((models[0] + models[1]) / 2, axis=1)]

    output = {}
    for mode in ("new_subject", "within_subject"):
        predictions = np.empty(len(yr), dtype=object)
        fold_ids = np.full(len(yr), -1, dtype=int)
        splits = []
        if mode == "new_subject":
            splits = list(GroupKFold(n_splits=5).split(xr, yr, subjects))
        else:
            for subject in sorted(set(subjects)):
                indices = np.flatnonzero(subjects == subject)
                reps = repetitions[indices]
                for train, test in GroupKFold(n_splits=min(5, len(set(reps)))).split(xr[indices], yr[indices], reps):
                    splits.append((indices[train], indices[test]))
        for fold, (train, test) in enumerate(splits):
            grouping = subjects if mode == "new_subject" else repetitions
            if set(grouping[train]) & set(grouping[test]) or np.any(fold_ids[test] != -1):
                raise ValueError("Overlapping train/test groups or repeated test window")
            predictions[test] = run_fold(train, test)
            fold_ids[test] = fold
            print(f"{mode}: fold {fold + 1}/{len(splits)}", flush=True)
        if np.any(fold_ids < 0):
            raise ValueError("Some windows were not evaluated")
        report = rf_track._make_report(yr, predictions, groups=subjects, folds=fold_ids,
                                       n_groups=22, **({"split_unit": "repetition (within subject)", "group_unit": "subject"} if mode == "within_subject" else {}))
        output[mode] = {"summary": report["summary"], "macro_f1": report["macro_f1"]}
        target = ROOT / "outputs" / "mixed_5050"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f"{mode}_predictions.npz", y=yr, prediction=predictions.astype(str), subject=subjects, repetition=repetitions, fold=fold_ids)
    (target / "summary.json").write_text(json.dumps(output, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()), encoding="utf-8")
    print(json.dumps(output, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()))


if __name__ == "__main__":
    main()
