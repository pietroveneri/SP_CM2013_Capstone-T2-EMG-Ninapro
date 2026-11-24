import numpy as np
from sklearn.base import clone
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from adapter import OverSampled, Recording, SMOTEd
from development_folds import development_folds
from preprocessing import GuardedAverageReference, filter_windows
from bsp.notebook_checks import assert_no_subject_leak


def configure_imbalance(estimator, cfg):
    """Apply a supported training-fold sampler to a classifier pipeline."""
    mode = cfg["imbalance"]
    if mode == "none":
        return estimator
    if mode == "resample":
        return OverSampled(estimator=estimator, random_state=cfg["seed"])
    if mode in ("smote", "adasyn"):
        return SMOTEd(estimator=estimator, random_state=cfg["seed"],
                      k_neighbors=cfg["smote_k"], sampler=mode)
    raise ValueError(f"{estimator.steps[-1][0]}: unsupported imbalance={mode!r}. "
                     "Available sampling modes: none, resample, smote, adasyn.")


def fold_roc(model, X_test, y_test, labels):
    if set(model.classes_) != set(labels) or set(y_test) != set(labels):
        return None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
    else:
        scores = model.predict_proba(X_test)
    grid = np.linspace(0, 1, 101)
    curves, areas = [], []
    for label in labels:
        column = np.flatnonzero(model.classes_ == label)[0]
        fpr, tpr, _ = roc_curve(y_test == label, scores[:, column])
        curves.append(np.interp(grid, fpr, tpr))
        areas.append(auc(fpr, tpr))
    return {"tpr": np.mean(curves, axis=0), "auc": float(np.mean(areas))}


def evaluate_modes_progress(track, records, cfg, *, preprocessing="none", archived_folds=False):
    track.assert_no_holdout(records)
    if not records:
        raise ValueError("No recordings to evaluate")
    for record in records:
        if "rep" not in record.meta or record.meta["rep"] is None:
            raise ValueError(f"{record.group}: missing repetition metadata")
        reps = np.asarray(record.meta["rep"])
        if reps.ndim != 1 or len(reps) != len(record.labels) or not len(reps):
            raise ValueError(f"{record.group}: repetition metadata must match the windows")
        if any(rep is None or str(rep).strip() == "" for rep in reps):
            raise ValueError(f"{record.group}: invalid repetition metadata")
        if np.issubdtype(reps.dtype, np.number) and not np.isfinite(reps).all():
            raise ValueError(f"{record.group}: invalid repetition metadata")
    cfg = track._cfg(cfg)
    if preprocessing == "guarded-reference":
        fs = records[0].fs
        if any(r.fs != fs for r in records):
            raise ValueError("Expected the same sampling rate for all subjects")
        X = np.concatenate([r.epochs["emg"] for r in records])
        y = np.concatenate([r.labels for r in records])
        subjects = np.concatenate([np.repeat(r.group, len(r.labels)) for r in records])
        repetitions = np.concatenate([r.meta["rep"] for r in records])
    else:
        records = [filter_windows(r, preprocessing) for r in records]
        X, y, subjects, repetitions = track._build(records, cfg)
    track._assert_no_holdout_groups(subjects.tolist())
    unique_subjects = sorted(set(subjects.tolist()))
    if len(unique_subjects) < 2:
        raise ValueError("New-subject evaluation requires at least two subjects")
    for subject in unique_subjects:
        mask = subjects == subject
        if len(np.unique(repetitions[mask])) < 2 or len(np.unique(y[mask])) < 2:
            raise ValueError(f"{subject}: within-subject evaluation requires at least two repetitions and classes")
    if archived_folds:
        splits = development_folds(subjects, repetitions)
    else:
        if len(unique_subjects) <= cfg["loso_max_groups"]:
            splitter = LeaveOneGroupOut()
        else:
            splitter = GroupKFold(n_splits=min(cfg["cv_max_splits"], len(unique_subjects)))
        within = []
        for subject in unique_subjects:
            ids = np.flatnonzero(subjects == subject)
            reps = repetitions[ids]
            cv = GroupKFold(n_splits=min(5, len(np.unique(reps))))
            within.extend((ids[tr], ids[te]) for tr, te in cv.split(ids, y[ids], reps))
        splits = {"new_subject": splitter.split(X, y, subjects), "within_subject": within}

    reports = {}
    for mode, folds in splits.items():
        test_rows, predictions, fold_ids, rocs = [], [], [], []
        for fold, (train, test) in enumerate(folds):
            if mode == "new_subject":
                assert_no_subject_leak(subjects[train], subjects[test])
            X_train, X_test = X[train], X[test]
            if preprocessing == "guarded-reference":
                reference = GuardedAverageReference().fit(X_train)
                def referenced_features(windows, labels):
                    rec = Recording("fold", fs, {"emg": reference.transform(windows)}, labels)
                    return track.extract_features(rec, cfg)[0]
                X_train = referenced_features(X_train, y[train])
                X_test = referenced_features(X_test, y[test])
            selector = track.select_features(X_train, y[train], cfg)
            model = clone(track._baseline(cfg))
            model.fit(selector.transform(X_train), y[train])
            X_test = selector.transform(X_test)
            predicted = model.predict(X_test)
            if predicted.shape != y[test].shape:
                raise ValueError("The number of predictions does not match the test windows")
            test_rows.append(test)
            predictions.append(predicted)
            fold_ids.append(np.full(len(test), fold))
            rocs.append(fold_roc(model, X_test, y[test], track.meta.classes))

        ids = np.concatenate(test_rows)
        if not np.array_equal(np.sort(ids), np.arange(len(y))):
            name = "New-subject" if mode == "new_subject" else "Within-subject"
            raise ValueError(f"{name} evaluation must predict each window exactly once")
        report = track._make_report(
            y[ids], np.concatenate(predictions), groups=subjects[ids], folds=np.concatenate(fold_ids),
            split_unit="subject" if mode == "new_subject" else "repetition (within subject)",
            group_unit="subject", n_groups=len(unique_subjects),
        )
        report.update(roc_folds=rocs, source_window=ids, repetition=repetitions[ids])
        reports[mode] = report
    return reports
