import numpy as np
from sklearn.base import clone
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from bsp.notebook_checks import assert_no_subject_leak


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


def evaluate_modes_progress(track, records, cfg):
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
    X, y, subjects, repetitions = track._build(records, cfg)
    track._assert_no_holdout_groups(subjects.tolist())
    unique_subjects = sorted(set(subjects.tolist()))
    if len(unique_subjects) < 2:
        raise ValueError("New-subject evaluation requires at least two subjects")
    for subject in unique_subjects:
        mask = subjects == subject
        if len(np.unique(repetitions[mask])) < 2 or len(np.unique(y[mask])) < 2:
            raise ValueError(f"{subject}: within-subject evaluation requires at least two repetitions and classes")
    if len(unique_subjects) <= int(cfg.get("loso_max_groups", 12)):
        splitter = LeaveOneGroupOut()
    else:
        splitter = GroupKFold(n_splits=min(int(cfg.get("cv_max_splits", 5)), len(unique_subjects)))

    def evaluate_fold(X_train, y_train, X_test, y_test):
        selector = track.select_features(X_train, y_train, cfg)
        model = clone(track._baseline(cfg))
        model.fit(selector.transform(X_train), y_train)
        X_test = selector.transform(X_test)
        predicted = model.predict(X_test)
        if np.asarray(predicted).shape != y_test.shape:
            raise ValueError("The number of predictions does not match the test windows")
        roc = fold_roc(model, X_test, y_test, track.meta.classes)
        return predicted, roc

    yt, yp, groups, folds = [], [], [], []
    roc_folds = []
    counts = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(splitter.split(X, y, subjects)):
        assert_no_subject_leak(subjects[train], subjects[test])
        predicted, roc = evaluate_fold(X[train], y[train], X[test], y[test])
        np.add.at(counts, test, 1)
        roc_folds.append(roc)
        yt.extend(y[test]); yp.extend(predicted)
        groups.extend(subjects[test]); folds.extend([fold] * len(test))
    if not np.all(counts == 1) or len(yp) != len(y):
        raise ValueError("New-subject evaluation must predict each window exactly once")
    new_subject = track._make_report(yt, yp, groups=groups, folds=folds,
                                     n_groups=len(unique_subjects))
    new_subject["roc_folds"] = roc_folds

    yt, yp, groups, folds = [], [], [], []
    roc_folds = []
    counts = np.zeros(len(y), dtype=int)
    fold = 0
    for subject in unique_subjects:
        mask = subjects == subject
        Xs, ys, reps = X[mask], y[mask], repetitions[mask]
        indices = np.flatnonzero(mask)
        n_reps = len(set(reps.tolist()))
        for train, test in GroupKFold(n_splits=min(5, n_reps)).split(Xs, ys, reps):
            predicted, roc = evaluate_fold(Xs[train], ys[train], Xs[test], ys[test])
            np.add.at(counts, indices[test], 1)
            roc_folds.append(roc)
            yt.extend(ys[test]); yp.extend(predicted)
            groups.extend([subject] * len(test)); folds.extend([fold] * len(test))
            fold += 1
    if not np.all(counts == 1) or len(yp) != len(y):
        raise ValueError("Within-subject evaluation must predict each window exactly once")
    within = track._make_report(yt, yp, groups=groups, folds=folds,
                                split_unit="repetition (within subject)",
                                n_groups=len(unique_subjects), group_unit="subject")
    within["roc_folds"] = roc_folds
    return {"within_subject": within, "new_subject": new_subject}
