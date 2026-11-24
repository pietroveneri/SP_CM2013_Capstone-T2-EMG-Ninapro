"""Check indexed development predictions against the archived folds and reports."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from MLP.score_predictions import score_predictions
from development_folds import BASELINE

ARCHIVE = SRC.parent / "docs/audit_20261119"
LOGS = {"rf": "random_forest/rf_20261101", "lda": "lda_regularized/lda_20261101",
        "svm": "svm_rbf/svm_20261101", "knn": "knn/knn_20261101_5nn"}
FEATURE_SETS = {"rf_full_feature": "full_70", "rf_full_60": "full_60",
                "rf_time": "time_40", "rf_amplitude": "amplitude_20"}
IDENTITY = ["source_window", "subject", "repetition", "fold", "y_true"]


def score(frame):
    return {"macro_f1": f1_score(frame.y_true, frame.y_pred, average="macro"),
            "kappa": cohen_kappa_score(frame.y_true, frame.y_pred)}


def summarize(frame, mode, name):
    subject = pd.DataFrame({s: score(g) for s, g in frame.groupby("subject")}).T
    folds = pd.Series({f: f1_score(g.y_true, g.y_pred, average="macro")
                       for f, g in frame.groupby("fold")})
    pooled = score(frame)
    s = subject.macro_f1
    summary = {"run": name, "mode": mode, "setting": "No calibration" if mode == "new_subject" else "",
               "windows": len(frame), "subjects": len(s), "folds": len(folds),
               "subject_f1_mean": s.mean(), "subject_f1_sd": s.std(),
               "subject_f1_min": s.min(), "subject_f1_max": s.max(),
               "fold_f1_mean": folds.mean(), "fold_f1_sd": folds.std(),
               "pooled_f1": pooled["macro_f1"], "subject_kappa_mean": subject.kappa.mean(),
               "pooled_kappa": pooled["kappa"]}
    return summary, subject.macro_f1, folds


def check_log(log, domain, mode, s):
    panel = log.split(f"### {domain} - {mode} (", 1)[1].split("### ", 1)[0]
    expected = (f"mean macro_f1 {s['subject_f1_mean']:.3f} (sd {s['subject_f1_sd']:.3f}, "
                f"range {s['subject_f1_min']:.3f}-{s['subject_f1_max']:.3f} across 22 subjects); "
                f"pooled {s['pooled_f1']:.3f}")
    kappa = (f"cohens_kappa: pooled {s['pooled_kappa']:.3f}, "
             f"mean over subjects {s['subject_kappa_mean']:.3f}")
    if expected not in panel or kappa not in panel:
        raise ValueError(f"{domain}/{mode}: replay differs from historical log: {expected}; {kappa}")


def verify_mlp(reference, mode):
    # The historical MLP and Windows baseline enumerate subjects in different orders.
    keys = ["subject", "subject_window"]
    reference = reference.assign(subject_window=reference.groupby("subject").cumcount()).set_index(keys)
    for size in (20, 40, 60):
        path = SRC.parent / "docs/MLP_verified" / f"mlp_{size}"
        frame = pd.read_csv(path / f"{mode}_predictions.csv")
        np.testing.assert_array_equal(frame.window, np.arange(len(reference)))
        aligned = frame.assign(subject_window=frame.groupby("subject").cumcount()).set_index(keys)
        aligned = aligned.loc[reference.index]
        np.testing.assert_array_equal(aligned["true"], reference.y_true)
        np.testing.assert_array_equal(aligned.repetition, reference.repetition)
        different = set(reference.fold[aligned.fold != reference.fold])
        if len(different) != (6 if mode == "within_subject" else 0):
            raise ValueError(f"Unexpected historical MLP-{size} fold assignment for {mode}")
        subjects, summary = score_predictions(frame, mode)
        pd.testing.assert_frame_equal(subjects, pd.read_csv(path / f"{mode}_subject_scores.csv"),
                                      atol=1e-12, rtol=0)
        current = pd.DataFrame([summary]).set_index("mode")
        for folder in (path, ARCHIVE / "mlp_legacy_scores" / f"mlp_{size}"):
            saved = pd.read_csv(folder / "cnn_summary.csv").set_index("mode").loc[[mode]]
            pd.testing.assert_frame_equal(current[saved.columns], saved, atol=1e-12, rtol=0)


def verify(classifiers, smooth):
    rows = []
    old_subject = pd.read_csv(ARCHIVE / "feature_selection/subject_scores.csv")
    old_fold = pd.read_csv(ARCHIVE / "feature_selection/fold_scores.csv")
    for mode in ("new_subject", "within_subject"):
        reference = pd.read_csv(BASELINE / mode / "predictions.csv")
        ordered = reference.sort_values("source_window").reset_index(drop=True)
        assert len(ordered) == 60182 and set(ordered.subject) == {f"S{i}" for i in range(1, 23)}
        verify_mlp(ordered, mode)
        for model, path in LOGS.items():
            domains = ["full_feature", "time", "amplitude"]
            if model != "rf":
                domains.append("baseline")
            elif mode == "new_subject":
                domains.append("full_60")
            log = (SRC.parent / "docs/classifiers" / path).read_text(encoding="utf-8")
            for domain in domains:
                name = f"{model}_{domain}"
                frame = pd.read_csv(classifiers / name / f"{mode}.csv.gz")
                pd.testing.assert_frame_equal(ordered[IDENTITY], frame[IDENTITY])
                s, subject, folds = summarize(frame, mode, name)
                if domain != "full_60":
                    check_log(log, domain, mode, s)
                if mode == "new_subject" and name in FEATURE_SETS:
                    for origin, key, scores in ((old_subject, "subject", subject), (old_fold, "fold", folds)):
                        values = origin[origin.feature_set == FEATURE_SETS[name]].set_index(key).macro_f1
                        np.testing.assert_allclose(scores.loc[values.index], values, atol=1e-12, rtol=0)
                rows.append(s)

        with np.load(ARCHIVE / "mixed_5050" / f"{mode}_predictions.npz", allow_pickle=False) as data:
            frame = pd.DataFrame({"source_window": np.arange(len(data["y"])), "subject": data["subject"],
                                  "repetition": data["repetition"], "fold": data["fold"],
                                  "y_true": data["y"], "y_pred": data["prediction"]})
        pd.testing.assert_frame_equal(ordered[IDENTITY], frame[IDENTITY])
        s, _, _ = summarize(frame, mode, "mixed_5050")
        historical = json.loads((ARCHIVE / "mixed_5050/summary.json").read_text())[mode]
        assert round(s["pooled_f1"], 3) == historical["macro_f1"]
        assert historical["summary"] == (f"mean macro_f1 {s['subject_f1_mean']:.3f} "
                f"(sd {s['subject_f1_sd']:.3f}, range {s['subject_f1_min']:.3f}-"
                f"{s['subject_f1_max']:.3f} across 22 subjects)")
        rows.append(s)

        # November's archive nests the recipe. Current single-run outputs store modes directly.
        recipe = "median_savgol_no_normalization"
        legacy = (smooth / recipe).is_dir()
        folder = smooth / recipe if legacy else smooth
        if not legacy:
            run = json.loads((smooth / "run.json").read_text(encoding="utf-8"))
            if (run["status"], run["model"], run["features"], run["preprocessing"]) != (
                    "complete", "rf", "baseline", "median-savgol"):
                raise ValueError("Expected a completed RF baseline median-savgol replay")
        frame = pd.read_csv(folder / mode / "predictions.csv")
        keys = ["subject", "fold", "y_true"] if legacy else IDENTITY
        pd.testing.assert_frame_equal(reference[keys], frame[keys])
        if not legacy:
            original = SRC.parent / "docs/Preprocessing/no_normalization_audit_20261119" / recipe
            pd.testing.assert_series_equal(frame.y_pred, pd.read_csv(original / mode / "predictions.csv").y_pred)
        s, _, _ = summarize(frame, mode, "smoothing")
        summaries = json.loads((smooth / "summary.json").read_text(encoding="utf-8"))
        summary = (summaries[recipe] if legacy else summaries)[mode]
        for key, saved in (("subject", "per_subject_spread"), ("fold", "per_fold_spread")):
            origin = summary[saved]["macro_f1"]
            np.testing.assert_allclose([s[f"{key}_f1_mean"], s[f"{key}_f1_sd"]],
                                       [origin["mean"], origin["sample_sd"]], atol=1e-12, rtol=0)
        rows.append(s)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifiers", type=Path, default=ARCHIVE / "classifiers")
    parser.add_argument("--smoothing", type=Path,
                        default=SRC.parent / "docs/Preprocessing/no_normalization_audit_20261119")
    parser.add_argument("--save", type=Path, help="Write the checked metrics to a CSV")
    args = parser.parse_args()
    summary = verify(args.classifiers, args.smoothing)
    if args.save:
        summary.to_csv(args.save, index=False)
    print(f"Verified {len(summary)} replay/ensemble results and six historical MLP results.")
