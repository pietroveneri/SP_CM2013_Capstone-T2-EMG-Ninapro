"""Compute MLP/CNN score tables from out-of-fold predictions without retraining."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score


def score_predictions(frame, mode):
    def scores(rows):
        return {"macro_f1": f1_score(rows["true"], rows.predicted, average="macro"),
                "cohens_kappa": cohen_kappa_score(rows["true"], rows.predicted)}

    subjects = pd.DataFrame([{"subject": s, **scores(rows)}
                             for s, rows in frame.groupby("subject")])
    folds = pd.Series([f1_score(rows["true"], rows.predicted, average="macro")
                       for _, rows in frame.groupby("fold")])
    f1, kappa = subjects.macro_f1, subjects.cohens_kappa
    pooled = scores(frame)
    summary = {
        "mode": mode, "n_subjects": len(subjects), "n_folds": len(folds),
        "mean_subject_macro_f1": f1.mean(), "subject_sd": f1.std(),
        "subject_min": f1.min(), "subject_max": f1.max(),
        "pooled_macro_f1": pooled["macro_f1"],
        "mean_fold_macro_f1": folds.mean(), "fold_sd": folds.std(),
        "mean_subject_cohens_kappa": kappa.mean(), "subject_cohens_kappa_sd": kappa.std(),
        "subject_cohens_kappa_min": kappa.min(), "subject_cohens_kappa_max": kappa.max(),
        "pooled_cohens_kappa": pooled["cohens_kappa"],
    }
    return subjects, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for mode in ("new_subject", "within_subject"):
        frame = pd.read_csv(args.run / f"{mode}_predictions.csv")
        subjects, summary = score_predictions(frame, mode)
        subjects.to_csv(args.output / f"{mode}_subject_scores.csv", index=False)
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(args.output / "cnn_summary.csv", index=False)


if __name__ == "__main__":
    main()
