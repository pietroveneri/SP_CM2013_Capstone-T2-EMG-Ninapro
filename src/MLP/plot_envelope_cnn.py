from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

PLOT_DIR = Path(__file__).resolve().parent
OUTPUT = PLOT_DIR / "mlp_40"
MODES = ("within_subject", "new_subject")


def plot_results(output=OUTPUT):
    fig_cm, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    fig_f1, ax_f1 = plt.subplots(figsize=(11, 4), layout="constrained")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5), layout="constrained")
    grid = np.linspace(0, 1, 201)
    has_roc = False

    for ax_cm, mode in zip(axes, MODES):
        pred = pd.read_csv(output / f"{mode}_predictions.csv")
        scores = pd.read_csv(output / f"{mode}_subject_scores.csv")
        scores = scores.sort_values(
            "subject", key=lambda s: s.str[1:].astype(int)
        )

        prob_cols = [col for col in pred if col.startswith("p_")]
        classes = (
            [col[2:] for col in prob_cols]
            if prob_cols else sorted(set(pred["true"]) | set(pred["predicted"]))
        )
        title = mode.replace("_", "-")

        ConfusionMatrixDisplay.from_predictions(
            pred["true"], pred["predicted"],
            labels=classes, normalize="true",
            cmap="Blues", colorbar=False,
            im_kw={"vmin": 0, "vmax": 1},
            values_format=".2f", xticks_rotation=45, ax=ax_cm,
        )
        ax_cm.set_title(title)

        ax_f1.plot(scores["subject"], scores["macro_f1"], "o-", label=title)

        if not prob_cols:
            print(f"{mode}: no probabilities saved; skipping ROC.")
            continue

        curves, areas = [], []
        for label, col in zip(classes, prob_cols):
            target = pred["true"].eq(label)
            if not target.any() or target.all():
                continue
            fpr, tpr, _ = roc_curve(target, pred[col])
            curves.append(np.interp(grid, fpr, tpr))
            areas.append(auc(fpr, tpr))

        if curves:
            ax_roc.plot(
                grid, np.mean(curves, axis=0),
                label=f"{title} (macro AUC={np.mean(areas):.3f})",
            )
            has_roc = True

    ax_f1.set(xlabel="Subject", ylabel="Macro-F1", ylim=(0, 1))
    ax_f1.tick_params(axis="x", rotation=45)
    ax_f1.legend()

    figures = [(fig_cm, "cnn_confusion"), (fig_f1, "cnn_macro_f1")]

    if has_roc:
        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax_roc.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="Pooled out-of-fold macro-average ROC",
        )
        ax_roc.legend()
        figures.append((fig_roc, "cnn_roc_auc"))

    for fig, name in figures:
        path = OUTPUT / f"{name}.png"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
        plt.close(fig)
    plt.close(fig_roc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT)
    plot_results(parser.parse_args().output)