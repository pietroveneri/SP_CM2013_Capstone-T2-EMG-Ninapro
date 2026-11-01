if __name__ == "__main__":
    raise SystemExit("Run run_lda_regularized.py to evaluate the classifier and generate plots.")

import matplotlib.pyplot as plt
import numpy as np

from report import plot_confusion


def save_plots(results, domain, output):
    output.mkdir(parents=True, exist_ok=True)
    title = domain.replace("_", " ").title()
    modes = ("within_subject", "new_subject")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mode in zip(axes, modes):
        result = results[mode]
        plot_confusion(result["y_true"], result["y_pred"], labels=result["labels"],
                       ax=ax, title=f"{mode}: normalized confusion")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / f"{domain}_confusion.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, mode in zip(axes, modes):
        rows = sorted(results[mode]["per_group"], key=lambda item: int(item["group"][1:]))
        labels = [item["group"] for item in rows]
        ax.bar(np.arange(len(labels)), [item["macro_f1"] for item in rows])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{mode}: per subject")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / f"{domain}_macro_f1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 101)
    for mode in modes:
        all_folds = results[mode]["roc_folds"]
        valid = [fold for fold in all_folds if fold is not None]
        if valid:
            tpr = np.mean([fold["tpr"] for fold in valid], axis=0)
            mean_auc = np.mean([fold["auc"] for fold in valid])
            ax.plot(grid, tpr, label=f"{mode}: mean AUC {mean_auc:.3f} ({len(valid)}/{len(all_folds)} folds)")
        else:
            ax.plot([], [], label=f"{mode}: no eligible folds")
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
           ylabel="True positive rate", title=f"{title}: mean one-vs-rest ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"{domain}_roc_auc.png", dpi=150)
    plt.close(fig)
