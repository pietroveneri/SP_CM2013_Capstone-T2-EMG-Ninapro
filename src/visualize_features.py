from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from experiments import ExpandedFeatureTrack
from features import expanded_feature_names



ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw"
OUTPUT_DIR = ROOT / "outputs" / "feature_analysis"

SEED = 0
WITHIN_SUBJECT = "S1"


def save_figure(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # for extension in ("png"):
    extension = "png"
    path = OUTPUT_DIR / f"{name}_no_rms.{extension}"
    fig.savefig(path, format=extension, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")


def plot_feature_hierarchy(X_train, names):
    """Cluster nonconstant features using absolute Pearson correlation."""
    keep = np.var(X_train, axis=0) > 0
    X = X_train[:, keep]
    names = np.asarray(names)[keep]

    if X.shape[1] < 2:
        raise ValueError("Need at least two nonconstant features.")

    correlation = np.corrcoef(X, rowvar=False)
    correlation = np.clip(correlation, -1, 1)

    if not np.isfinite(correlation).all():
        raise ValueError("Feature correlations contain NaN or infinity.")

    distance = 1 - np.abs(correlation)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)

    hierarchy = linkage(
        squareform(distance, checks=True),
        method="average",
        optimal_ordering=True,
    )

    fig = plt.figure(figsize=(19, 16))
    grid = fig.add_gridspec(
        2, 1, height_ratios=[1, 5], hspace=0.03
    )

    dendrogram_ax = fig.add_subplot(grid[0])
    tree = dendrogram(
        hierarchy,
        ax=dendrogram_ax,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
    )
    dendrogram_ax.set_ylabel("Distance")
    dendrogram_ax.set_title(
        "Expanded feature hierarchy — subject-fold training data"
    )
    dendrogram_ax.set_xticks([])

    order = tree["leaves"]
    ordered = correlation[np.ix_(order, order)]
    ordered_names = names[order]

    heatmap_ax = fig.add_subplot(grid[1])
    image = heatmap_ax.imshow(
        ordered,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    positions = np.arange(len(ordered_names))
    heatmap_ax.set_xticks(
        positions, ordered_names, rotation=90, fontsize=6
    )
    heatmap_ax.set_yticks(
        positions, ordered_names, fontsize=6
    )

    # Separate colourbar so both panels retain aligned widths.
    fig.subplots_adjust(right=0.90)
    colour_ax = fig.add_axes([0.92, 0.20, 0.015, 0.45])
    fig.colorbar(
        image, cax=colour_ax, label="Pearson correlation"
    )

    save_figure(fig, "feature_correlation_hierarchy")


def plot_lda(X, y, train_indices, test_indices, title, name):
    projection = make_pipeline(
        VarianceThreshold(),
        StandardScaler(),
        LinearDiscriminantAnalysis(
            n_components=2,
            solver="svd",
        ),
    )

    Z_train = projection.fit_transform(
        X[train_indices], y[train_indices]
    )
    Z_test = projection.transform(X[test_indices])

    classes = np.unique(y)
    colours = plt.get_cmap("tab20").colors
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    for ax, Z, labels, panel_title in (
        (
            axes[0], Z_train, y[train_indices],
            "Training windows",
        ),
        (
            axes[1], Z_test, y[test_indices],
            "Unseen windows",
        ),
    ):
        for index, gesture in enumerate(classes):
            rows = np.flatnonzero(labels == gesture)

            # Limit displayed points, without changing LDA fitting.
            if len(rows) > 400:
                rows = rng.choice(
                    rows, size=400, replace=False
                )

            ax.scatter(
                Z[rows, 0],
                Z[rows, 1],
                s=9,
                alpha=0.35,
                color=colours[index],
                label=gesture,
                rasterized=True,
            )

        ax.set_title(panel_title)
        ax.set_xlabel("LD1")
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("LD2")
    axes[1].legend(
        title="Gesture",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        markerscale=2,
    )

    fig.suptitle(
        f"{title}\n"
        "Exploratory supervised projection; fitted on training data only"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    save_figure(fig, name)
    

def plot_feature_importance(X, y, train, test, names):
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    print("Fitting random forest for feature importance...")
    model.fit(X[train], y[train])

    print("Calculating permutation importance...")
    importance = permutation_importance(
        model,
        X[test],
        y[test],
        scoring="f1_macro",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )

    means = importance.importances_mean
    stds = importance.importances_std
    names = np.asarray(names)

    # Display the 20 highest-ranked features.
    order = np.argsort(means)[-20:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        names[order],
        means[order],
        xerr=stds[order],
        color="steelblue",
        alpha=0.85,
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Decrease in macro-F1 after shuffling")
    ax.set_title(
        "Permutation importance — unseen development subjects\n"
        "First subject-grouped fold; error bars: SD over 5 shuffles"
    )
    fig.tight_layout()

    save_figure(fig, "permutation_importance")

    # Save the complete ranking, including negative values.
    import csv

    with (
        OUTPUT_DIR / "permutation_importance_no_rms.csv"
    ).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "feature",
            "mean_macro_f1_decrease",
            "shuffle_sd",
        ])

        for index in np.argsort(means)[::-1]:
            writer.writerow([
                names[index],
                means[index],
                stds[index],
            ])

    predictions = model.predict(X[test])
    print(
        "Reference fold macro-F1:",
        f1_score(y[test], predictions, average="macro"),
    )


def main():
    track = ExpandedFeatureTrack()
    records = track.load(DATA_DIR, exercise="E1")

    if not records:
        raise RuntimeError(f"No E1 recordings found in {DATA_DIR}")

    track.assert_no_holdout(records)

    feature_rows = []
    labels = []
    subjects = []
    repetitions = []

    for record in records:
        X_record, y_record, subject = track.extract_features(record)

        feature_rows.append(X_record)
        labels.append(y_record)
        subjects.append(
            np.repeat(subject, len(y_record))
        )
        repetitions.append(
            np.asarray(record.meta["rep"])
        )

    X = np.vstack(feature_rows)
    y = np.concatenate(labels)
    subjects = np.concatenate(subjects)
    repetitions = np.concatenate(repetitions)

    names = expanded_feature_names()

    if X.shape[1] != len(names):
        raise ValueError(
            f"Found {X.shape[1]} columns but {len(names)} feature names."
        )

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity.")

    print(f"Feature matrix: {X.shape}")
    print(f"Development subjects: {np.unique(subjects)}")

    # First fold of the same five-fold subject-grouped scheme.
    subject_cv = GroupKFold(n_splits=5)
    train, test = next(
        subject_cv.split(X, y, groups=subjects)
    )

    print("New-subject training:", np.unique(subjects[train]))
    print("New-subject test:", np.unique(subjects[test]))

    plot_feature_hierarchy(X[train], names)

    plot_lda(
        X, y, train, test,
        title="New-subject LDA — first subject-grouped fold",
        name="lda_new_subject",
    )

    # Within-subject: keep each repetition entirely in one fold.
    subject_rows = np.flatnonzero(
        subjects == WITHIN_SUBJECT
    )

    if len(subject_rows) == 0:
        raise ValueError(f"{WITHIN_SUBJECT} was not loaded.")

    repetition_cv = GroupKFold(n_splits=5)
    train_local, test_local = next(
        repetition_cv.split(
            X[subject_rows],
            y[subject_rows],
            groups=repetitions[subject_rows],
        )
    )

    plot_feature_importance(X, y, train, test, names)

    plot_lda(
        X, y,
        subject_rows[train_local],
        subject_rows[test_local],
        title=f"Within-subject LDA — {WITHIN_SUBJECT}, held-out repetitions",
        name="lda_within_subject",
    )

    # Save enough information to identify the splits later.
    np.savez_compressed(
        OUTPUT_DIR / "projection_split_metadata.npz",
        subject_train=np.unique(subjects[train]),
        subject_test=np.unique(subjects[test]),
        within_subject=WITHIN_SUBJECT,
        within_train_repetitions=np.unique(
            repetitions[subject_rows[train_local]]
        ),
        within_test_repetitions=np.unique(
            repetitions[subject_rows[test_local]]
        ),
        feature_names=np.asarray(names),
        seed=SEED,
    )


if __name__ == "__main__":
    main()