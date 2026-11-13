import json 
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

import matplotlib

matplotlib.use("Agg")  # Save figures without opening a window.
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from emg_ninapro import EMGNinaproTrack

torch_version = tuple(
    int(part) for part in version("torch").split("+", 1)[0].split(".")[:2]
)
numpy_major_version = int(np.__version__.split(".", maxsplit=1)[0])
if torch_version < (2, 4) and numpy_major_version >= 2:
    raise RuntimeError(
        "This PyTorch version cannot run with NumPy 2. Install the CNN "
        "dependencies into this interpreter with: "
        f'"{sys.executable}" -m pip install -r "{ROOT / "requirements-cnn.txt"}"'
    )

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import argparse
from features import extract_window_features, expanded_feature_names


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

track = EMGNinaproTrack()
records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
track.assert_no_holdout(records)

X = np.concatenate([r.epochs['emg'] for r in records])
labels = np.concatenate([r.labels for r in records])
subjects = np.concatenate([np.repeat(r.group, len(r.labels)) for r in records])
repetitions = np.concatenate([r.meta["rep"] for r in records])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", choices=("envelope", "60", "40", "20"), default="envelope"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=256)
args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

feature_sets = {
    "60": {"MAV", "WL", "VAR", "relative_MAV",
           "mean_frequency", "median_frequency"},
    "40": {"MAV", "WL", "VAR", "relative_MAV"},
    "20": {"MAV", "relative_MAV"},
}

use_cnn = args.input == "envelope"
run_name = "envelope_cnn" if use_cnn else f"mlp_{args.input}"

if not use_cnn:
    names = expanded_feature_names()
    columns = [
        i for i, name in enumerate(names)
        if name.split("_", 1)[1] in feature_sets[args.input]
    ]

    rows = []
    for record in records:
        features, feature_labels, subject = extract_window_features(record)
        assert features.shape[1] == len(names)
        assert np.array_equal(feature_labels, record.labels)
        assert subject == record.group
        rows.append(features[:, columns])

    X = np.concatenate(rows).astype(np.float32)
    assert X.shape == (len(labels), int(args.input))
else:
    assert X.shape == (len(labels), 20, 10)

assert np.isfinite(X).all()

if use_cnn:
    X = X.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
classes = list(track.meta.classes)
class_to_id = {label: i for i, label in enumerate(classes)}
y = np.array([class_to_id[label] for label in labels], dtype=np.int64)

# -------------------------------------------------------------------------------------------------
# train a fresh CNN inside each fold, and predict the test samples. The model is trained for a fixed number of epochs, with a balanced loss function.

def fit_predict(train, test):
    torch.manual_seed(SEED)

    # One mean and SD per channel, estimated from training samples only.
    X_train = X[train]
    scale_axes = (0, 2) if use_cnn else 0
    mean = X_train.mean(axis=scale_axes, keepdims=True)
    sd = X_train.std(axis=scale_axes, keepdims=True).clip(min=1e-6)

    x_train = torch.from_numpy((X_train - mean) / sd)
    x_test = torch.from_numpy((X[test] - mean) / sd)

    batches = DataLoader(
        TensorDataset(x_train, torch.from_numpy(y[train])),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    if use_cnn:
        model = nn.Sequential(
            nn.Conv1d(10, 16, kernel_size=3, padding=1), # 10 channels, 16 filters
            nn.ReLU(),  # Activation function
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32 * 4, len(classes)),
        )
    else:
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, len(classes)),
        )

    model = model.to(DEVICE)

    # Balanced loss weights, calculated from this training fold.
    counts = np.bincount(y[train], minlength=len(classes))
    assert np.all(counts > 0), "A gesture is missing from the training fold"
    weights = len(train) / (len(classes) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in batches:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probabilities = np.concatenate([
            model(xb.to(DEVICE)).softmax(dim=1).cpu().numpy()
            for xb in x_test.split(BATCH_SIZE)
        ])

    return probabilities.argmax(axis=1), probabilities

# -------------------------------------------------------------------------------------------------
#  Load the pre-defined folds for new-subject evaluation, and generate 5-fold within-subject splits.

fold_path = (
    ROOT / "docs" / "Baseline"
    / "baseline_20261022_holdout_separated" / "folds.json"
)

archived = json.loads(fold_path.read_text())

new_subject = []
for f in sorted(
    (f for f in archived if f["mode"] == "new_subject"),
    key=lambda f: f["fold"],
):
    train_groups = set(f["train_groups"])
    test_groups = set(f["test_groups"])
    assert train_groups.isdisjoint(test_groups)

    train = np.flatnonzero(np.isin(subjects, list(train_groups)))
    test = np.flatnonzero(np.isin(subjects, list(test_groups)))
    assert len(train) == f["train_n"] and len(test) == f["test_n"]
    new_subject.append((train, test))

assert len(new_subject) == 5

within_subject = []
for subject in np.unique(subjects):
    rows = np.flatnonzero(subjects == subject)
    reps = repetitions[rows]
    cv = GroupKFold(n_splits=5)
    for tr, te in cv.split(X[rows], y[rows], groups=reps):
        assert set(reps[tr]).isdisjoint(reps[te])
        within_subject.append((rows[tr], rows[te]))

output = Path(__file__).resolve().parent / run_name
output.mkdir(parents=True, exist_ok=True)


def evaluate_mode(mode, splits):
    probabilities = np.full((len(y), len(classes)), np.nan, dtype=np.float32)
    predictions = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_scores = []

    for fold, (train, test) in enumerate(splits):
        assert np.all(fold_ids[test] == -1), "Repeated test window"
        print(f"{mode}: fold {fold + 1}/{len(splits)}", flush=True)

        predictions[test], probabilities[test] = fit_predict(train, test)
        fold_ids[test] = fold
        fold_scores.append(
            f1_score(y[test], predictions[test], average="macro")
        )

    assert np.all(fold_ids >= 0), "Missing test windows"
    assert np.isfinite(probabilities).all(), "Missing class probabilities"

    subject_df = pd.DataFrame([
        {
            "subject": subject,
            "macro_f1": f1_score(
                y[subjects == subject],
                predictions[subjects == subject],
                average="macro",
            ),
        }
        for subject in np.unique(subjects)
    ])
    subject_df.to_csv(output / f"{mode}_subject_scores.csv", index=False)

    prediction_df = pd.DataFrame({
        "window": np.arange(len(y)),
        "subject": subjects,
        "repetition": repetitions,
        "fold": fold_ids,
        "true": labels,
        "predicted": np.asarray(classes)[predictions],
    })
    for k, label in enumerate(classes):
        prediction_df[f"p_{label}"] = probabilities[:, k]
    prediction_df.to_csv(output / f"{mode}_predictions.csv", index=False)

    scores = subject_df["macro_f1"]
    summary = {
        "mode": mode,
        "n_subjects": len(scores),
        "n_folds": len(splits),
        "mean_subject_macro_f1": scores.mean(),
        "subject_sd": scores.std(ddof=1),
        "subject_min": scores.min(),
        "subject_max": scores.max(),
        "pooled_macro_f1": f1_score(y, predictions, average="macro"),
        "mean_fold_macro_f1": np.mean(fold_scores),
        "fold_sd": np.std(fold_scores, ddof=1),
    }

    print(
        f"{mode}: mean subject macro-F1 "
        f"{summary['mean_subject_macro_f1']:.4f} ± {summary['subject_sd']:.4f}; "
        f"pooled {summary['pooled_macro_f1']:.4f}",
        flush=True,
    )
    return summary


summaries = [
    evaluate_mode("new_subject", new_subject),
    evaluate_mode("within_subject", within_subject),
]
pd.DataFrame(summaries).to_csv(output / "cnn_summary.csv", index=False)

# python -u src/envelope_cnn/run_envelope_cnn.py --input 60
# python src/envelope_cnn/plot_envelope_cnn.py outputs/mlp_60