import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from MLP.score_predictions import score_predictions
from development_folds import development_folds
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

# cuBLAS reads this when the CUDA context is created.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from features import extract_window_features, expanded_feature_names


torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", choices=("envelope", "60", "40", "20"), default="envelope"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--output-dir", type=Path, help="Fresh directory for this run")
args = parser.parse_args()

track = EMGNinaproTrack()
records = track.load(ROOT / "NinaPro_Mat", exercise="E1")
track.assert_no_holdout(records)

X = np.concatenate([r.epochs['emg'] for r in records])
labels = np.concatenate([r.labels for r in records])
subjects = np.concatenate([np.repeat(r.group, len(r.labels)) for r in records])
repetitions = np.concatenate([r.meta["rep"] for r in records])

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
# Map both modes to the archived subject and repetition assignments.
splits = development_folds(subjects, repetitions)
output = args.output_dir or ROOT / "outputs" / run_name
output.mkdir(parents=True, exist_ok=False)
settings = {"input": args.input, "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "torch": torch.__version__, "numpy": np.__version__, "device": str(DEVICE),
            "device_name": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else "CPU",
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
(output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def evaluate_mode(mode, splits):
    probabilities = np.full((len(y), len(classes)), np.nan, dtype=np.float32)
    predictions = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)

    for fold, (train, test) in enumerate(splits):
        assert np.all(fold_ids[test] == -1), "Repeated test window"
        print(f"{mode}: fold {fold + 1}/{len(splits)}", flush=True)

        predictions[test], probabilities[test] = fit_predict(train, test)
        fold_ids[test] = fold
    assert np.all(fold_ids >= 0), "Missing test windows"
    assert np.isfinite(probabilities).all(), "Missing class probabilities"

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

    subject_df, summary = score_predictions(prediction_df, mode)
    subject_df.to_csv(output / f"{mode}_subject_scores.csv", index=False)

    print(
        f"{mode}: mean subject macro-F1 "
        f"{summary['mean_subject_macro_f1']:.4f} ± {summary['subject_sd']:.4f}; "
        f"pooled {summary['pooled_macro_f1']:.4f}",
        flush=True,
    )
    return summary


summaries = [evaluate_mode(mode, folds) for mode, folds in splits.items()]
pd.DataFrame(summaries).to_csv(output / "cnn_summary.csv", index=False)
