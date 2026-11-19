import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from adapter import Recording
from experiments import ExpandedFeatureTrack
from visualize_features import FEATURE_NAMES, FEATURE_SET
from development_folds import BASELINE, development_folds
from knn.run_knn import KNNTrack
from lda_regularized.run_lda_regularized import LDATrack
from svm_rbf.run_svm_rbf import SVMTrack


def test_plot_labels_match_the_extracted_columns():
    t = np.arange(20)[:, None]
    gain = np.arange(1, 11)[None, :]
    windows = (2 + np.sin(2 * np.pi * t / 5)) * gain
    record = Recording("S1", 100, {"emg": windows[None]}, np.array(["g01"]))
    values = ExpandedFeatureTrack().extract_features(record)[0][0]
    by_name = dict(zip(FEATURE_NAMES, values, strict=True))
    assert FEATURE_SET == "expanded_70"
    for channel in range(10):
        x = windows[:, channel]
        prefix = f"ch{channel + 1:02d}_"
        assert by_name[prefix + "MAV"] == pytest.approx(x.mean())
        assert by_name[prefix + "RMS"] == pytest.approx(np.sqrt(np.mean(x ** 2)))
        assert by_name[prefix + "WL"] == pytest.approx(np.abs(np.diff(x)).sum())
        assert by_name[prefix + "VAR"] == pytest.approx(x.var())
        assert by_name[prefix + "relative_MAV"] == pytest.approx(x.mean() / windows.mean(axis=0).sum())
        assert by_name[prefix + "mean_frequency"] == pytest.approx(20, abs=0.05)
        assert by_name[prefix + "median_frequency"] == 20


def test_model_config_reaches_estimators():
    lda = LDATrack("baseline", {"shrinkage": 0.3})
    assert "shrinkage" not in lda.feature_cfg_keys()
    assert lda.baseline()[-1].shrinkage == 0.3
    assert lda.baseline({"shrinkage": 0.1})[-1].shrinkage == 0.1
    assert KNNTrack("time").baseline({"n_neighbors": 3})[-1].n_neighbors == 3
    svm = SVMTrack("amplitude")
    assert svm.baseline()[-1].class_weight == "balanced"
    params = svm.baseline({"imbalance": "none", "C": 2, "gamma": 0.5, "seed": 42})[-1]
    assert (params.class_weight, params.C, params.gamma, params.random_state) == (None, 2, 0.5, 42)


@pytest.mark.parametrize("track", [LDATrack, KNNTrack])
def test_fold_resampling_reads_seed(track):
    model = track("baseline").baseline({"imbalance": "resample", "seed": 92})
    assert model.random_state == 92


@pytest.mark.parametrize("track", [LDATrack, KNNTrack, SVMTrack])
def test_unsupported_imbalance_is_reported(track):
    with pytest.raises(ValueError, match="imbalance"):
        track("baseline").baseline({"imbalance": "balanced_subsample"})


def test_archive_splits_survive_different_window_order():
    within = pd.read_csv(BASELINE / "within_subject/predictions.csv").sample(frac=1, random_state=0)
    new_folds = pd.read_csv(BASELINE / "new_subject/predictions.csv").set_index("source_window").fold
    expected = {"within_subject": within.fold.to_numpy(),
                "new_subject": new_folds.loc[within.source_window].to_numpy()}
    for mode, splits in development_folds(within.subject, within.repetition).items():
        for fold, (train, test) in enumerate(splits):
            assert np.all(expected[mode][test] == fold)
            assert not np.intersect1d(train, test).size
