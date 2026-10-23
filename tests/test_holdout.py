"""Hold-out guards using synthetic signals only; never read reserved data."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapter import Recording
from emg_ninapro import EMGNinaproTrack, N_CH, WIN


def synthetic_record(group):
    return Recording(
        group=group,
        fs=100.0,
        epochs={"emg": np.ones((2, WIN, N_CH))},
        labels=np.array(["g01", "g02"]),
        meta={"rep": np.array([1, 2])},
    )


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.track = EMGNinaproTrack()

    def test_evaluate_modes_rejects_every_holdout_before_feature_construction(self):
        with patch.object(self.track, "_features_for") as features:
            for group in self.track.meta.holdout_ids:
                with self.subTest(group=group), self.assertRaisesRegex(ValueError, group):
                    self.track.evaluate_modes([synthetic_record("S1"), synthetic_record(group)])
            features.assert_not_called()

    def test_direct_evaluate_rejects_every_holdout_before_model_or_selector(self):
        X = np.ones((4, 1))
        y = np.array(["g01", "g02", "g01", "g02"])
        with patch.object(self.track, "_baseline") as baseline, \
                patch.object(self.track, "select_features") as selector:
            for group in self.track.meta.holdout_ids:
                groups = np.array(["S1", "S2", group, group])
                with self.subTest(group=group), self.assertRaisesRegex(ValueError, group):
                    self.track.evaluate(X, y, groups)
            baseline.assert_not_called()
            selector.assert_not_called()

    def test_train_baseline_rejects_every_holdout_before_feature_construction(self):
        with patch.object(self.track, "_features_for") as features:
            for group in self.track.meta.holdout_ids:
                with self.subTest(group=group), self.assertRaisesRegex(ValueError, group):
                    self.track.train_baseline([synthetic_record("S1"), synthetic_record(group)])
            features.assert_not_called()

    def test_frozen_inference_and_submission_accept_synthetic_holdout_without_refitting(self):
        model = self.track.train_baseline(
            [synthetic_record("S1"), synthetic_record("S2")],
            clf=DummyClassifier(strategy="most_frequent"),
        )
        holdout = synthetic_record("S23")
        with patch.object(model.clf, "fit", side_effect=AssertionError("inference must not refit")), \
                patch.object(self.track, "select_features") as selector, \
                tempfile.TemporaryDirectory() as tmp:
            predictions = self.track.infer(model, holdout)
            np.testing.assert_array_equal(predictions, ["g01", "g01"])
            path = Path(tmp) / "predictions.csv"
            self.track.write_submission([holdout], path, model)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [
                {"record": "S23", "epoch": "0", "label": "g01"},
                {"record": "S23", "epoch": "1", "label": "g01"},
            ])
            selector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
