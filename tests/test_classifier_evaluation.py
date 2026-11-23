import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classifier_evaluation import evaluate_modes_progress
from emg_ninapro import EMGNinaproTrack
from classifiers import MODELS, SVMTrack
from preprocessing import GuardedAverageReference, filter_windows


class ClassifierEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.track = SVMTrack("baseline")
        self.records = EMGNinaproTrack().smoke(n_subjects=2, n_reps=2, win_per_rep=2)

    def test_missing_repetitions_fail_before_build(self):
        del self.records[0].meta["rep"]
        with patch.object(self.track, "_build") as build:
            with self.assertRaisesRegex(ValueError, "missing repetition"):
                evaluate_modes_progress(self.track, self.records, self.track.cfg)
            build.assert_not_called()

    def test_misaligned_repetitions_fail_before_build(self):
        self.records[0].meta["rep"] = np.array([1, 2])
        with patch.object(self.track, "_build") as build:
            with self.assertRaisesRegex(ValueError, "must match"):
                evaluate_modes_progress(self.track, self.records, self.track.cfg)
            build.assert_not_called()

    def test_unevaluable_subjects_fail_before_fit(self):
        for field in ("repetitions", "classes"):
            with self.subTest(field=field):
                records = EMGNinaproTrack().smoke(n_subjects=2, n_reps=2, win_per_rep=2)
                if field == "repetitions":
                    records[0].meta["rep"][:] = 1
                else:
                    records[0].labels[:] = "g01"
                with patch.object(self.track, "_baseline") as model:
                    with self.assertRaisesRegex(ValueError, "S1.*at least two"):
                        evaluate_modes_progress(self.track, records, self.track.cfg)
                    model.assert_not_called()

    def test_incomplete_new_subject_predictions_fail(self):
        class IncompleteSplit:
            def split(self, X, y, groups):
                yield np.flatnonzero(groups == "S1"), np.flatnonzero(groups == "S2")

        with patch("classifier_evaluation.LeaveOneGroupOut", return_value=IncompleteSplit()):
            with self.assertRaisesRegex(ValueError, "New-subject.*exactly once"):
                evaluate_modes_progress(self.track, self.records, self.track.cfg)

    def test_duplicate_new_subject_predictions_fail(self):
        class DuplicateSplit:
            def split(self, X, y, groups):
                train, test = np.flatnonzero(groups == "S1"), np.flatnonzero(groups == "S2")
                yield train, test
                yield train, test

        with patch("classifier_evaluation.LeaveOneGroupOut", return_value=DuplicateSplit()):
            with self.assertRaisesRegex(ValueError, "New-subject.*exactly once"):
                evaluate_modes_progress(self.track, self.records, self.track.cfg)

    def test_incomplete_within_subject_predictions_fail(self):
        class IncompleteSplit:
            def split(self, X, y, groups):
                yield np.flatnonzero(groups == 1), np.flatnonzero(groups == 2)

        with patch("classifier_evaluation.GroupKFold", return_value=IncompleteSplit()):
            with self.assertRaisesRegex(ValueError, "Within-subject.*exactly once"):
                evaluate_modes_progress(self.track, self.records, self.track.cfg)

    def test_smoothed_rf_uses_requested_columns_and_window_axis(self):
        from scipy.ndimage import median_filter
        from scipy.signal import savgol_filter
        from features import extract_window_features

        rec = self.records[0]
        windows = rec.epochs["emg"]
        expected = savgol_filter(median_filter(windows, size=(1, 3, 1), mode="nearest"),
                                 5, 2, axis=1, mode="nearest")
        smoothed = filter_windows(rec, "median-savgol")
        np.testing.assert_array_equal(smoothed.epochs["emg"], expected)
        expanded = extract_window_features(smoothed)[0].reshape(len(windows), 10, 7)
        for name, columns in (("time_40", [0, 1, 2, 3]), ("amplitude_20", [0, 3]),
                              ("full_60", list(range(6))), ("full_70", list(range(7)))):
            with self.subTest(name=name):
                actual, labels, subject = MODELS["rf"](name).extract_features(smoothed)
                np.testing.assert_array_equal(actual, expanded[:, :, columns].reshape(len(windows), -1))
                np.testing.assert_array_equal(labels, rec.labels)
                self.assertEqual(subject, rec.group)

    def test_guarded_reference_fits_only_the_training_windows(self):
        # A per-window marker identifies precisely which inputs reach each fit.
        ids = np.arange(sum(len(r.labels) for r in self.records))
        seen = []
        original_fit = GuardedAverageReference.fit

        def capture_fit(reference, train):
            seen.append(train[:, 0, 0].astype(int))
            return original_fit(reference, train)

        start = 0
        for record in self.records:
            stop = start + len(record.labels)
            record.epochs["emg"][:, 0, 0] = ids[start:stop]
            start = stop
        with patch.object(GuardedAverageReference, "fit", capture_fit):
            results = evaluate_modes_progress(self.track, self.records, self.track.cfg,
                                              preprocessing="guarded-reference")

        fits = iter(seen)
        for mode, result in results.items():
            for fold in np.unique(result["y_fold"]):
                test = result["source_window"][result["y_fold"] == fold]
                eligible = ids
                if mode == "within_subject":
                    subject_ids = result["y_group"] == result["y_group"][result["y_fold"] == fold][0]
                    eligible = result["source_window"][subject_ids]
                np.testing.assert_array_equal(next(fits), np.setdiff1d(eligible, test))
        self.assertIsNone(next(fits, None))

    def test_valid_data_predict_every_window(self):
        results = evaluate_modes_progress(self.track, self.records, self.track.cfg)
        for result in results.values():
            self.assertEqual(len(result["y_pred"]), sum(len(record.labels) for record in self.records))


if __name__ == "__main__":
    unittest.main()
