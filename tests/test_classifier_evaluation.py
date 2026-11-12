import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classifier_evaluation import evaluate_modes_progress
from emg_ninapro import EMGNinaproTrack
from svm_rbf.run_svm_rbf import CFG, SVMTrack


class ClassifierEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.track = SVMTrack("baseline")
        self.records = EMGNinaproTrack().smoke(n_subjects=2, n_reps=2, win_per_rep=2)

    def test_missing_repetitions_fail_before_build(self):
        del self.records[0].meta["rep"]
        with patch.object(self.track, "_build") as build:
            with self.assertRaisesRegex(ValueError, "missing repetition"):
                evaluate_modes_progress(self.track, self.records, CFG)
            build.assert_not_called()

    def test_misaligned_repetitions_fail_before_build(self):
        self.records[0].meta["rep"] = np.array([1, 2])
        with patch.object(self.track, "_build") as build:
            with self.assertRaisesRegex(ValueError, "must match"):
                evaluate_modes_progress(self.track, self.records, CFG)
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
                        evaluate_modes_progress(self.track, records, CFG)
                    model.assert_not_called()

    def test_incomplete_new_subject_predictions_fail(self):
        class IncompleteSplit:
            def split(self, X, y, groups):
                yield np.flatnonzero(groups == "S1"), np.flatnonzero(groups == "S2")

        with patch("classifier_evaluation.LeaveOneGroupOut", return_value=IncompleteSplit()):
            with self.assertRaisesRegex(ValueError, "New-subject.*exactly once"):
                evaluate_modes_progress(self.track, self.records, CFG)

    def test_duplicate_new_subject_predictions_fail(self):
        class DuplicateSplit:
            def split(self, X, y, groups):
                train, test = np.flatnonzero(groups == "S1"), np.flatnonzero(groups == "S2")
                yield train, test
                yield train, test

        with patch("classifier_evaluation.LeaveOneGroupOut", return_value=DuplicateSplit()):
            with self.assertRaisesRegex(ValueError, "New-subject.*exactly once"):
                evaluate_modes_progress(self.track, self.records, CFG)

    def test_incomplete_within_subject_predictions_fail(self):
        class IncompleteSplit:
            def split(self, X, y, groups):
                yield np.flatnonzero(groups == 1), np.flatnonzero(groups == 2)

        with patch("classifier_evaluation.GroupKFold", return_value=IncompleteSplit()):
            with self.assertRaisesRegex(ValueError, "Within-subject.*exactly once"):
                evaluate_modes_progress(self.track, self.records, CFG)

    def test_valid_data_predict_every_window(self):
        results = evaluate_modes_progress(self.track, self.records, CFG)
        for result in results.values():
            self.assertEqual(len(result["y_pred"]), sum(len(record.labels) for record in self.records))


if __name__ == "__main__":
    unittest.main()
