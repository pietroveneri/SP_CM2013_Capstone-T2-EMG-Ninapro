"""Leakage checks for signal-level preprocessing and grouped CV."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adapter import denoise
from emg_ninapro import EMGNinaproTrack
from fold_safe_preprocessing import GuardedAverageReference, FoldSafePreprocessingTrack


class PreprocessingTests(unittest.TestCase):
    def test_held_out_data_cannot_change_training_reference(self):
        rng = np.random.default_rng(47)
        train = rng.uniform(0.2, 1.5, (8, 20, 4))
        test = rng.uniform(0.2, 1.5, (4, 20, 4))
        first = GuardedAverageReference().fit(train)
        before = first.transform(train)
        second = GuardedAverageReference().fit(train)
        second.transform(test * 1e6)
        np.testing.assert_allclose(before, second.transform(train))
        np.testing.assert_array_equal(first.live_, second.live_)

    def test_option06_uses_fitted_live_mask_for_test_reference(self):
        train = np.ones((3, 20, 3))
        train[..., 2] = 0.001
        transform = GuardedAverageReference().fit(train)
        np.testing.assert_array_equal(transform.live_, [True, True, False])
        test = np.ones((1, 20, 3))
        test[..., 2] = 100
        out = transform.transform(test)
        np.testing.assert_allclose(out[..., :2], 0)
        np.testing.assert_allclose(out[..., 2], 99)

    def test_options_1_to_5_pass_fold_windows_to_features_without_normalizing(self):
        rng = np.random.default_rng(11)
        windows = rng.uniform(3, 7, (5, 20, 3))
        labels = np.array(["a", "b", "a", "b", "a"])
        tr, te = np.array([0, 1, 2]), np.array([3, 4])
        for option in range(1, 6):
            with self.subTest(option=option):
                track = FoldSafePreprocessingTrack(option=option)
                seen = []
                def capture(rec):
                    seen.append(rec.epochs["emg"].copy())
                    return np.zeros((len(rec.labels), 1)), rec.labels, rec.group
                track.extract_features = capture
                track._fold_features(windows, labels, 100, tr, te)
                np.testing.assert_array_equal(seen[0], windows[tr])
                np.testing.assert_array_equal(seen[1], windows[te])

    def test_identity_options_match_baseline_predictions(self):
        baseline = EMGNinaproTrack()
        records = baseline.smoke(n_subjects=2, n_reps=2, win_per_rep=1)
        expected = baseline.evaluate_modes(records)
        for option in (2, 5):
            with self.subTest(option=option):
                track = FoldSafePreprocessingTrack(option=option)
                for rec in records:
                    np.testing.assert_array_equal(track.preprocess(rec).epochs["emg"],
                                                  rec.epochs["emg"])
                actual = track.evaluate_modes(records)
                for mode in ("within_subject", "new_subject"):
                    for key in ("y_true", "y_pred", "y_group", "y_fold"):
                        np.testing.assert_array_equal(actual[mode][key], expected[mode][key])

    def test_fixed_signal_filters_match_hannah_defaults(self):
        for option in range(1, 7):
            with self.subTest(option=option):
                track = FoldSafePreprocessingTrack(option=option)
                rec = track.smoke(n_subjects=1, n_reps=2, win_per_rep=1)[0]
                oriented = rec.replace_epochs({"emg": rec.epochs["emg"].swapaxes(-1, -2)})
                if option == 3:
                    expected = denoise(oriented, impulsive="median", win=3)
                    expected = denoise(expected, broadband="savgol", win=5, savgol_order=2)
                elif option in (1, 4):
                    expected = denoise(oriented, impulsive="median",
                                       baseline="highpass" if option == 4 else None,
                                       broadband="savgol", win=7, savgol_order=2,
                                       **({"hp_fc": 0.1, "causal": False} if option == 4 else {}))
                else:
                    expected = oriented
                np.testing.assert_allclose(track.preprocess(rec).epochs["emg"],
                                           expected.epochs["emg"].swapaxes(-1, -2))

    def test_cv_splits_and_heldout_perturbation(self):
        for option in range(1, 7):
            with self.subTest(option=option):
                track = FoldSafePreprocessingTrack(option=option)
                records = track.smoke(n_subjects=3, n_reps=3, win_per_rep=1)
                groups = np.concatenate([np.repeat(r.group, len(r.labels)) for r in records])
                reps = np.concatenate([r.meta["rep"] for r in records])
                fit_sets = []
                original = track._fold_features

                def checked(windows, labels, fs, tr, te):
                    self.assertFalse(set(tr).intersection(te))
                    if len(set(groups[tr])) == 1 and len(set(groups[te])) == 1 and groups[tr[0]] == groups[te[0]]:
                        self.assertFalse(set(reps[tr]).intersection(reps[te]))
                        fit_sets.append("within")
                    else:
                        self.assertFalse(set(groups[tr]).intersection(groups[te]))
                        fit_sets.append("new")
                    xtr, xte = original(windows, labels, fs, tr, te)
                    altered = windows.copy()
                    altered[te] *= 1000
                    changed_train, _ = original(altered, labels, fs, tr, te)
                    np.testing.assert_allclose(xtr, changed_train)
                    return xtr, xte

                track._fold_features = checked
                track.evaluate_modes(records, cfg={"loso_max_groups": 12})
                self.assertEqual(set(fit_sets), {"within", "new"})

    def test_flat_channels_remain_finite(self):
        zeros = np.zeros((2, 20, 3))
        result = GuardedAverageReference().fit(zeros).transform(zeros)
        self.assertTrue(np.isfinite(result).all())

    def test_unsafe_inherited_entry_points_are_disabled(self):
        with self.assertRaises(NotImplementedError):
            FoldSafePreprocessingTrack().build_dataset([])


if __name__ == "__main__":
    unittest.main()
