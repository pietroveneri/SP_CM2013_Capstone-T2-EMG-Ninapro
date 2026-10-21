"""Exercise the dataset split using temporary MAT files, without real hold-out data."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emg_ninapro import EMGNinaproTrack, HOLDOUT_SUBJECTS, N_CH, STEP, WIN


def write_recording(root, subject):
    path = root / f"s{subject}" / f"S{subject}_A1_E1.mat"
    path.parent.mkdir(parents=True, exist_ok=True)
    n = WIN + STEP
    savemat(path, {
        "emg": np.ones((n, N_CH)),
        "restimulus": np.ones((n, 1)),
        "rerepetition": np.ones((n, 1)),
    })
    return path


class DatasetLayoutTests(unittest.TestCase):
    def test_mixed_root_skips_holdout_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_recording(root, 1)
            for subject in HOLDOUT_SUBJECTS:
                path = root / "holdout" / f"s{subject}" / f"S{subject}_A1_E1.mat"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"Must never be parsed during development")
            recs = EMGNinaproTrack().load(root)
        self.assertEqual([rec.group for rec in recs], ["S1"])
        self.assertEqual(recs[0].epochs["emg"].shape, (2, WIN, N_CH))
        self.assertEqual(recs[0].labels.tolist(), ["g01", "g01"])

    def test_holdout_directory_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject in HOLDOUT_SUBJECTS:
                write_recording(root, subject)
            track = EMGNinaproTrack()
            self.assertEqual(track.load(root), [])
            recs = track.load(root, include_holdout=True)
        self.assertEqual({rec.group for rec in recs}, set(track.meta.holdout_ids))

    def test_download_reuses_subject_folders_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject in range(1, 23):
                write_recording(root, subject)
            with patch("urllib.request.urlopen") as request:
                EMGNinaproTrack().download(root, subset=range(1, 28))
            request.assert_not_called()
            self.assertFalse(any((root / f"s{s}").exists() for s in HOLDOUT_SUBJECTS))

    def test_new_downloads_use_subject_folders_and_skip_holdout(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("S22_A1_E1.mat", b"downloaded fixture")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("urllib.request.urlopen") as request:
                request.return_value.read.return_value = archive.getvalue()
                result = EMGNinaproTrack().download(str(root), subset=range(22, 28))
            self.assertEqual(result, str(root))
            self.assertEqual(request.call_count, 1)
            self.assertTrue(request.call_args.args[0].endswith("s22.zip"))
            self.assertEqual((root / "s22" / "S22_A1_E1.mat").read_bytes(), b"downloaded fixture")
            self.assertFalse(any((root / f"s{s}").exists() for s in HOLDOUT_SUBJECTS))


if __name__ == "__main__":
    unittest.main()
