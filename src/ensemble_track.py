# The fixed RF-time / LDA-baseline ensemble for the course adapter.

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline

from classifiers import CV_CONFIG, LDATrack
from emg_ninapro import EMGNinaproTrack
from features import FeatureTrack
from preprocessing import filter_windows


class FilteredEnsembleTrack(EMGNinaproTrack):
    # Use the iteration-31 recipe with the inherited CV, training and inference methods

    SUPPORTED_CFG_KEYS = EMGNinaproTrack.SUPPORTED_CFG_KEYS | {"preprocess"}

    def __init__(self):
        super().__init__({**CV_CONFIG, "imbalance": "balanced", "preprocess": "median-savgol"})
        self.time_features = FeatureTrack("time_40")

    def preprocess(self, rec, cfg=None):
        recipe = self._cfg(cfg)["preprocess"]
        if recipe != "median-savgol":
            raise ValueError("The filtered ensemble requires median-savgol preprocessing.")
        return filter_windows(rec, recipe)

    def extract_features(self, rec, cfg=None):
        rf_features, _, _ = self.time_features.extract_features(rec)
        lda_features, labels, group = super().extract_features(rec, cfg)

        if rf_features.shape != (len(lda_features), 40) or lda_features.shape[1] != 50:
            raise ValueError("Expected 40 RF and 50 LDA features for each DB1 window.")

        return np.column_stack((rf_features, lda_features)), labels, group

    def baseline(self, cfg=None):
        cfg = self._cfg(cfg)
        if cfg["select"] != "none" or cfg["imbalance"] != "balanced":
            raise ValueError("The validated ensemble uses all features and balances only the RF.")

        rf_columns = ColumnTransformer([("time_40", "passthrough", slice(0, 40))])
        lda_columns = ColumnTransformer([("baseline_50", "passthrough", slice(40, 90))])

        rf = make_pipeline(rf_columns, super().baseline(cfg))
        lda = make_pipeline(lda_columns, LDATrack("baseline").baseline())

        return VotingClassifier(
            [("rf", rf), ("lda", lda)],
            voting="soft",
            weights=(0.5, 0.5),
        )