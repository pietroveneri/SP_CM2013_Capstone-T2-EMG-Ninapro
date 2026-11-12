from emg_ninapro import EMGNinaproTrack


class FilterOption02(EMGNinaproTrack):
    """Identity preprocessing, matching the fold-safe option 2."""

    def preprocess(self, rec, cfg=None):
        return rec


if __name__ == "__main__":
    t = FilterOption02()
    print(t.dataset_card())
    modes = t.evaluate_modes(t.smoke())
    from sklearn.metrics import f1_score
    for name, rep in modes.items():
        print("SMOKE %-15s macroF1=%.3f kappa=%.3f (%s)" % (
            name, f1_score(rep["y_true"], rep["y_pred"], average="macro"),
            rep["cohens_kappa"], rep["split_unit"]))
        print("    spread:", rep["summary"])
