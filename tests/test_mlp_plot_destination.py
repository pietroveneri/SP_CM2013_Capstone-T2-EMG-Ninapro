import importlib.util
from pathlib import Path

import pandas as pd


def test_plot_writes_to_requested_directory(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "src/MLP/plot_envelope_cnn.py"
    spec = importlib.util.spec_from_file_location("mlp_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    default = tmp_path / "default"
    requested = tmp_path / "requested"
    default.mkdir()
    requested.mkdir()
    monkeypatch.setattr(module, "OUTPUT", default)
    for mode in module.MODES:
        pd.DataFrame({"true": ["g01", "g02"], "predicted": ["g01", "g01"],
                      "p_g01": [0.8, 0.6], "p_g02": [0.2, 0.4]}).to_csv(
            requested / f"{mode}_predictions.csv", index=False
        )
        pd.DataFrame({"subject": ["S1"], "macro_f1": [0.5]}).to_csv(
            requested / f"{mode}_subject_scores.csv", index=False
        )

    module.plot_results(requested)

    assert {p.name for p in requested.glob("*.png")} == {
        "cnn_confusion.png", "cnn_macro_f1.png", "cnn_roc_auc.png"
    }
    assert not list(default.iterdir())
