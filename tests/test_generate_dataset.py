import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.build_dataset import generate_dataset


def test_generate_dataset_regime_encoding(tmp_path):
    n = 1100
    rng = pd.date_range("2021-01-01", periods=n, freq="h")
    np.random.seed(0)
    base = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "timestamp": rng,
        "open": base,
        "high": base + np.random.rand(n),
        "low": base - np.random.rand(n),
        "close": base + 0.5 * np.random.randn(n),
        "volume": 1000 + np.random.rand(n) * 10,
    })

    raw_csv = tmp_path / "raw.csv"
    df.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "out"

    generate_dataset(
        raw_path=str(raw_csv),
        output_dir=str(out_dir),
        version="tst",
        task="classification",
        regime_target_encoding=True,
    )

    feat_file = out_dir / "selected_features_tst.csv"
    selected = pd.read_csv(feat_file, header=None)[0].tolist()
    assert "regime_te" in selected
