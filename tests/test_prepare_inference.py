import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.build_dataset import generate_dataset, prepare_inference_data  # noqa: E402


def test_prepare_inference_basic(tmp_path):
    n = 200
    rng = pd.date_range("2021-01-01", periods=n, freq="h")
    np.random.seed(0)
    base = 100 + np.cumsum(np.random.randn(n))
    df_train = pd.DataFrame({
        "timestamp": rng,
        "open": base,
        "high": base + np.random.rand(n),
        "low": base - np.random.rand(n),
        "close": base + 0.5 * np.random.randn(n),
        "volume": 1000 + np.random.rand(n) * 10,
    })

    raw_csv = tmp_path / "raw.csv"
    df_train.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "out"

    generate_dataset(
        raw_path=str(raw_csv),
        output_dir=str(out_dir),
        version="tst",
        task="classification",
    )

    # New data for inference
    rng_new = pd.date_range("2021-02-01", periods=n, freq="h")
    np.random.seed(1)
    base_new = 120 + np.cumsum(np.random.randn(n))
    df_new = pd.DataFrame(
        {
            "open": base_new,
            "high": base_new + np.random.rand(n),
            "low": base_new - np.random.rand(n),
            "close": base_new + 0.5 * np.random.randn(n),
            "volume": 1000 + np.random.rand(n) * 10,
        },
        index=rng_new,
    )

    X_inf = prepare_inference_data(df_new, str(out_dir), "tst")
    X_train = pd.read_parquet(out_dir / "X_tst.parquet")

    assert list(X_inf.columns) == list(X_train.columns)
    assert X_inf.shape == X_train.shape
