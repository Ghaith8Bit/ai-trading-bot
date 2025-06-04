import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

from utils.build_dataset import build_features


def test_build_features_basic():
    # Create deterministic sample data
    rng = pd.date_range("2021-01-01", periods=60, freq="H")
    base = np.linspace(100, 160, num=60)
    df = pd.DataFrame({
        "open": base,
        "high": base + 1,
        "low": base - 1,
        "close": base + 0.5,
        "volume": np.linspace(1000, 1060, num=60),
    }, index=rng)

    features = build_features(df)

    assert "return_1h" in features.columns
    assert "rsi_14" in features.columns
    assert pd.api.types.is_numeric_dtype(features["return_1h"])
    assert pd.api.types.is_numeric_dtype(features["rsi_14"])
