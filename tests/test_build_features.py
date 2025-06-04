import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np

from utils.build_dataset import build_features


def test_build_features_basic():
    # Create deterministic sample data
    rng = pd.date_range("2021-01-01", periods=60, freq="h")
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
    assert "body_pct" in features.columns
    assert "upper_wick_pct" in features.columns
    assert "willr_14" in features.columns
    assert "stochrsi_14" in features.columns
    assert "aroon_up_14" in features.columns
    assert "aroon_down_14" in features.columns
    assert pd.api.types.is_numeric_dtype(features["return_1h"])
    assert pd.api.types.is_numeric_dtype(features["rsi_14"])


def test_build_features_unsupervised():
    rng = pd.date_range("2021-01-01", periods=64, freq="h")
    base = np.linspace(100, 164, num=64)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.5,
            "volume": np.linspace(1000, 1063, num=64),
        },
        index=rng,
    )

    features = build_features(df, unsupervised=True)

    assert "wavelet_c1" in features.columns
    assert any(col.startswith("ae_feat") for col in features.columns)


def test_build_features_higher_intervals():
    rng = pd.date_range("2021-01-01", periods=60, freq="h")
    base = np.linspace(100, 160, num=60)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1,
            "low": base - 1,
            "close": base + 0.5,
            "volume": np.linspace(1000, 1060, num=60),
        },
        index=rng,
    )

    features = build_features(df, higher_intervals=["4H"])

    assert "rsi_14_4H" in features.columns
    assert "close_ma_3_4H" in features.columns
