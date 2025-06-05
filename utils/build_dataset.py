import os
import warnings
import time
import hashlib
from typing import Callable

import pandas as pd
import numpy as np
from joblib import dump, load

from ta.volatility import (
    BollingerBands,
    AverageTrueRange,
    KeltnerChannel
)
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    ROCIndicator,
    WilliamsRIndicator,
    StochRSIIndicator,
)
from ta.trend import (
    MACD,
    ADXIndicator,
    CCIIndicator,
    AroonIndicator,
)
from ta.volume import (
    OnBalanceVolumeIndicator,
    AccDistIndexIndicator,
    MFIIndicator
)

from sklearn.feature_selection import (
    mutual_info_classif,
    RFE,
    VarianceThreshold,
    SequentialFeatureSelector,
    mutual_info_regression,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")


def _validate_path(path: str, base_dir: str) -> str:
    """Return absolute path if it resides within base_dir, else raise."""
    abs_base = os.path.abspath(base_dir)
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(abs_base + os.sep):
        raise ValueError(f"Path {path} is outside of {base_dir}")
    return abs_path


def _sha256(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_hash(path: str) -> None:
    """Write sha256 hash next to the file."""
    with open(path + ".sha256", "w") as f:
        f.write(_sha256(path))


def dump_joblib(obj, path: str) -> None:
    dump(obj, path)
    _write_hash(path)


def load_joblib(path: str, base_dir: str):
    abs_path = _validate_path(path, base_dir)
    hash_file = abs_path + ".sha256"
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            expected = f.read().strip()
        actual = _sha256(abs_path)
        if expected != actual:
            raise ValueError(f"Hash mismatch for {abs_path}")
    return load(abs_path)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    df.to_parquet(path)
    _write_hash(path)


def save_csv(data, path: str, **kwargs) -> None:
    data.to_csv(path, **kwargs)
    _write_hash(path)


def _generate_windows(base_windows, df=None, scheme=None):
    """Return sorted and unique window sizes based on the given scheme.

    Parameters
    ----------
    base_windows : list[int]
        Default set of windows to fall back to.
    df : pd.DataFrame, optional
        DataFrame used for volatility based calculations.
    scheme : str or callable, optional
        ``"fibonacci"`` to use fibonacci numbers up to ``max(base_windows)`` or
        ``"volatility"`` to scale windows based on recent volatility. A callable
        is expected to return an iterable of window sizes.
    """

    if scheme is None:
        windows = list(base_windows)
    elif callable(scheme):
        try:
            windows = list(scheme(base_windows, df))
        except TypeError:
            windows = list(scheme(base_windows))
    elif scheme == "fibonacci":
        max_w = max(base_windows)
        fib = [1, 2]
        while fib[-1] < max_w:
            fib.append(fib[-1] + fib[-2])
        windows = [w for w in fib if w <= max_w]
    elif scheme == "volatility":
        if df is None or "close" not in df.columns:
            windows = list(base_windows)
        else:
            vol = df["close"].pct_change().rolling(24).std().mean()
            factor = 1 if pd.isna(vol) else float(min(max(vol * 50, 0.5), 2.0))
            windows = [int(max(1, round(w * factor))) for w in base_windows]
    else:
        windows = list(base_windows)

    if df is not None:
        max_len = len(df)
        windows = [w for w in windows if w <= max_len]

    return sorted(set(int(w) for w in windows if w > 0))


def build_features(
    df_raw: pd.DataFrame,
    min_periods: int = 24,
    unsupervised: bool = False,
    higher_intervals: list[str] | None = None,
    extra_data: dict[str, pd.DataFrame] | None = None,
    cyclical: str = "sin",
    rbf_sigma: float = 1.0,
    use_gpu: bool = False,
    window_scheme: str | Callable | None = None,
) -> pd.DataFrame:
    """
    Enhanced feature engineering (long-only) — constructs a wide range of technical
    indicators, rolling stats, regime labels and cyclical features. Returns a
    DataFrame indexed by timestamp with all numeric columns (no raw OHLCV except
    volume is used to compute indicators). If ``unsupervised`` is ``True``,
    additional components from a wavelet transform and a tiny autoencoder on the
    closing price are appended as features.

    Parameters
    ----------
    df_raw:
        Raw OHLCV dataframe indexed by timestamp.
    min_periods:
        Minimum periods for rolling calculations.
    unsupervised:
        Whether to add wavelet/autoencoder features.
    higher_intervals:
        Optional list of higher time-frame intervals (e.g. ``["4H", "1D"]``) to
        resample the raw data and add coarse features.
    extra_data:
        Optional dictionary of additional market data dataframes keyed by name.
    cyclical:
        "sin" for sine/cosine encoding, "rbf" for radial basis functions or
        "both" to include all cyclical features.
    rbf_sigma:
        Width parameter for radial basis function time features.
    use_gpu:
        Convert the dataframe to ``cudf`` and use RAPIDS rolling operations
        where possible.
    window_scheme:
        Optional scheme for generating window sizes. Can be ``"fibonacci"``,
        ``"volatility"`` or a custom callable.
    """
    if use_gpu:
        try:
            import cudf
        except ImportError as e:
            raise ImportError(
                "cudf is required for GPU acceleration"
            ) from e
        df = cudf.DataFrame.from_pandas(df_raw.copy())
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
    else:
        df = df_raw.copy()
        df = df[~df.index.duplicated(keep="first")]
        df.sort_index(inplace=True)

    # Validate presence of required columns
    required_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure volume column exists
    if "volume" not in df.columns:
        df["volume"] = 1.0  # default to 1 if volume is missing

    # ======================
    # Higher Interval Features
    # ======================
    if higher_intervals:
        hi_frames = []
        for interval in higher_intervals:
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            resampled = df_raw.resample(interval).agg(agg)

            suffix = f"_{interval}"

            resampled[f"close_ma_3{suffix}"] = (
                resampled["close"].rolling(3, min_periods=1).mean()
            )
            resampled[f"close_ma_6{suffix}"] = (
                resampled["close"].rolling(6, min_periods=1).mean()
            )
            resampled[f"rsi_14{suffix}"] = RSIIndicator(
                resampled["close"], window=14
            ).rsi()

            features_to_add = resampled[[
                f"close_ma_3{suffix}",
                f"close_ma_6{suffix}",
                f"rsi_14{suffix}",
            ]]
            features_to_add = features_to_add.reindex(df.index, method="ffill")
            if use_gpu:
                import cudf
                features_to_add = cudf.DataFrame.from_pandas(features_to_add)
            hi_frames.append(features_to_add)

        if hi_frames:
            if use_gpu:
                import cudf
                df = df.join(cudf.concat(hi_frames, axis=1))
            else:
                df = df.join(pd.concat(hi_frames, axis=1))

    # ======================
    # Core Price Transformations
    # ======================
    price_cols = {
        "hl2": (df["high"] + df["low"]) / 2,
        "hlc3": (df["high"] + df["low"] + df["close"]) / 3,
        "ohlc4": (df["open"] + df["high"] + df["low"] + df["close"]) / 4,
    }
    df = df.assign(**price_cols)

    # ======================
    # Candlestick Patterns
    # ======================
    price_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]).abs() / price_range
    df["upper_wick_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / price_range
    df["lower_wick_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / price_range

    df["doji"] = (df["body_pct"] < 0.1).astype(int)
    df["bull_engulf"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["open"] <= df["close"].shift(1)) &
        (df["close"] >= df["open"].shift(1))
    ).astype(int)
    df["bear_engulf"] = (
        (df["close"] < df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["open"] >= df["close"].shift(1)) &
        (df["close"] <= df["open"].shift(1))
    ).astype(int)

    # ======================
    # Returns & Volatility
    # ======================
    r1 = df["close"].pct_change(1)
    ret_vol_cols = {"return_1h": r1}

    for window in _generate_windows([2, 4, 8, 12, 24, 48, 72], df, window_scheme):
        current_min_periods = min(min_periods, window)
        ret_vol_cols[f"return_{window}h"] = df["close"].pct_change(window)
        ret_vol_cols[f"volatility_{window}"] = (
            r1.rolling(window, min_periods=current_min_periods).std()
        )
        ret_vol_cols[f"close_ma_{window}"] = (
            df["close"].rolling(window, min_periods=current_min_periods).mean()
        )
        ret_vol_cols[f"volume_ma_{window}"] = (
            df["volume"].rolling(window, min_periods=current_min_periods).mean()
        )

    df = df.assign(**ret_vol_cols)

    # Advanced volatility
    adv_vol_cols = {
        "volatility_ewm": r1.ewm(span=24, adjust=False, min_periods=min_periods).std(),
        "garch_vol": np.sqrt((r1 ** 2).ewm(alpha=0.1, min_periods=10).mean()),
    }
    df = df.assign(**adv_vol_cols)

    # ======================
    # Additional Market Data
    # ======================
    if extra_data:
        extra_frames = []
        for name, extra_df in extra_data.items():
            edf = extra_df.copy()
            edf = edf[~edf.index.duplicated(keep="first")]
            edf.sort_index(inplace=True)
            if "close" not in edf.columns:
                raise ValueError(f"Extra dataframe '{name}' missing 'close' column")

            feats = pd.DataFrame(index=edf.index)
            feats[f"return_1h_{name}"] = edf["close"].pct_change(1)
            feats[f"return_24h_{name}"] = edf["close"].pct_change(24)
            feats[f"close_ma_24_{name}"] = (
                edf["close"].rolling(24, min_periods=min_periods).mean()
            )

            feats = feats.reindex(df.index)
            if use_gpu:
                import cudf
                feats = cudf.DataFrame.from_pandas(feats)
            extra_frames.append(feats)

        if extra_frames:
            if use_gpu:
                import cudf
                df = df.join(cudf.concat(extra_frames, axis=1), how="left")
            else:
                df = df.join(pd.concat(extra_frames, axis=1), how="left")

    # ======================
    # Momentum Indicators
    # ======================
    momentum_cols = {}
    df_pd = df.to_pandas() if use_gpu else df
    for w in _generate_windows([7, 14, 21], df, window_scheme):
        momentum_cols[f"rsi_{w}"] = RSIIndicator(df_pd["close"], window=w).rsi()
        stoch = StochasticOscillator(df_pd["high"], df_pd["low"], df_pd["close"], window=w)
        momentum_cols[f"stoch_k_{w}"] = stoch.stoch()
        momentum_cols[f"stoch_d_{w}"] = stoch.stoch_signal()

    for w in _generate_windows([10, 14, 20], df, window_scheme):
        momentum_cols[f"roc_{w}"] = ROCIndicator(df_pd["close"], window=w).roc()

    for w in _generate_windows([7, 14, 21], df, window_scheme):
        momentum_cols[f"willr_{w}"] = WilliamsRIndicator(
            df_pd["high"], df_pd["low"], df_pd["close"], lbp=w
        ).williams_r()
        momentum_cols[f"stochrsi_{w}"] = StochRSIIndicator(df_pd["close"], window=w).stochrsi()

    macd_pairs_base = [(12, 26), (10, 20), (8, 16)]
    macd_windows = _generate_windows(
        [w for pair in macd_pairs_base for w in pair],
        df,
        window_scheme,
    )
    macd_pairs = [
        (macd_windows[i], macd_windows[i + 1])
        for i in range(0, len(macd_windows) - 1, 2)
    ]
    if not macd_pairs:
        macd_pairs = macd_pairs_base
    for fast, slow in macd_pairs:
        macd = MACD(df_pd["close"], window_slow=slow, window_fast=fast)
        momentum_cols[f"macd_{fast}_{slow}"] = macd.macd()
        momentum_cols[f"macd_signal_{fast}_{slow}"] = macd.macd_signal()
        momentum_cols[f"macd_hist_{fast}_{slow}"] = macd.macd_diff()

    if use_gpu:
        import cudf
        df = df.join(cudf.DataFrame.from_pandas(pd.DataFrame(momentum_cols, index=df_pd.index)))
    else:
        df = df.assign(**momentum_cols)

    # ======================
    # Trend Indicators
    # ======================
    trend_cols = {}
    df_pd = df.to_pandas() if use_gpu else df
    for w in _generate_windows([14, 20, 28], df, window_scheme):
        adx = ADXIndicator(df_pd["high"], df_pd["low"], df_pd["close"], window=w)
        trend_cols[f"adx_{w}"] = adx.adx()
        trend_cols[f"di_plus_{w}"] = adx.adx_pos()
        trend_cols[f"di_minus_{w}"] = adx.adx_neg()
        trend_cols[f"cci_{w}"] = CCIIndicator(df_pd["high"], df_pd["low"], df_pd["close"], window=w).cci()

    for w in _generate_windows([14, 20, 28], df, window_scheme):
        aroon = AroonIndicator(df_pd["high"], df_pd["low"], window=w)
        trend_cols[f"aroon_up_{w}"] = aroon.aroon_up()
        trend_cols[f"aroon_down_{w}"] = aroon.aroon_down()

    if use_gpu:
        import cudf
        df = df.join(cudf.DataFrame.from_pandas(pd.DataFrame(trend_cols, index=df_pd.index)))
    else:
        df = df.assign(**trend_cols)

    # ======================
    # Volatility Indicators
    # ======================
    vol_ind_cols = {}
    df_pd = df.to_pandas() if use_gpu else df
    for window in _generate_windows([10, 20, 50], df, window_scheme):
        bb = BollingerBands(df_pd["close"], window=window, window_dev=2)
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        vol_ind_cols[f"bb_upper_{window}"] = upper
        vol_ind_cols[f"bb_lower_{window}"] = lower
        vol_ind_cols[f"bb_width_{window}"] = (upper - lower) / upper.replace(0, np.nan)

    for w in _generate_windows([7, 14, 21], df, window_scheme):
        vol_ind_cols[f"atr_{w}"] = AverageTrueRange(
            df_pd["high"], df_pd["low"], df_pd["close"], window=w
        ).average_true_range()

        kc = KeltnerChannel(df_pd["high"], df_pd["low"], df_pd["close"], window=w)
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        vol_ind_cols[f"kc_upper_{w}"] = kc_upper
        vol_ind_cols[f"kc_lower_{w}"] = kc_lower
        vol_ind_cols[f"kc_width_{w}"] = (kc_upper - kc_lower) / kc_upper.replace(0, np.nan)

    if use_gpu:
        import cudf
        df = df.join(cudf.DataFrame.from_pandas(pd.DataFrame(vol_ind_cols, index=df_pd.index)))
    else:
        df = df.assign(**vol_ind_cols)

    # ======================
    # Volume Indicators
    # ======================
    vol_cols = {}
    df_pd = df.to_pandas() if use_gpu else df
    vol_cols["obv"] = OnBalanceVolumeIndicator(df_pd["close"], df_pd["volume"]).on_balance_volume()
    vol_cols["adi"] = AccDistIndexIndicator(df_pd["high"], df_pd["low"], df_pd["close"], df_pd["volume"]).acc_dist_index()

    for w in _generate_windows([14, 20], df, window_scheme):
        vol_cols[f"mfi_{w}"] = MFIIndicator(
            df_pd["high"], df_pd["low"], df_pd["close"], df_pd["volume"], window=w
        ).money_flow_index()

    for w in _generate_windows([12, 24, 48], df, window_scheme):
        vol_roll_mean = df_pd["volume"].rolling(w).mean()
        vol_roll_std = df_pd["volume"].rolling(w).std().replace(0, 1e-6)
        vol_cols[f"volume_z_{w}"] = (df_pd["volume"] - vol_roll_mean) / vol_roll_std
        vol_cols[f"volume_roc_{w}"] = df_pd["volume"].pct_change(w)

    vol_cols["vpd_5"] = (
        (df_pd["volume"] - df_pd["volume"].rolling(5).mean())
        * (df_pd["close"] - df_pd["close"].rolling(5).mean())
    )
    vol_cols["vpd_20"] = (
        (df_pd["volume"] - df_pd["volume"].rolling(20).mean())
        * (df_pd["close"] - df_pd["close"].rolling(20).mean())
    )
    if use_gpu:
        import cudf
        df = df.join(cudf.DataFrame.from_pandas(pd.DataFrame(vol_cols, index=df_pd.index)))
    else:
        df = df.assign(**vol_cols)

    # ======================
    # Optional Unsupervised Features
    # ======================
    if unsupervised:
        df_pd = df.to_pandas() if use_gpu else df
        unsup_df = pd.DataFrame(index=df_pd.index)

        if len(df_pd) >= 32:
            try:
                import pywt

                coeffs = pywt.swt(df_pd["close"], "db1", level=2)
                for idx, (_, cD) in enumerate(coeffs, start=1):
                    unsup_df[f"wavelet_c{idx}"] = cD
            except Exception as e:
                print(f"⚠️ Wavelet transform failed: {e}")
        else:
            print("⚠️ Skipping wavelet transform: insufficient data")

        ae_window = 10
        min_samples = 32
        if len(df_pd) >= max(ae_window + min_samples, 32):
            try:
                close_series = df_pd["close"]
                window_matrix = np.column_stack([
                    close_series.shift(i) for i in range(ae_window)
                ])

                hidden_size = 3
                features = np.full((len(df), hidden_size), np.nan, dtype=np.float32)

                for i in range(ae_window + min_samples, len(df_pd) + 1):
                    X_hist = window_matrix[: i - 1]
                    mask = ~np.isnan(X_hist).any(axis=1)
                    X_train = X_hist[mask]
                    if len(X_train) < min_samples:
                        continue

                    scaler_ae = StandardScaler()
                    X_scaled = scaler_ae.fit_transform(X_train)

                    ae = MLPRegressor(
                        hidden_layer_sizes=(hidden_size,),
                        max_iter=200,
                        random_state=42,
                        solver="lbfgs",
                    )
                    ae.fit(X_scaled, X_scaled)

                    current = window_matrix[i - 1 : i]
                    if np.isnan(current).any():
                        continue
                    current_scaled = scaler_ae.transform(current)
                    hidden = np.maximum(
                        0, np.dot(current_scaled, ae.coefs_[0]) + ae.intercepts_[0]
                    )
                    features[i - 1] = hidden[0]

                for j in range(hidden_size):
                    unsup_df[f"ae_feat{j+1}"] = features[:, j]
            except Exception as e:
                print(f"⚠️ Autoencoder feature extraction failed: {e}")
        else:
            print("⚠️ Skipping autoencoder features: insufficient data")

        if not unsup_df.empty:
            if use_gpu:
                import cudf
                df = cudf.concat([df, cudf.DataFrame.from_pandas(unsup_df)], axis=1)
            else:
                df = pd.concat([df, unsup_df], axis=1)

    # ======================
    # Time & Cyclical Features
    # ======================
    time_cols = {
        "hour": df.index.hour,
        "dow": df.index.dayofweek,
        "month": df.index.month,
    }


    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        if cyclical in ("sin", "both"):
            time_cols[f"{col}_sin"] = np.sin(2 * np.pi * time_cols[col] / period)
            time_cols[f"{col}_cos"] = np.cos(2 * np.pi * time_cols[col] / period)

        if cyclical in ("rbf", "both"):
            centers = [0, period / 2]
            for i, c in enumerate(centers, start=1):
                time_cols[f"{col}_rbf{i}"] = np.exp(
                    -0.5 * ((time_cols[col] - c) / rbf_sigma) ** 2
                )

    df = df.assign(**time_cols)

    # ======================
    # Advanced Interaction Features
    # ======================
    rsi_w = max(_generate_windows([14], df, window_scheme))
    vol_w = max(_generate_windows([24], df, window_scheme))
    macd_fast = max(_generate_windows([12], df, window_scheme))
    macd_slow = max(_generate_windows([26], df, window_scheme))
    rsi_col = f"rsi_{rsi_w}"
    vol_col = f"volatility_{vol_w}"
    macd_col = f"macd_{macd_fast}_{macd_slow}"
    adx_col = f"adx_{rsi_w}"

    inter_cols = {
        "rsi_vol": df[rsi_col] * df[vol_col],
        "macd_vol": df.get(macd_col, pd.Series(index=df.index)) * df[vol_col],
        "adx_vol": df[adx_col] * df[vol_col],
    }

    for w in [3, 5]:
        inter_cols[f"price_rsi_div_{w}"] = (
            (df["close"].diff(w) / df["close"].shift(w).replace(0, np.nan))
            - (df[rsi_col].diff(w) / 100)
        )
        inter_cols[f"price_macd_div_{w}"] = (
            (df["close"].diff(w) / df["close"].shift(w).replace(0, np.nan))
            - df.get(macd_col, pd.Series(index=df.index)).diff(w)
        )

    df = df.assign(**inter_cols)

    # ======================
    # Volatility Regime (One-Hot)
    # ======================
    vol_series = df[vol_col].copy()
    regime_cols = {
        "vol_regime_low": 0,
        "vol_regime_medium": 0,
        "vol_regime_high": 0,
    }

    if len(vol_series.dropna()) > 100:
        low_threshold = vol_series.expanding(min_periods=100).quantile(0.25)
        high_threshold = vol_series.expanding(min_periods=100).quantile(0.75)

        regime_cols["vol_regime_low"] = (vol_series < low_threshold).fillna(0).astype(int)
        regime_cols["vol_regime_medium"] = (
            (vol_series >= low_threshold) & (vol_series <= high_threshold)
        ).fillna(0).astype(int)
        regime_cols["vol_regime_high"] = (vol_series > high_threshold).fillna(0).astype(int)

    df = df.assign(**regime_cols)

    # ======================
    # Lagged Features (numeric only)
    # ======================
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    lag_cols = {}
    for feature in numerical_cols:
        if feature.startswith("vol_regime"):
            continue
        series = df[feature]
        for lag in [1, 2, 3, 6, 12]:
            lag_cols[f"{feature}_lag{lag}"] = series.shift(lag)

    df = pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)

    # ======================
    # Rolling-Window Summaries
    # ======================
    roll_cols = {}
    for window in [6, 12, 24]:
        for col in [rsi_col, macd_col, "volume"]:
            if col in df.columns:
                roll_cols[f"{col}_ma_{window}"] = df[col].rolling(window).mean()
                roll_cols[f"{col}_std_{window}"] = df[col].rolling(window).std()
    df = df.assign(**roll_cols)

    # ======================
    # Market Regime Detection via Bayesian GMM
    # ======================
    volume_z_window = max(_generate_windows([24], df, window_scheme))
    regime_features = [
        "return_1h",
        vol_col,
        rsi_col,
        f"volume_z_{volume_z_window}",
    ]
    regime_data = df[regime_features].dropna()

    if len(regime_data) > 1000:
        try:
            from sklearn.mixture import BayesianGaussianMixture

            window = 1000
            df["regime"] = -1
            bgm = BayesianGaussianMixture(
                n_components=5,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                random_state=42,
                max_iter=500,
            )

            regimes = []
            for i, idx in enumerate(regime_data.index):
                end = i + 1
                start = max(0, end - window)
                hist = regime_data.iloc[start:end]
                if hist.shape[0] < 2:
                    regimes.append(-1)
                    continue
                labels = bgm.fit_predict(hist)
                regimes.append(labels[-1])

            df.loc[regime_data.index, "regime"] = regimes
            df["regime_change"] = df["regime"].diff().abs().gt(0).astype(int)
            regime_numeric = df["regime"].copy()
            df = pd.get_dummies(df, columns=["regime"], prefix="regime")
            df["regime"] = regime_numeric
        except Exception as e:
            print(f"⚠️ Market regime detection failed: {e}")

    # ======================
    # Cleanup
    # ======================
    intermediate_cols = (
        ["hl2", "hlc3", "ohlc4", "hour", "dow", "month"]
        + [c for c in df.columns if c.startswith("bb_") or c.startswith("kc_")]
    )
    df.drop(columns=intermediate_cols, inplace=True, errors="ignore")

    # Replace infinities with NaN, then forward-fill up to 2 bars, then drop remaining NaNs

    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    for col in all_nan_cols:
        df[col] = 0.0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill(limit=2).dropna()

    # Convert numeric columns to float32 for smaller memory footprint
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)

    print(f"✅ Built {len(df.columns)} features | {len(df)} samples")
    if use_gpu:
        df = df.to_pandas()
    return df

def create_labels_regression(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Creates TP, SL, and combined ratio targets"""
    df = df.copy()

    future_high = df["high"].shift(-horizon).rolling(horizon, min_periods=1).max()
    future_low = df["low"].shift(-horizon).rolling(horizon, min_periods=1).min()
    close = df["close"]

    df["y_tp"] = (future_high / close) - 1.0
    df["y_sl"] = (future_low / close) - 1.0
    df["y_ratio"] = df["y_tp"] / (abs(df["y_sl"]) + 1e-6)

    return df

def create_labels_classification(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Creates binary labels for long-only strategy."""
    df = df.copy()
    future_high = df["high"].shift(-horizon).rolling(horizon, min_periods=1).max()

    if "volatility_24" in df.columns:
        volatility = df["volatility_24"].clip(lower=0.001)
    else:
        r1 = df["close"].pct_change(1)
        volatility = r1.rolling(24, min_periods=1).std().clip(lower=0.001)

    if "rsi_14" not in df.columns:
        from ta.momentum import RSIIndicator

        df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()

    upper_threshold = df["close"] * (1 + volatility)
    long_condition = (future_high > upper_threshold) & (df["rsi_14"] > 40)

    if "volume_z_24" in df.columns:
        vol_z = df["volume_z_24"]
        long_condition &= vol_z > -0.5

    df["y_class"] = 0
    df.loc[long_condition, "y_class"] = 1

    return df

def feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 40,
    task: str = "classification",  # "classification" or "regression"
    use_gpu: bool = False,
    importance_method: str | None = None,
    importance_threshold: float = 0.0,
    log_path: str | None = None,
    group_by_time: bool = False,
    ml_logger: str | None = None,
    tracking_uri: str | None = None,
) -> pd.Index:
    """
    Robust feature selection pipeline that works for both classification and regression.
    
    Steps:
      1) Variance threshold
      2) High-correlation filter
      3) Mutual information prescreen (top 100)
      4) RFE to exactly n_features
      5) Optional forward SFS if RFE still > n_features*1.5
    
    Returns the Index of selected feature names.

    Args:
      - X: DataFrame of shape (n_samples, n_candidate_features)
      - y: Series of labels (binary/discrete for classification; continuous for regression)
      - n_features: number of features to select via RFE/SFS
      - task: "classification" or "regression"
      - use_gpu: enable GPU accelerated estimators
      - importance_method: optional "shap" or "permutation" to compute
        post-selection feature importances
      - importance_threshold: prune features with importance below this value
      - log_path: optional path to save RFE ranking CSV
      - group_by_time: use time-series aware cross-validation
    """
    cv_obj = TimeSeriesSplit(n_splits=3) if group_by_time else None

    start_time = time.time()
    print(f"🔍 Starting feature selection on {X.shape[1]} features for {task}...")

    mlflow_run = None
    wandb_run = None
    metrics: dict[str, float] = {}

    if ml_logger == "mlflow":
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if mlflow.active_run() is not None:
            mlflow_run = mlflow.start_run(nested=True)
        else:
            mlflow_run = mlflow.start_run()
        mlflow.log_params({"task": task, "target_features": n_features})
    elif ml_logger == "wandb":
        import wandb

        wandb_run = wandb.init(project="feature_selection", dir=tracking_uri)
        wandb_run.config.update({"task": task, "target_features": n_features})

    t0 = time.time()
    # 1) Remove low-variance features
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(X)
    kept_mask = var_thresh.get_support()
    X_filtered = pd.DataFrame(X_var, columns=X.columns[kept_mask], index=X.index)
    metrics["variance_time"] = time.time() - t0
    print(f"🧹 Variance threshold: {X.shape[1] - X_filtered.shape[1]} features removed")

    t0 = time.time()
    # 2) Remove high-correlation features (>0.95)
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_filtered.drop(columns=high_corr, inplace=True, errors="ignore")
    metrics["correlation_time"] = time.time() - t0
    print(f"🧹 Correlation filter: {len(high_corr)} features removed")

    t0 = time.time()
    # 3) Mutual Information prescreen (top 100)
    if X_filtered.shape[0] > 10000:
        sample_idx = np.random.choice(X_filtered.index, 10000, replace=False)
        X_sample = X_filtered.loc[sample_idx]
        y_sample = y.loc[sample_idx]
    else:
        X_sample = X_filtered
        y_sample = y

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "classification":
            if use_gpu:
                try:
                    from cuml.feature_selection import mutual_info_classif as gpu_mic
                    mi_scores = gpu_mic(
                        X_sample.values.astype(np.float32),
                        y_sample.values.astype(np.int32),
                    ).to_array()
                except Exception:
                    mi_scores = mutual_info_classif(
                        X_sample, y_sample, random_state=42, n_neighbors=5
                    )
            else:
                mi_scores = mutual_info_classif(
                    X_sample, y_sample, random_state=42, n_neighbors=5
                )
        else:  # regression
            if use_gpu:
                try:
                    from cuml.feature_selection import mutual_info_regression as gpu_mir
                    mi_scores = gpu_mir(
                        X_sample.values.astype(np.float32),
                        y_sample.values.astype(np.float32),
                    ).to_array()
                except Exception:
                    mi_scores = mutual_info_regression(
                        X_sample, y_sample, random_state=42, n_neighbors=5
                    )
            else:
                mi_scores = mutual_info_regression(
                    X_sample, y_sample, random_state=42, n_neighbors=5
                )

    mi_ranking = pd.Series(mi_scores, index=X_filtered.columns).sort_values(ascending=False)
    top_mi_features = mi_ranking.head(min(100, len(mi_ranking))).index
    metrics["mutual_info_time"] = time.time() - t0
    print(f"📊 MI selected {len(top_mi_features)} candidate features")

    # 4) Recursive Feature Elimination (RFE) to exactly n_features
    if task == "classification":
        if use_gpu:
            from xgboost import XGBClassifier
            estimator = XGBClassifier(
                n_estimators=100,
                tree_method="gpu_hist",
                random_state=42,
                verbosity=0,
            )
        else:
            estimator = GradientBoostingClassifier(n_estimators=50, random_state=42)
    else:
        if use_gpu:
            from xgboost import XGBRegressor
            estimator = XGBRegressor(
                n_estimators=100,
                tree_method="gpu_hist",
                random_state=42,
                verbosity=0,
            )
        else:
            estimator = GradientBoostingRegressor(n_estimators=50, random_state=42)

    selector = RFE(
        estimator,
        n_features_to_select=n_features,
        step=0.1,
        importance_getter="feature_importances_"
    )
    t0 = time.time()
    selector.fit(X_filtered[top_mi_features], y)
    rfe_features = X_filtered[top_mi_features].columns[selector.support_]
    metrics["rfe_time"] = time.time() - t0

    if log_path:
        importances = pd.Series(
            selector.estimator_.feature_importances_, index=rfe_features
        ).sort_values(ascending=False)
        df_log = importances.reset_index()
        df_log.columns = ["feature", "importance"]
        save_csv(pd.DataFrame(df_log), log_path, index=False)

    # 5) If RFE still returned too many (> 1.5 * n_features), do forward SFS
    if len(rfe_features) > n_features * 1.5:
        if task == "classification":
            sfs_estimator = RandomForestClassifier(n_estimators=25, random_state=42)
        else:
            sfs_estimator = RandomForestRegressor(n_estimators=25, random_state=42)

        sfs = SequentialFeatureSelector(
            sfs_estimator,
            n_features_to_select=n_features,
            direction="forward",
            cv=cv_obj if cv_obj is not None else 3,
            n_jobs=-1
        )
        t0 = time.time()
        sfs.fit(X_filtered[rfe_features], y)
        final_features = rfe_features[sfs.get_support()]
        metrics["sfs_time"] = time.time() - t0
    else:
        final_features = rfe_features

    # Optional post-selection importance pruning
    if importance_method:
        if task == "classification":
            imp_model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            imp_model = RandomForestRegressor(n_estimators=50, random_state=42)

        imp_model.fit(X_filtered[final_features], y)

        if importance_method == "permutation":
            result = permutation_importance(
                imp_model,
                X_filtered[final_features],
                y,
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
            )
            importances = pd.Series(result.importances_mean, index=final_features)
        elif importance_method == "shap":
            try:
                import shap

                explainer = shap.TreeExplainer(imp_model)
                shap_values = explainer.shap_values(X_filtered[final_features])
                importances = pd.Series(
                    np.abs(shap_values).mean(axis=0), index=final_features
                )
            except Exception as e:
                print(f"SHAP importance failed: {e}; falling back to permutation")
                result = permutation_importance(
                    imp_model,
                    X_filtered[final_features],
                    y,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1,
                )
                importances = pd.Series(result.importances_mean, index=final_features)
        else:
            raise ValueError("importance_method must be 'shap' or 'permutation'")

        importances.sort_values(ascending=False, inplace=True)
        print("⭐ Feature importances:")
        for feat, score in importances.items():
            print(f"  {feat}: {score:.4f}")

        if importance_threshold > 0:
            keep = importances[importances >= importance_threshold].index
            pruned = len(final_features) - len(keep)
            if pruned > 0:
                print(
                    f"🗑️ Pruned {pruned} features below {importance_threshold}"
                )
            final_features = pd.Index(keep)

    total_time = time.time() - start_time
    metrics["total_time"] = total_time
    metrics["selected_features"] = len(final_features)
    print(f"⏱️ Feature selection completed in {total_time:.2f}s")
    print(f"🎯 Final feature count: {len(final_features)}")

    if ml_logger == "mlflow":
        import mlflow

        mlflow.log_metrics(metrics)
        mlflow.end_run()
    elif ml_logger == "wandb" and wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return final_features



def generate_dataset(
    raw_path: str,
    output_dir: str,
    version: str = "v1",
    task: str = "classification",
    horizon: int = 3,
    clean: bool = True,
    use_gpu: bool = False,
    regime_target_encoding: bool = False,
    extra_data: dict[str, pd.DataFrame] | None = None,
    ml_logger: str | None = None,
    tracking_uri: str | None = None,
    window_scheme: str | Callable | None = None,
):
    """Complete dataset generation pipeline.

    Performs feature engineering, optional higher interval aggregation,
    feature selection and optional PCA.

    Args:
        raw_path: CSV path with OHLCV data
        output_dir: directory to store artifacts
        version: dataset version string
        task: "classification" or "regression"
        horizon: label horizon
        clean: drop rows with NaN labels
        use_gpu: forward to feature_selection for GPU acceleration
        extra_data: optional dictionary of additional price data passed through
            to ``build_features`` for feature augmentation.
        regime_target_encoding: apply target encoding to the ``regime`` column
            after label creation if present.
        ml_logger: "mlflow" or "wandb" to enable experiment tracking.
        tracking_uri: optional tracking URI or directory used by the logger.
        window_scheme: forwarded to ``build_features`` for custom window logic.
    """
    metrics: dict[str, float] = {}
    run_mlflow = None
    run_wandb = None

    if ml_logger == "mlflow":
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if mlflow.active_run() is not None:
            run_mlflow = mlflow.start_run(nested=True)
        else:
            run_mlflow = mlflow.start_run()
        mlflow.log_params({
            "task": task,
            "window_scheme": window_scheme,
        })
    elif ml_logger == "wandb":
        import wandb

        run_wandb = wandb.init(project="dataset_generation", dir=tracking_uri)
        run_wandb.config.update({"task": task, "window_scheme": window_scheme})

    start = time.time()
    # 1. Load and prepare data
    df = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
    metrics["load_time"] = time.time() - start
    print(f"✅ Loaded {len(df)} rows from {raw_path}")
    
    t0 = time.time()
    # 2. Feature engineering
    df = build_features(df, extra_data=extra_data, window_scheme=window_scheme)
    metrics["feature_engineering_time"] = time.time() - t0
    
    t0 = time.time()
    # 3. Create labels based on task
    if task == "classification":
        df = create_labels_classification(df, horizon)
        label_col = "y_class"
    elif task == "regression":
        df = create_labels_regression(df, horizon)
        label_col = "y_ratio"
    else:
        raise ValueError("Task must be 'classification' or 'regression'")
    metrics["label_time"] = time.time() - t0

    # 3b. Optional target encoding of market regime
    if regime_target_encoding and "regime" in df.columns:
        try:
            from category_encoders import TargetEncoder

            target_col = "y_class" if task == "classification" else "y_ratio"
            te = TargetEncoder(cols=["regime"])
            enc_start = time.time()
            encoded = te.fit_transform(df[["regime"]], df[target_col])
            df["regime_te"] = encoded["regime"]
            metrics["target_encoding_time"] = time.time() - enc_start
        except Exception as e:
            print(f"⚠️ Regime target encoding failed: {e}")
    
    # 4. Clean data
    if clean:
        clean_start = time.time()
        initial_count = len(df)
        df = df[df[label_col].notna()]
        metrics["clean_time"] = time.time() - clean_start
        print(f"🧹 Removed {initial_count - len(df)} rows with NaN labels")
    
    # 5. Prepare features and labels
    feature_cols = [c for c in df.columns if c not in
                   {"open", "high", "low", "close", "volume", "y_class", "y_tp", "y_sl", "y_ratio"}]
    if ml_logger == "mlflow":
        import mlflow
        mlflow.log_param("initial_feature_count", len(feature_cols))
    elif ml_logger == "wandb" and run_wandb is not None:
        run_wandb.config.update({"initial_feature_count": len(feature_cols)})
    X = df[feature_cols]
    
    if task == "classification":
        y = df[["y_class"]]
    else:
        y = df[["y_tp", "y_sl", "y_ratio"]]
    
    # 6. Handle missing values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # 7. Feature selection
    fs_start = time.time()
    if task == "regression":
        y_selection = y["y_ratio"]
        selected_features = feature_selection(
            X,
            y_selection,
            n_features=40,
            task="regression",
            use_gpu=use_gpu,
            ml_logger=ml_logger,
            tracking_uri=tracking_uri,
        )
    else:  # classification
        y_selection = y["y_class"]
        selected_features = feature_selection(
            X,
            y_selection,
            n_features=40,
            task="classification",
            use_gpu=use_gpu,
            ml_logger=ml_logger,
            tracking_uri=tracking_uri,
        )
    metrics["feature_selection_time"] = time.time() - fs_start

    if regime_target_encoding and "regime_te" in X.columns:
        if "regime_te" not in selected_features:
            selected_features = list(selected_features) + ["regime_te"]

    X_sel = X[selected_features]

    if ml_logger == "mlflow":
        import mlflow
        mlflow.log_param("selected_feature_count", len(selected_features))
    elif ml_logger == "wandb" and run_wandb is not None:
        run_wandb.config.update({"selected_feature_count": len(selected_features)})
    
    # Save selected feature names
    os.makedirs(output_dir, exist_ok=True)
    feature_name_path = os.path.join(output_dir, f"selected_features_{version}.csv")
    save_csv(pd.Series(selected_features), feature_name_path, index=False, header=False)
    print(f"💾 Saved selected feature names to {feature_name_path}")
    
    # 8. PCA Transformation (with preservation)
    if len(selected_features) > 30:
        pca_start = time.time()
        # Apply PCA with standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
        
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Save transformation artifacts
        dump_joblib(scaler, os.path.join(output_dir, f"scaler_{version}.joblib"))
        dump_joblib(pca, os.path.join(output_dir, f"pca_{version}.joblib"))
        
        # Create readable PCA features
        pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_final = pd.DataFrame(X_pca, index=X.index, columns=pca_cols)
        
        # Save component mapping
        components_df = pd.DataFrame(
            pca.components_,
            columns=selected_features,
            index=pca_cols
        )
        save_csv(components_df, os.path.join(output_dir, f"pca_components_{version}.csv"))
        
        # Save feature mapping
        feature_mapper = {
            "feature_set": "pca",
            "original_features": list(selected_features),
            "pca_components": pca_cols,
            "scaler": f"scaler_{version}.joblib",
            "pca": f"pca_{version}.joblib"
        }
        dump_joblib(feature_mapper, os.path.join(output_dir, f"feature_mapper_{version}.joblib"))
        metrics["pca_time"] = time.time() - pca_start
        print(f"📊 PCA reduced to {len(pca_cols)} components (95% variance)")
    else:
        X_final = X_sel
        # Save feature mapping for non-PCA case
        feature_mapper = {
            "feature_set": "original",
            "features": list(selected_features),
            "imputation": "replace_inf_and_fillna(0)"
        }
        dump_joblib(feature_mapper, os.path.join(output_dir, f"feature_mapper_{version}.joblib"))
        metrics["pca_time"] = 0.0
        print("📊 Using original features without PCA")
    
    # 9. Save dataset
    X_path = os.path.join(output_dir, f"X_{version}.parquet")
    y_path = os.path.join(output_dir, f"y_{version}.parquet")

    # Convert numeric columns to float32 before saving
    num_cols = X_final.select_dtypes(include=[np.number]).columns
    X_final[num_cols] = X_final[num_cols].astype(np.float32)
    y_num_cols = y.select_dtypes(include=[np.number]).columns
    y[y_num_cols] = y[y_num_cols].astype(np.float32)

    save_start = time.time()
    save_parquet(X_final, X_path)
    save_parquet(y, y_path)
    metrics["save_time"] = time.time() - save_start

    print(f"✅ Dataset saved to {output_dir}")
    print(f"📊 Final shape: {X_final.shape} features, {y.shape} labels")
    metrics["total_time"] = time.time() - start
    
    # 10. Create feature reference file for deployment
    feature_reference = {
        "required_inputs": ["open", "high", "low", "close", "volume"],
        "feature_engineering": "build_features() function",
        "selected_features": list(selected_features),
        "preprocessing_steps": [
            "replace([np.inf, -np.inf], np.nan)",
            "fillna(0)"
        ],
        "pca_applied": len(selected_features) > 30,
        "feature_mapper": f"feature_mapper_{version}.joblib",
        "version": version,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    if feature_reference["pca_applied"]:
        feature_reference["pca_components"] = pca_cols
        feature_reference["scaler"] = f"scaler_{version}.joblib"
        feature_reference["pca_model"] = f"pca_{version}.joblib"
    
    dump_joblib(
        feature_reference,
        os.path.join(output_dir, f"feature_reference_{version}.joblib"),
    )
    print("📝 Saved comprehensive feature reference for deployment")

    if ml_logger == "mlflow":
        import mlflow

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(feature_name_path)
        mlflow.log_artifact(os.path.join(output_dir, f"feature_reference_{version}.joblib"))
        mlflow.end_run()
    elif ml_logger == "wandb" and run_wandb is not None:
        run_wandb.log(metrics)
        run_wandb.save(feature_name_path)
        run_wandb.save(os.path.join(output_dir, f"feature_reference_{version}.joblib"))
        run_wandb.finish()

# For deployment
def load_feature_mapper(output_dir: str, version: str):
    """Load feature mapper for deployment"""
    mapper_path = os.path.join(output_dir, f"feature_mapper_{version}.joblib")
    return load_joblib(mapper_path, output_dir)

def prepare_inference_data(raw_data: pd.DataFrame, output_dir: str, version: str) -> pd.DataFrame:
    """
    Prepare data for inference using saved feature engineering artifacts
    Returns DataFrame with correct features for model input
    """
    # 1. Load feature mapper
    feature_mapper = load_feature_mapper(output_dir, version)
    
    # 2. Apply feature engineering
    df = build_features(raw_data)
    
    # 3. Select features
    if feature_mapper["feature_set"] == "pca":
        features = feature_mapper["original_features"]
    else:
        features = feature_mapper["features"]
    
    X = df[features]
    
    # 4. Apply preprocessing
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # 5. Apply transformations if PCA was used
    if feature_mapper["feature_set"] == "pca":
        scaler = load_joblib(os.path.join(output_dir, feature_mapper["scaler"]), output_dir)
        pca = load_joblib(os.path.join(output_dir, feature_mapper["pca"]), output_dir)
        
        X_scaled = scaler.transform(X)
        X_final = pd.DataFrame(
            pca.transform(X_scaled),
            index=X.index,
            columns=feature_mapper["pca_components"]
        )
    else:
        X_final = X
        
    return X_final

if __name__ == "__main__":
    # Example usage
    generate_dataset(
        raw_path="data/raw/BTCUSDT_1h.csv",
        output_dir="data/processed/classification",
        version="v1",
        task="classification",
        horizon=3
    )
    
    generate_dataset(
        raw_path="data/raw/BTCUSDT_1h.csv",
        output_dir="data/processed/regression",
        version="v1",
        task="regression",
        horizon=3
    )
