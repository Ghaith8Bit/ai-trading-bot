import os
import pandas as pd
import numpy as np
import warnings
import time
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
    mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")


def build_features(
    df_raw: pd.DataFrame,
    min_periods: int = 24,
    unsupervised: bool = False,
    higher_intervals: list[str] | None = None,
    extra_data: dict[str, pd.DataFrame] | None = None,
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
    """
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
            df = df.join(features_to_add)

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

    for window in [2, 4, 8, 12, 24, 48, 72]:
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
            df = df.join(feats, how="left")

    # ======================
    # Momentum Indicators
    # ======================
    momentum_cols = {}
    for w in [7, 14, 21]:
        momentum_cols[f"rsi_{w}"] = RSIIndicator(df["close"], window=w).rsi()
        stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=w)
        momentum_cols[f"stoch_k_{w}"] = stoch.stoch()
        momentum_cols[f"stoch_d_{w}"] = stoch.stoch_signal()

    for w in [10, 14, 20]:
        momentum_cols[f"roc_{w}"] = ROCIndicator(df["close"], window=w).roc()

    for w in [7, 14, 21]:
        momentum_cols[f"willr_{w}"] = WilliamsRIndicator(
            df["high"], df["low"], df["close"], lbp=w
        ).williams_r()
        momentum_cols[f"stochrsi_{w}"] = StochRSIIndicator(df["close"], window=w).stochrsi()

    for fast, slow in [(12, 26), (10, 20), (8, 16)]:
        macd = MACD(df["close"], window_slow=slow, window_fast=fast)
        momentum_cols[f"macd_{fast}_{slow}"] = macd.macd()
        momentum_cols[f"macd_signal_{fast}_{slow}"] = macd.macd_signal()
        momentum_cols[f"macd_hist_{fast}_{slow}"] = macd.macd_diff()

    df = df.assign(**momentum_cols)

    # ======================
    # Trend Indicators
    # ======================
    trend_cols = {}
    for w in [14, 20, 28]:
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=w)
        trend_cols[f"adx_{w}"] = adx.adx()
        trend_cols[f"di_plus_{w}"] = adx.adx_pos()
        trend_cols[f"di_minus_{w}"] = adx.adx_neg()
        trend_cols[f"cci_{w}"] = CCIIndicator(df["high"], df["low"], df["close"], window=w).cci()

    for w in [14, 20, 28]:
        aroon = AroonIndicator(df["high"], df["low"], window=w)
        trend_cols[f"aroon_up_{w}"] = aroon.aroon_up()
        trend_cols[f"aroon_down_{w}"] = aroon.aroon_down()

    df = df.assign(**trend_cols)

    # ======================
    # Volatility Indicators
    # ======================
    vol_ind_cols = {}
    for window in [10, 20, 50]:
        bb = BollingerBands(df["close"], window=window, window_dev=2)
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        vol_ind_cols[f"bb_upper_{window}"] = upper
        vol_ind_cols[f"bb_lower_{window}"] = lower
        vol_ind_cols[f"bb_width_{window}"] = (upper - lower) / upper.replace(0, np.nan)

    for w in [7, 14, 21]:
        vol_ind_cols[f"atr_{w}"] = AverageTrueRange(
            df["high"], df["low"], df["close"], window=w
        ).average_true_range()

        kc = KeltnerChannel(df["high"], df["low"], df["close"], window=w)
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        vol_ind_cols[f"kc_upper_{w}"] = kc_upper
        vol_ind_cols[f"kc_lower_{w}"] = kc_lower
        vol_ind_cols[f"kc_width_{w}"] = (kc_upper - kc_lower) / kc_upper.replace(0, np.nan)

    df = df.assign(**vol_ind_cols)

    # ======================
    # Volume Indicators
    # ======================
    vol_cols = {
        "obv": OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume(),
        "adi": AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"]).acc_dist_index(),
    }

    for w in [14, 20]:
        vol_cols[f"mfi_{w}"] = MFIIndicator(
            df["high"], df["low"], df["close"], df["volume"], window=w
        ).money_flow_index()

    for w in [12, 24, 48]:
        vol_roll_mean = df["volume"].rolling(w).mean()
        vol_roll_std = df["volume"].rolling(w).std().replace(0, 1e-6)
        vol_cols[f"volume_z_{w}"] = (df["volume"] - vol_roll_mean) / vol_roll_std
        vol_cols[f"volume_roc_{w}"] = df["volume"].pct_change(w)

    vol_cols["vpd_5"] = (
        (df["volume"] - df["volume"].rolling(5).mean())
        * (df["close"] - df["close"].rolling(5).mean())
    )
    vol_cols["vpd_20"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        * (df["close"] - df["close"].rolling(20).mean())
    )

    df = df.assign(**vol_cols)

    # ======================
    # Optional Unsupervised Features
    # ======================
    if unsupervised:
        try:
            import pywt

            coeffs = pywt.swt(df["close"], "db1", level=2)
            for idx, (_, cD) in enumerate(coeffs, start=1):
                df[f"wavelet_c{idx}"] = cD
        except Exception as e:
            print(f"⚠️ Wavelet transform failed: {e}")

        try:
            ae_window = 10
            close_series = df["close"]
            window_matrix = np.column_stack([
                close_series.shift(i) for i in range(ae_window)
            ])
            mask = ~np.isnan(window_matrix).any(axis=1)
            scaler_ae = StandardScaler()
            X_train = scaler_ae.fit_transform(window_matrix[mask])

            ae = MLPRegressor(
                hidden_layer_sizes=(3,),
                max_iter=200,
                random_state=42,
                solver="lbfgs",
            )
            ae.fit(X_train, X_train)

            window_all = np.column_stack([
                close_series.shift(i).fillna(method="bfill").fillna(method="ffill").values
                for i in range(ae_window)
            ])
            X_all_scaled = scaler_ae.transform(window_all)
            hidden = np.maximum(0, np.dot(X_all_scaled, ae.coefs_[0]) + ae.intercepts_[0])

            for j in range(hidden.shape[1]):
                df[f"ae_feat{j+1}"] = hidden[:, j]
        except Exception as e:
            print(f"⚠️ Autoencoder feature extraction failed: {e}")

    # ======================
    # Time & Cyclical Features
    # ======================
    time_cols = {
        "hour": df.index.hour,
        "dow": df.index.dayofweek,
        "month": df.index.month,
    }

    for col, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        time_cols[f"{col}_sin"] = np.sin(2 * np.pi * time_cols[col] / period)
        time_cols[f"{col}_cos"] = np.cos(2 * np.pi * time_cols[col] / period)

    df = df.assign(**time_cols)

    # ======================
    # Advanced Interaction Features
    # ======================
    inter_cols = {
        "rsi_vol": df["rsi_14"] * df["volatility_24"],
        "macd_vol": df["macd_12_26"] * df["volatility_24"],
        "adx_vol": df["adx_14"] * df["volatility_24"],
    }

    for w in [3, 5]:
        inter_cols[f"price_rsi_div_{w}"] = (
            (df["close"].diff(w) / df["close"].shift(w).replace(0, np.nan))
            - (df["rsi_14"].diff(w) / 100)
        )
        inter_cols[f"price_macd_div_{w}"] = (
            (df["close"].diff(w) / df["close"].shift(w).replace(0, np.nan))
            - df["macd_12_26"].diff(w)
        )

    df = df.assign(**inter_cols)

    # ======================
    # Volatility Regime (One-Hot)
    # ======================
    vol_series = df["volatility_24"].copy()
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
    for window in [6, 12, 24]:
        roll_cols = {}
        for col in ["rsi_14", "macd_12_26", "volume"]:
            roll_cols[f"{col}_ma_{window}"] = df[col].rolling(window).mean()
            roll_cols[f"{col}_std_{window}"] = df[col].rolling(window).std()
        df = df.assign(**roll_cols)

    # ======================
    # Market Regime Detection via Bayesian GMM
    # ======================
    regime_features = ["return_1h", "volatility_24", "rsi_14", "volume_z_24"]
    regime_data = df[regime_features].dropna()

    if len(regime_data) > 1000:
        try:
            from sklearn.mixture import BayesianGaussianMixture

            bgm = BayesianGaussianMixture(
                n_components=5,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                random_state=42,
                max_iter=500,
            )
            df["regime"] = -1
            df.loc[regime_data.index, "regime"] = bgm.fit_predict(regime_data)
            df["regime_change"] = (
                df["regime"].diff().abs().gt(0).astype(int)
            )
            df = pd.get_dummies(df, columns=["regime"], prefix="regime")
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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill(limit=2).dropna()

    print(f"✅ Built {len(df.columns)} features | {len(df)} samples")
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
    """Creates binary labels for long-only strategy"""
    df = df.copy()
    future_high = df["high"].shift(-horizon).rolling(horizon, min_periods=1).max()
    
    volatility = df["volatility_24"].clip(lower=0.001)
    upper_threshold = df["close"] * (1 + 1.0 * volatility)
    
    long_condition = (future_high > upper_threshold) & (df["rsi_14"] > 40)
    
    if "volume_z_24" in df.columns:
        vol_z = df["volume_z_24"]
        long_condition &= (vol_z > -0.5)
    
    df["y_class"] = 0
    df.loc[long_condition, "y_class"] = 1
    
    return df

def feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 40,
    task: str = "classification",  # "classification" or "regression"
    use_gpu: bool = False,
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
    """
    start_time = time.time()
    print(f"🔍 Starting feature selection on {X.shape[1]} features for {task}...")

    # 1) Remove low-variance features
    var_thresh = VarianceThreshold(threshold=0.01)
    X_var = var_thresh.fit_transform(X)
    kept_mask = var_thresh.get_support()
    X_filtered = pd.DataFrame(X_var, columns=X.columns[kept_mask], index=X.index)
    print(f"🧹 Variance threshold: {X.shape[1] - X_filtered.shape[1]} features removed")

    # 2) Remove high-correlation features (>0.95)
    corr_matrix = X_filtered.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_filtered.drop(columns=high_corr, inplace=True, errors="ignore")
    print(f"🧹 Correlation filter: {len(high_corr)} features removed")

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
    selector.fit(X_filtered[top_mi_features], y)
    rfe_features = X_filtered[top_mi_features].columns[selector.support_]

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
            cv=3,
            n_jobs=-1
        )
        sfs.fit(X_filtered[rfe_features], y)
        final_features = rfe_features[sfs.get_support()]
    else:
        final_features = rfe_features

    print(f"⏱️ Feature selection completed in {time.time() - start_time:.2f}s")
    print(f"🎯 Final feature count: {len(final_features)}")
    return final_features



def generate_dataset(
    raw_path: str,
    output_dir: str,
    version: str = "v1",
    task: str = "classification",
    horizon: int = 3,
    clean: bool = True,
    use_gpu: bool = False,
    extra_data: dict[str, pd.DataFrame] | None = None,
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
    """
    # 1. Load and prepare data
    df = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
    print(f"✅ Loaded {len(df)} rows from {raw_path}")
    
    # 2. Feature engineering
    df = build_features(df, extra_data=extra_data)
    
    # 3. Create labels based on task
    if task == "classification":
        df = create_labels_classification(df, horizon)
        label_col = "y_class"
    elif task == "regression":
        df = create_labels_regression(df, horizon)
        label_col = "y_ratio"
    else:
        raise ValueError("Task must be 'classification' or 'regression'")
    
    # 4. Clean data
    if clean:
        initial_count = len(df)
        df = df[df[label_col].notna()]
        print(f"🧹 Removed {initial_count - len(df)} rows with NaN labels")
    
    # 5. Prepare features and labels
    feature_cols = [c for c in df.columns if c not in 
                   {"open", "high", "low", "close", "volume", "y_class", "y_tp", "y_sl", "y_ratio"}]
    X = df[feature_cols]
    
    if task == "classification":
        y = df[["y_class"]]
    else:
        y = df[["y_tp", "y_sl", "y_ratio"]]
    
    # 6. Handle missing values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    # 7. Feature selection
    if task == "regression":
        y_selection = y["y_ratio"]
        selected_features = feature_selection(
            X,
            y_selection,
            n_features=40,
            task="regression",
            use_gpu=use_gpu,
        )
    else:  # classification
        y_selection = y["y_class"]
        selected_features = feature_selection(
            X,
            y_selection,
            n_features=40,
            task="classification",
            use_gpu=use_gpu,
        )

    X_sel = X[selected_features]
    
    # Save selected feature names
    os.makedirs(output_dir, exist_ok=True)
    feature_name_path = os.path.join(output_dir, f"selected_features_{version}.csv")
    pd.Series(selected_features).to_csv(feature_name_path, index=False, header=False)
    print(f"💾 Saved selected feature names to {feature_name_path}")
    
    # 8. PCA Transformation (with preservation)
    if len(selected_features) > 30:
        # Apply PCA with standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
        
        pca = PCA(n_components=0.95, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Save transformation artifacts
        dump(scaler, os.path.join(output_dir, f"scaler_{version}.joblib"))
        dump(pca, os.path.join(output_dir, f"pca_{version}.joblib"))
        
        # Create readable PCA features
        pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_final = pd.DataFrame(X_pca, index=X.index, columns=pca_cols)
        
        # Save component mapping
        components_df = pd.DataFrame(
            pca.components_,
            columns=selected_features,
            index=pca_cols
        )
        components_df.to_csv(os.path.join(output_dir, f"pca_components_{version}.csv"))
        
        # Save feature mapping
        feature_mapper = {
            "feature_set": "pca",
            "original_features": list(selected_features),
            "pca_components": pca_cols,
            "scaler": f"scaler_{version}.joblib",
            "pca": f"pca_{version}.joblib"
        }
        dump(feature_mapper, os.path.join(output_dir, f"feature_mapper_{version}.joblib"))
        print(f"📊 PCA reduced to {len(pca_cols)} components (95% variance)")
    else:
        X_final = X_sel
        # Save feature mapping for non-PCA case
        feature_mapper = {
            "feature_set": "original",
            "features": list(selected_features),
            "imputation": "replace_inf_and_fillna(0)"
        }
        dump(feature_mapper, os.path.join(output_dir, f"feature_mapper_{version}.joblib"))
        print("📊 Using original features without PCA")
    
    # 9. Save dataset
    X_path = os.path.join(output_dir, f"X_{version}.parquet")
    y_path = os.path.join(output_dir, f"y_{version}.parquet")
    
    X_final.to_parquet(X_path)
    y.to_parquet(y_path)
    
    print(f"✅ Dataset saved to {output_dir}")
    print(f"📊 Final shape: {X_final.shape} features, {y.shape} labels")
    
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
    
    dump(feature_reference, os.path.join(output_dir, f"feature_reference_{version}.joblib"))
    print(f"📝 Saved comprehensive feature reference for deployment")

# For deployment
def load_feature_mapper(output_dir: str, version: str):
    """Load feature mapper for deployment"""
    mapper_path = os.path.join(output_dir, f"feature_mapper_{version}.joblib")
    return load(mapper_path)

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
        scaler = load(os.path.join(output_dir, feature_mapper["scaler"]))
        pca = load(os.path.join(output_dir, feature_mapper["pca"]))
        
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
