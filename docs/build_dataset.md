# build_dataset.py Reference

This document explains the dataset generation utilities provided in
[`utils/build_dataset.py`](../utils/build_dataset.py). The module assembles
technical indicators, rolling statistics and various helper features used to
train trading models.

## Overview

The main entry point is `generate_dataset`, which reads raw OHLCV data,
builds features through `build_features`, creates targets and applies optional
feature selection and dimensionality reduction. The resulting feature matrix and
labels are written as Parquet files alongside a small feature mapping used at
inference time.

The following sections detail each function and the categories of features that
are produced.

## `build_features`

`build_features(df_raw, ...)` accepts a DataFrame of at least `open`, `high`,
`low`, `close` and an optional `volume` column. It can optionally resample data
at higher intervals, include extra market data, compute unsupervised features
and encode cyclical time information. Important arguments are:

- `min_periods` – minimum look‑back when computing rolling statistics.
- `unsupervised` – adds wavelet components and a tiny autoencoder on price.
- `higher_intervals` – list of higher‑timeframe intervals to resample and join.
- `extra_data` – dictionary of other price series to merge as basic indicators.
- `cyclical` – `"sin"`, `"rbf"` or `"both"` to control cyclical encodings.
- `use_gpu` – use RAPIDS (cudf) for rolling operations when available.
- `window_scheme` – custom logic for generating rolling window sizes.

### Feature categories

1. **Higher interval aggregates** – moving averages and RSI computed on
   resampled bars when `higher_intervals` is supplied.
2. **Core price transforms** – midpoint prices such as `hl2`, `hlc3` and
   `ohlc4`.
3. **Candlestick patterns** – body and wick percentages plus simple patterns
   like `doji`, `bull_engulf` and `bear_engulf`.
4. **Returns & volatility** – percentage change of closing price over multiple
   horizons (`return_1h`, `return_24h`, ...), rolling volatility (`volatility_24`),
   moving averages of close and volume and advanced measures such as
   `volatility_ewm` and a simple GARCH approximation (`garch_vol`).
5. **Additional market data** – when `extra_data` is provided, basic returns and
   averages from each series are suffixed with the given key (e.g.
   `return_24h_btc`).
6. **Momentum indicators** – RSI, stochastic oscillators, ROC, Williams %R,
   StochRSI and multiple MACD configurations.
7. **Trend indicators** – ADX and directional movement, CCI and Aroon.
8. **Volatility indicators** – Bollinger and Keltner bands and ATR values.
9. **Volume indicators** – OBV, accumulation/distribution index, MFI, z‑scores
   and rate of change of volume plus simple price/volume divergence measures.
10. **Unsupervised components** – optional discrete wavelet coefficients and
    hidden activations from a small autoencoder built on the closing price.
11. **Time & cyclical features** – hour of day, day of week and month encoded
    either via sine/cosine pairs or radial basis functions (RBF).
12. **Interaction features** – products and divergences between RSI, MACD, ADX
    and volatility measures.
13. **Volatility regime flags** – one‑hot columns marking low, medium and high
    volatility regimes based on expanding quantiles.
14. **Lagged features** – numeric columns are shifted by 1–12 steps to provide
    simple autoregressive terms.
15. **Rolling summaries** – short moving averages and standard deviations of key
    indicators (RSI, MACD and volume).
16. **Market regime detection** – an optional Bayesian Gaussian Mixture is
    fitted on a sliding window to label market regimes and detect regime
    changes.

After feature creation intermediate helper columns are dropped, infinities are
replaced with NaN, forward filled for up to two bars and remaining NaNs are
removed. All numeric data are cast to `float32` before returning the final
DataFrame.

## Label creation

- `create_labels_classification(df, horizon)` adds `y_class`, which is `1` when
  the future high over `horizon` periods exceeds the current close price by the
  current volatility and RSI is above 40.
- `create_labels_regression(df, horizon)` adds `y_tp`/`y_sl` returns and their
  ratio `y_ratio`.

## `feature_selection`

`feature_selection(X, y, n_features, ...)` performs a multi‑step reduction of
candidate features:

1. Remove low variance features.
2. Drop highly correlated columns (>0.95).
3. Rank the remainder by mutual information and keep the top 100.
4. Apply recursive feature elimination (RFE) with a gradient boosting estimator
   (XGBoost if GPU is requested).
5. Optionally run a sequential forward selector when RFE still yields too many
   features.
6. Optional importance ranking by permutation or SHAP values can further prune
   the result.

The function returns an index of the selected feature names.

## `generate_dataset`

This convenience pipeline ties everything together:

1. Load the raw CSV containing OHLCV data.
2. Build features with `build_features` (plus any `extra_data`).
3. Create classification or regression labels.
4. Optionally target‑encode the market regime feature.
5. Clean missing values and invoke `feature_selection` to keep roughly 40
   features.
6. If more than 30 features remain, apply PCA to preserve 95% of the variance
   and save the scaler/PCA artefacts.
7. Store the resulting feature matrix `X` and label DataFrame `y` to Parquet and
   write a small `feature_reference_*.joblib` describing the feature set.

## Deployment helpers

`load_feature_mapper` and `prepare_inference_data` load the saved artefacts and
apply the same feature engineering and optional PCA transformation to new raw
data prior to inference.

---
