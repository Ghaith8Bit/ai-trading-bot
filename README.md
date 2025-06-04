# AI Trading Bot Dataset Generation

This repository contains utilities for building feature-rich datasets from crypto price data. Input
CSV files should provide `open`, `high`, `low`, `close` and `volume` columns indexed by timestamp.

## GPU Acceleration

The dataset pipeline can optionally leverage GPU acceleration during feature
selection. When calling `generate_dataset` set `use_gpu=True` to enable GPU
optimizations. This requires:

- A CUDA compatible GPU with drivers installed
- NVIDIA CUDA toolkit and supported libraries
- Optional RAPIDS packages (`cuml`, `cupy`) for faster mutual information
  calculation

Example:

```python
from utils.build_dataset import generate_dataset

generate_dataset(
    raw_path="data/raw/BTCUSDT_1h.csv",
    output_dir="data/processed/classification",
    version="v1",
    task="classification",
    horizon=3,
    use_gpu=True,
)
```

If `use_gpu` is disabled (default), CPU implementations are used.

### Additional Market Data

`build_features` and `generate_dataset` accept an optional `extra_data` argument:

```python
extra = {"btc": btc_df, "sp500": sp500_df}
df = build_features(df, extra_data=extra)
```

Each dataframe in `extra_data` must use a timestamp index and provide at least a
`close` column. Basic indicators—hourly and 24‑hour returns plus a 24‑hour
moving average—are computed and merged into the feature set with suffixes such
as `return_1h_btc` or `close_ma_24_sp500`. Missing timestamps are gracefully
forward‑filled during processing.

## Labels

During dataset generation, targets are created in
[`utils/build_dataset.py`](utils/build_dataset.py) by the
`create_labels_classification` and `create_labels_regression` functions.
These produce the following columns:

- **`y_class`** – binary label for "buy vs. hold" classification. It becomes `1`
  when the next `future_high` exceeds `close * (1 + volatility)` and the
  14‑period RSI is greater than 40 (with a mild volume filter when available).
- **`y_tp`** and **`y_sl`** – take‑profit and stop‑loss returns calculated from
  `future_high` and `future_low` relative to the current close price.
- **`y_ratio`** – ratio of `y_tp` to the absolute value of `y_sl`; used as the
  primary regression target.

## Unsupervised Feature Extraction

`build_features` now accepts an optional `unsupervised=True` flag to add a few
wavelet- and autoencoder-based components derived from the closing price. These
features can capture additional structure in the price series.

Example:

```python
from utils.build_dataset import build_features

df = pd.read_csv("prices.csv", parse_dates=["timestamp"], index_col="timestamp")
features = build_features(df, unsupervised=True)
```

## Example Training

After creating a processed dataset you can train a small model using
`examples/train_example.py`:

```bash
python examples/train_example.py --task classification
```

The script saves the trained model under `models/` and prints simple metrics.

## Training the Classification Model

For a more complete example that trains both a logistic regression and an optional XGBoost model, run:

```bash
python examples/train_classification.py
```

The script expects the processed classification dataset in `data/processed/classification/`.  After training it reports accuracy, precision, recall, F1 and ROC-AUC scores for each model.  The logistic regression classifier is saved to `models/classification_model_v1.joblib`.

