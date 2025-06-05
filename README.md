# AI Trading Bot Dataset Utilities

This repository provides tools to transform raw cryptocurrency data into rich feature
datasets for model training. The focus is on generating technical indicators,
performing optional feature selection, and creating both classification and
regression targets.

## Table of Contents
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
  - [GPU Acceleration](#gpu-acceleration)
  - [Additional Market Data](#additional-market-data)
  - [Unsupervised Features](#unsupervised-features)
- [Time Series CV](#time-series-cv)
- [Labels](#labels)
- [Build Dataset Details](#build-dataset-details)
- [Example Training](#example-training)
- [Inference](#inference)

## Installation
Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Generation
Use `generate_dataset` from `utils.build_dataset` to build a processed dataset.
The input CSV should contain `open`, `high`, `low`, `close` and `volume` columns
indexed by timestamp.

```python
from utils.build_dataset import generate_dataset

generate_dataset(
    raw_path="data/raw/BTCUSDT_1h.csv",
    output_dir="data/processed/classification",
    version="v1",
    task="classification",
    horizon=3,
    regime_target_encoding=True,
    ml_logger="mlflow",  # or "wandb"
    tracking_uri="file:./mlruns",
)
```

When enabled, the market ``regime`` feature is encoded against the target
variable using `category_encoders.TargetEncoder`.

Set ``ml_logger`` to ``"mlflow"`` or ``"wandb"`` (with optional ``tracking_uri``)
to automatically record parameters, metrics and artifacts during dataset
creation.

### Reduced Memory Footprint
All numeric columns are stored as `float32` when datasets are generated. This
cuts the disk and memory usage roughly in half compared to `float64` storage.

### GPU Acceleration
Set `use_gpu=True` when calling `generate_dataset` **or** `build_features` to
enable GPU-accelerated rolling statistics. This requires a CUDA capable GPU,
the NVIDIA toolkit and the RAPIDS `cudf` library. Optional packages such as
`cuml` and `cupy` can further speed up feature selection.

### Additional Market Data
`build_features` and `generate_dataset` accept an `extra_data` dictionary of
additional market dataframes:

```python
extra = {"btc": btc_df, "sp500": sp500_df}
df = build_features(df, extra_data=extra)
```

Each dataframe must use a timestamp index and provide a `close` column. Basic
indicators (returns and moving averages) are merged using suffixes like
`return_1h_btc` or `close_ma_24_sp500`.

### Unsupervised Features
Passing `unsupervised=True` to `build_features` adds wavelet and autoencoder
components derived from the closing price:

```python
features = build_features(df, unsupervised=True)
```

### Time Series CV
Use `group_by_time=True` (alias `time_cv`) with `feature_selection` to apply
`TimeSeriesSplit` during recursive elimination and the sequential selector. This
makes feature ranking respect chronological order.

## Labels
During dataset generation, the functions `create_labels_classification` and
`create_labels_regression` in
[`utils/build_dataset.py`](utils/build_dataset.py) produce these targets:

- **`y_class`** – binary label for "buy vs. hold" classification. It becomes `1`
  when the next `future_high` exceeds `close * (1 + volatility)` and the
  14‑period RSI is greater than 40.
- **`y_tp`** and **`y_sl`** – take‑profit and stop‑loss returns calculated from
  `future_high` and `future_low` relative to the current close price.
- **`y_ratio`** – ratio of `y_tp` to the absolute value of `y_sl`; used as the
  primary regression target.

## Example Training
After creating a processed dataset, you can train a simple model using the
provided script or the accompanying notebook.

### Using the script

```bash
python examples/train_classification.py
```

### Using the notebook

Open [`examples/train_classification.ipynb`](examples/train_classification.ipynb)
for an interactive walkthrough that mirrors the script and includes plots of the
training metrics.

Both approaches save the trained model under `models/` and display basic
metrics.

## Build Dataset Details

For a full explanation of every feature engineered by `build_dataset.py`, see
[the dedicated reference](docs/build_dataset.md).

## Inference

After training a model you can generate predictions on new data using
`examples/predict.py`. The script expects the trained model, the processed
dataset directory and a raw CSV file containing the OHLCV data to score.

```bash
python examples/predict.py \
    --model-path models/classification/bundle_v1.joblib \
    --dataset-path data/processed/classification \
    --csv-path data/raw/BTCUSDT_1h.csv \
    --version v1 \
    --output-csv predictions.csv
```

Behind the scenes `prepare_inference_data` loads the saved feature mapping via
`load_feature_mapper` to apply the same feature engineering pipeline and optional
PCA transformation used during dataset generation. This ensures the model
receives data in exactly the same format as it was trained on.
