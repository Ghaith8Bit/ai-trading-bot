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
- [Labels](#labels)
- [Example Training](#example-training)

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
)
```

### GPU Acceleration
Set `use_gpu=True` when calling `generate_dataset` to enable faster feature
selection. This requires a CUDA capable GPU, the NVIDIA toolkit, and optionally
RAPIDS packages such as `cuml` and `cupy`.

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
After creating a processed dataset, train a simple model with
`examples/train_example.py`:

```bash
python examples/train_example.py --task classification
```

The script saves the trained model under `models/` and prints basic metrics.

## Training the Classification Model

For a more complete example that trains both a logistic regression and an optional XGBoost model, run:

```bash
python examples/train_classification.py
```

The script expects the processed classification dataset in `data/processed/classification/`.  After training it reports accuracy, precision, recall, F1 and ROC-AUC scores for each model.  The logistic regression classifier is saved to `models/classification_model_v1.joblib`.

