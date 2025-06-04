# AI Trading Bot Dataset Generation

This repository contains utilities for building feature-rich datasets from crypto price data.

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

## Example Training

After creating a processed dataset you can train a small model using
`examples/train_example.py`:

```bash
python examples/train_example.py --task classification
```

The script saves the trained model under `models/` and prints simple metrics.

