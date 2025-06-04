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

