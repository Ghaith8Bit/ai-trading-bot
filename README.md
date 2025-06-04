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

