import argparse
from pathlib import Path
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

from utils.build_dataset import prepare_inference_data


def run_backtest(model_path: str, dataset_path: str, csv_path: str, version: str, threshold: float) -> None:
    model = load(model_path)

    df_raw = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    X = prepare_inference_data(df_raw, dataset_path, version)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = probs[:, 1] if probs.ndim == 2 else probs.ravel()
    else:
        preds = model.predict(X)

    df = df_raw.loc[X.index].copy()
    df["prediction"] = preds
    df["signal"] = (df["prediction"] > threshold).astype(int)
    df["future_return"] = df["close"].pct_change().shift(-1)
    df["strategy_return"] = df["future_return"] * df["signal"]
    df.dropna(subset=["future_return"], inplace=True)

    df["cumulative"] = (1 + df["strategy_return"].fillna(0)).cumprod() - 1

    win_trades = (df.loc[df["signal"] == 1, "strategy_return"] > 0).sum()
    total_trades = int(df["signal"].sum())
    win_ratio = win_trades / total_trades if total_trades > 0 else 0.0
    cum_return = df["cumulative"].iloc[-1]

    print(f"Total trades: {total_trades}")
    print(f"Win ratio: {win_ratio:.2f}")
    print(f"Cumulative return: {cum_return:.4f}")

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    df[["cumulative"]].plot(title="Strategy Cumulative Return")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(plots_dir / "backtest_equity_curve.png")
    plt.close()

    df.to_csv(plots_dir / "backtest_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple probability threshold backtest")
    parser.add_argument("--model-path", default="models/classification_model_v1.joblib", help="Trained classification model")
    parser.add_argument("--dataset-path", default="data/processed/classification", help="Processed dataset directory")
    parser.add_argument("--csv-path", default="data/raw/BTCUSDT_1h.csv", help="Historical OHLCV CSV")
    parser.add_argument("--version", default="v1", help="Dataset version tag")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold to enter a trade")
    args = parser.parse_args()

    run_backtest(args.model_path, args.dataset_path, args.csv_path, args.version, args.threshold)
