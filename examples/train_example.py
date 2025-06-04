import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import dump


def main(data_dir: str, task: str, version: str):
    base_path = Path(data_dir) / task
    X = pd.read_parquet(base_path / f"X_{version}.parquet")
    y = pd.read_parquet(base_path / f"y_{version}.parquet")

    if task == "classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y.squeeze())
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        print(f"Train accuracy: {acc:.4f}")
    else:
        model = GradientBoostingRegressor()
        model.fit(X, y.squeeze())
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        print(f"Train MSE: {mse:.4f}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{task}_model_{version}.joblib"
    dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train example model")
    parser.add_argument("--data-dir", default="data/processed", help="Base dataset directory")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--version", default="v1", help="Dataset version")
    args = parser.parse_args()
    main(args.data_dir, args.task, args.version)
