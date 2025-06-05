import argparse
from pathlib import Path
import pandas as pd
from joblib import load

from utils.build_dataset import prepare_inference_data


def main(model_path: str, dataset_path: str, version: str, csv_path: str, output_csv: str) -> None:
    model = load(model_path)
    df_raw = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    X = prepare_inference_data(df_raw, dataset_path, version)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] > 1:
            preds = probs[:, 1]
        else:
            preds = probs.ravel()
        result = pd.DataFrame({"timestamp": X.index, "probability": preds})
    else:
        labels = model.predict(X)
        result = pd.DataFrame({"timestamp": X.index, "prediction": labels})

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on OHLCV data")
    parser.add_argument("--model-path", required=True, help="Path to joblib model")
    parser.add_argument("--dataset-path", required=True, help="Processed dataset directory")
    parser.add_argument("--version", default="v1", help="Dataset version tag")
    parser.add_argument("--csv-path", required=True, help="Raw OHLCV CSV for inference")
    parser.add_argument("--output-csv", default="predictions.csv", help="Where to save predictions")
    args = parser.parse_args()

    main(args.model_path, args.dataset_path, args.version, args.csv_path, args.output_csv)
