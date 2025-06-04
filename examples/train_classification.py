from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def main():
    base = Path("data/processed/classification")
    X = pd.read_parquet(base / "X_v1.parquet")
    y = pd.read_parquet(base / "y_v1.parquet").squeeze()

    X = X.sort_index()
    y = y.loc[X.index]
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    print("LogisticRegression Metrics:")
    print(f"  Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print(f"  Precision: {precision_score(y_test, lr_preds):.4f}")
    print(f"  Recall: {recall_score(y_test, lr_preds):.4f}")
    print(f"  F1: {f1_score(y_test, lr_preds):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        print("XGBClassifier Metrics:")
        print(f"  Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
        print(f"  Precision: {precision_score(y_test, xgb_preds):.4f}")
        print(f"  Recall: {recall_score(y_test, xgb_preds):.4f}")
        print(f"  F1: {f1_score(y_test, xgb_preds):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, xgb_proba):.4f}")
    except Exception as e:
        print(f"XGBClassifier not available: {e}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dump(lr, models_dir / "classification_model_v1.joblib")
    print(f"Saved model to {models_dir / 'classification_model_v1.joblib'}")


if __name__ == "__main__":
    main()
