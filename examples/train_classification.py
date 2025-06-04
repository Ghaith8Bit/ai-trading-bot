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
    confusion_matrix,
    classification_report,
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
    lr_train_preds = lr.predict(X_train)
    lr_train_proba = lr.predict_proba(X_train)[:, 1]
    lr_test_preds = lr.predict(X_test)
    lr_test_proba = lr.predict_proba(X_test)[:, 1]
    train_metrics = {
        "accuracy": accuracy_score(y_train, lr_train_preds),
        "precision": precision_score(y_train, lr_train_preds),
        "recall": recall_score(y_train, lr_train_preds),
        "f1": f1_score(y_train, lr_train_preds),
        "roc_auc": roc_auc_score(y_train, lr_train_proba),
    }
    test_metrics = {
        "accuracy": accuracy_score(y_test, lr_test_preds),
        "precision": precision_score(y_test, lr_test_preds),
        "recall": recall_score(y_test, lr_test_preds),
        "f1": f1_score(y_test, lr_test_preds),
        "roc_auc": roc_auc_score(y_test, lr_test_proba),
    }
    print("LogisticRegression Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
    print("LogisticRegression Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
    print("LogisticRegression Test Confusion Matrix:")
    print(confusion_matrix(y_test, lr_test_preds))
    print("LogisticRegression Test Classification Report:")
    print(classification_report(y_test, lr_test_preds))

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        xgb_train_preds = xgb.predict(X_train)
        xgb_train_proba = xgb.predict_proba(X_train)[:, 1]
        xgb_test_preds = xgb.predict(X_test)
        xgb_test_proba = xgb.predict_proba(X_test)[:, 1]

        train_metrics = {
            "accuracy": accuracy_score(y_train, xgb_train_preds),
            "precision": precision_score(y_train, xgb_train_preds),
            "recall": recall_score(y_train, xgb_train_preds),
            "f1": f1_score(y_train, xgb_train_preds),
            "roc_auc": roc_auc_score(y_train, xgb_train_proba),
        }
        test_metrics = {
            "accuracy": accuracy_score(y_test, xgb_test_preds),
            "precision": precision_score(y_test, xgb_test_preds),
            "recall": recall_score(y_test, xgb_test_preds),
            "f1": f1_score(y_test, xgb_test_preds),
            "roc_auc": roc_auc_score(y_test, xgb_test_proba),
        }
        print("XGBClassifier Train Metrics:")
        for k, v in train_metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
        print("XGBClassifier Test Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")
        print("XGBClassifier Test Confusion Matrix:")
        print(confusion_matrix(y_test, xgb_test_preds))
        print("XGBClassifier Test Classification Report:")
        print(classification_report(y_test, xgb_test_preds))
    except Exception as e:
        print(f"XGBClassifier not available: {e}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dump(lr, models_dir / "classification_model_v1.joblib")
    print(f"Saved model to {models_dir / 'classification_model_v1.joblib'}")


if __name__ == "__main__":
    main()
