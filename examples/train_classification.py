from pathlib import Path
import pandas as pd
from joblib import dump
from utils import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "data/processed/classification"
    X = pd.read_parquet(base / "X_v1.parquet")
    y = pd.read_parquet(base / "y_v1.parquet").squeeze()

    X = X.sort_index()
    y = y.loc[X.index]
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Optional permutation importance pruning after feature selection
    selected = feature_selection(
        X_train,
        y_train,
        n_features=40,
        task="classification",
        importance_method="permutation",
        importance_threshold=0.01,
        ml_logger="mlflow",
    )
    X_train = X_train[selected]
    X_test = X_test[selected]

    lr = LogisticRegression(max_iter=1000)
    # Cross-validation on the training portion
    cv_metrics = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        scores = cross_val_score(
            lr,
            X_train,
            y_train,
            cv=5,
            scoring=metric,
        )
        cv_metrics[metric] = scores.mean()

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
    print("LogisticRegression 5-Fold CV Metrics:")
    for k, v in cv_metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
    print("LogisticRegression Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
    print("LogisticRegression Test Confusion Matrix:")
    print(confusion_matrix(y_test, lr_test_preds))
    print("LogisticRegression Test Classification Report:")
    print(classification_report(y_test, lr_test_preds))

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, lr_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="LogisticRegression")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(plots_dir / "roc_curve.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, lr_test_proba)
    plt.figure()
    plt.plot(recall, precision, label="LogisticRegression")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(plots_dir / "precision_recall_curve.png")
    plt.close()

    cm = confusion_matrix(y_test, lr_test_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(plots_dir / "confusion_matrix.png")
    plt.close()

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
