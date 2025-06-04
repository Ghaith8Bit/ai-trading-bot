import pandas as pd
import numpy as np
from unittest.mock import patch

from utils.build_dataset import feature_selection


def _dummy_data(n_samples=30, n_features=8):
    rng = pd.date_range("2021-01-01", periods=n_samples, freq="h")
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=rng,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, size=n_samples), index=rng)
    return X, y


def test_feature_selection_cpu():
    X, y = _dummy_data()
    selected = feature_selection(X, y, n_features=5, task="classification", use_gpu=False)
    assert len(selected) == 5


def test_feature_selection_gpu_tree_method():
    X, y = _dummy_data()
    calls = {}

    class DummyXGB:
        def __init__(self, *args, **kwargs):
            calls["kwargs"] = kwargs
            self.params = kwargs
            self.feature_importances_ = np.arange(X.shape[1])

        def fit(self, X, y):
            self.feature_importances_ = np.ones(X.shape[1])
            return self

        def get_params(self, deep=False):
            return self.params

        def set_params(self, **params):
            self.params.update(params)
            return self

    with patch("xgboost.XGBClassifier", DummyXGB):
        selected = feature_selection(X, y, n_features=4, task="classification", use_gpu=True)

    assert calls["kwargs"]["tree_method"] == "gpu_hist"
    assert len(selected) == 4


def test_feature_selection_permutation_pruning():
    X, y = _dummy_data()

    class DummyRF:
        def fit(self, X, y):
            pass

    class DummyResult:
        importances_mean = np.array([0.1, 0.05, 0.001, 0.2, 0.3, 0.0, 0.04, 0.02])

    with patch("utils.build_dataset.RandomForestClassifier", return_value=DummyRF()):
        with patch("utils.build_dataset.permutation_importance", return_value=DummyResult()):
            with patch("utils.build_dataset.RFE") as DummyRFE:
                DummyRFE.return_value.fit.return_value = None
                DummyRFE.return_value.support_ = np.array([True] * X.shape[1])
                selected = feature_selection(
                    X,
                    y,
                    n_features=8,
                    task="classification",
                    importance_method="permutation",
                    importance_threshold=0.05,
                )

    assert len(selected) == 4
