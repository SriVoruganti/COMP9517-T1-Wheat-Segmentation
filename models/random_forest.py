"""
Random Forest model for EWS pixel classification.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rf_model.pkl")


def build_model(class_weight: str = "balanced") -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )


def tune_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    param_dist = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }
    search = RandomizedSearchCV(
        build_model(), param_dist,
        n_iter=10, cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    return search.best_estimator_


def save_model(clf: RandomForestClassifier, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)
    print(f"  Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestClassifier:
    assert os.path.exists(path), f"No model found at {path}"
    return joblib.load(path)