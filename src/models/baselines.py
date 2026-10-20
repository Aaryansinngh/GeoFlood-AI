"""
src/models/baselines.py

Machine Learning Baseline Models for Flood Susceptibility:
1. Logistic Regression (Linear baseline with class weighting)
2. Random Forest Classifier (Non-linear ensemble with feature importance)
"""

from typing import Dict, Any, List
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> LogisticRegression:
    """Train Logistic Regression baseline with balanced class weights."""
    model = LogisticRegression(
        penalty=params.get("penalty", "l2"),
        C=params.get("C", 1.0),
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 42),
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> RandomForestClassifier:
    """Train Random Forest baseline with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 12),
        min_samples_split=params.get("min_samples_split", 10),
        min_samples_leaf=params.get("min_samples_leaf", 5),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importances(rf_model: RandomForestClassifier, feature_names: List[str]) -> Dict[str, float]:
    """Extract and sort Gini feature importances from trained Random Forest."""
    importances = rf_model.feature_importances_
    sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {name: float(imp) for name, imp in sorted_pairs}
