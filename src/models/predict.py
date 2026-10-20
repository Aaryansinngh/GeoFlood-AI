"""
src/models/predict.py

Inference utilities for GeoFlood AI.
Loads saved models and scalers without retraining, and outputs
calibrated susceptibility probabilities for arbitrary feature sets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd
import torch

from src.models.nn import FloodSusceptibilityNN, predict_proba_nn


class GeoFloodPredictor:
    def __init__(self, models_dir: Union[str, Path] = "models"):
        self.models_dir = Path(models_dir)
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.lr_model = joblib.load(self.models_dir / "logistic_regression.joblib")
        self.rf_model = joblib.load(self.models_dir / "random_forest.joblib")

        with open(self.models_dir / "feature_names.json", "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        with open(self.models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # PyTorch NN
        self.nn_model = FloodSusceptibilityNN(input_dim=len(self.feature_names))
        state_dict = torch.load(self.models_dir / "neural_network.pt", weights_only=True)
        self.nn_model.load_state_dict(state_dict)
        self.nn_model.eval()

    def predict(self, df_features: pd.DataFrame, model_type: str = "neural_network") -> np.ndarray:
        """
        Run inference using the specified model.
        Returns predicted probabilities in [0, 1].
        """
        X = df_features[self.feature_names].values
        if model_type == "logistic_regression":
            X_scaled = self.scaler.transform(X)
            return self.lr_model.predict_proba(X_scaled)[:, 1]
        elif model_type == "random_forest":
            return self.rf_model.predict_proba(X)[:, 1]
        elif model_type == "neural_network":
            X_scaled = self.scaler.transform(X)
            return predict_proba_nn(self.nn_model, X_scaled)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from 'neural_network', 'random_forest', 'logistic_regression'.")

    def predict_point(self, feature_dict: Dict[str, float], model_type: str = "neural_network") -> float:
        """Single point inference."""
        row_df = pd.DataFrame([feature_dict])
        probs = self.predict(row_df, model_type=model_type)
        return float(probs[0])
