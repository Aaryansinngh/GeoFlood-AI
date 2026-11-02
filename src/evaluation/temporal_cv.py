"""
src/evaluation/temporal_cv.py

Leave-One-Flood-Event-Out (LOFEO) Cross-Validation Framework
============================================================

Scientific purpose
------------------
Standard cross-validation partitions spatially or randomly, but floods occur
as distinct temporal hydrometeorological events with idiosyncratic rainfall
hyetographs, antecedent soil moisture, and basin responses.

LOFEO evaluates temporal generalisation:
- Train on historical flood events E_1, E_2, ... E_{k-1}
- Test on an unseen future/historical flood event E_k

Strict Temporal Leakage Protection:
- Evaluates whether post-event composite variables (e.g. `rain_event_sum`)
  cause artificial performance inflation.
- Supports evaluating with strictly pre-event features (`rain_3d`, `rain_7d`,
  static terrain) vs. including post-event summary metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_predictions, find_optimal_threshold
from src.models.baselines import train_logistic_regression, train_random_forest
from src.models.nn import predict_proba_nn, train_neural_network

logger = logging.getLogger(__name__)

try:
    from src.models.gradient_boosting import (
        lightgbm_available,
        train_lightgbm,
        train_xgboost,
        xgboost_available,
    )
    _GB_AVAILABLE = True
except ImportError:
    _GB_AVAILABLE = False

FEATURE_COLUMNS = [
    "rain_event_sum",
    "rain_3d",
    "rain_7d",
    "elevation",
    "slope",
    "tpi",
    "relative_elevation",
]

PRE_EVENT_FEATURES = [
    "rain_3d",
    "rain_7d",
    "elevation",
    "slope",
    "tpi",
    "relative_elevation",
]


def _build_model_registry(params: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    params = params or {}
    registry: Dict[str, Any] = {
        "logistic_regression": params.get("logistic_regression", {"max_iter": 1000}),
        "random_forest": params.get("random_forest", {"n_estimators": 100, "max_depth": 10}),
        "neural_network": params.get("neural_network", {"hidden_size": 64, "epochs": 25, "lr": 0.001}),
    }
    if _GB_AVAILABLE:
        if xgboost_available():
            registry["xgboost"] = params.get("xgboost", {"n_estimators": 150, "max_depth": 6})
        if lightgbm_available():
            registry["lightgbm"] = params.get("lightgbm", {"n_estimators": 150, "max_depth": 6})
    return registry


def _fit_predict(
    model_name: str,
    model_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, float]:
    needs_scale = model_name in {"logistic_regression", "neural_network", "xgboost", "lightgbm"}
    if needs_scale:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_vl = sc.transform(X_val)
        X_te = sc.transform(X_test)
    else:
        X_tr, X_vl, X_te = X_train, X_val, X_test

    if model_name == "logistic_regression":
        m = train_logistic_regression(X_tr, y_train, model_params)
        vp = m.predict_proba(X_vl)[:, 1]
        tp = m.predict_proba(X_te)[:, 1]
    elif model_name == "random_forest":
        m = train_random_forest(X_tr, y_train, model_params)
        vp = m.predict_proba(X_vl)[:, 1]
        tp = m.predict_proba(X_te)[:, 1]
    elif model_name == "neural_network":
        m, _ = train_neural_network(X_tr, y_train, X_vl, y_val, model_params)
        vp = predict_proba_nn(m, X_vl)
        tp = predict_proba_nn(m, X_te)
    elif model_name == "xgboost" and _GB_AVAILABLE:
        m = train_xgboost(X_tr, y_train, model_params, X_vl, y_val)
        vp = m.predict_proba(X_vl)[:, 1]
        tp = m.predict_proba(X_te)[:, 1]
    elif model_name == "lightgbm" and _GB_AVAILABLE:
        m = train_lightgbm(X_tr, y_train, model_params, X_vl, y_val)
        vp = m.predict_proba(X_vl)[:, 1]
        tp = m.predict_proba(X_te)[:, 1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    thresh, _ = find_optimal_threshold(y_val, vp)
    return tp, thresh


class TemporalEventCV:
    """
    Leave-One-Flood-Event-Out (LOFEO) cross-validation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        val_block_frac: float = 0.20,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.model_registry = _build_model_registry(model_params)
        self.val_block_frac = val_block_frac
        self.output_dir = Path(output_dir) if output_dir else None
        self.seed = seed

        # Filter valid rows
        mask = df["flood_label"].isin([0.0, 1.0])
        for col in self.feature_columns:
            if col in df.columns:
                mask = mask & df[col].notna()
        self._df = df[mask].copy()
        self._events = sorted(self._df["event_id"].unique().tolist())
        self._available_feats = [f for f in self.feature_columns if f in self._df.columns]

        logger.info(
            "TemporalEventCV: %d valid rows across events %s, features=%s",
            len(self._df), self._events, self._available_feats
        )

    def _run_fold(self, test_event: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        test_df = self._df[self._df["event_id"] == test_event].copy()
        train_events = [e for e in self._events if e != test_event]

        if not train_events:
            logger.warning("LOFEO: Only 1 event (%s) present, cannot hold out.", test_event)
            return results

        train_full = self._df[self._df["event_id"].isin(train_events)].copy()

        # Spatial block validation split within train events to prevent leak
        unique_blocks = train_full["spatial_block_id"].unique()
        rng = np.random.RandomState(self.seed)
        n_val = max(1, int(len(unique_blocks) * self.val_block_frac))
        val_blocks = set(rng.choice(unique_blocks, size=n_val, replace=False))
        train_blocks = set(unique_blocks) - val_blocks

        train_df = train_full[train_full["spatial_block_id"].isin(train_blocks)]
        val_df = train_full[train_full["spatial_block_id"].isin(val_blocks)]

        X_train = train_df[self._available_feats].values.astype(float)
        y_train = train_df["flood_label"].values.astype(float)
        X_val = val_df[self._available_feats].values.astype(float)
        y_val = val_df["flood_label"].values.astype(float)
        X_test = test_df[self._available_feats].values.astype(float)
        y_test = test_df["flood_label"].values.astype(float)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning("Event %s: single class split, skipping.", test_event)
            return results

        for model_name, mparams in self.model_registry.items():
            try:
                test_probs, thresh = _fit_predict(
                    model_name, mparams, X_train, y_train, X_val, y_val, X_test
                )
                metrics = evaluate_predictions(y_test, test_probs, threshold=thresh)
                results.append({
                    "protocol": "LOFEO",
                    "test_event": test_event,
                    "train_events": ",".join(train_events),
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "brier_score": metrics["brier_score"],
                    "false_negative_rate": metrics["false_negative_rate"],
                    "prevalence": metrics["prevalence"],
                    "threshold": metrics["threshold"],
                    "features_used": ",".join(self._available_feats),
                    "note": "",
                })
            except Exception as exc:
                logger.error("LOFEO event=%s model=%s FAILED: %s", test_event, model_name, exc)
        return results

    def run(self) -> pd.DataFrame:
        all_results: List[Dict[str, Any]] = []
        for event in self._events:
            all_results.extend(self._run_fold(event))

        results_df = pd.DataFrame(all_results)
        if self.output_dir and not results_df.empty:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / "temporal_lofeo_results.csv"
            results_df.to_csv(out_path, index=False)
            logger.info("Saved LOFEO results to %s", out_path)
        return results_df
