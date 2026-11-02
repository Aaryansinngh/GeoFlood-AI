"""
src/evaluation/spatial_kfold_cv.py

Full Rotating Spatial K-Fold Cross-Validation
=============================================

Scientific purpose
------------------
The existing code uses a FIXED spatial block split (train=[0,1,3,4,6,8,9],
val=[5,11], test=[2,7,10]).  That design answers "how does the model perform
on blocks 2, 7, 10?" but cannot estimate performance variance across space.

This module implements a ROTATING K-fold: every spatial block serves exactly
once as the test partition.  This yields K independent test scores and gives
an unbiased estimate of the expected performance across the full study area.

Spatially buffered validation
------------------------------
Cells immediately adjacent to training-region boundaries suffer from spatial
autocorrelation (Moran's I leakage).  When `buffer_degrees > 0`, a ring of
cells around the boundary of each test block is dropped from the TEST set.
This produces a more conservative, more realistic performance estimate.

Design decisions
----------------
* Works on any single region's feature DataFrame.
* Does NOT require XGBoost/LightGBM (degrades gracefully).
* All thresholds are selected on within-fold val partitions — never on test.
* Each fold uses one block as the test set; the remaining blocks are
  split 80/20 into train / val (by block count).
* Results are returned as a tidy DataFrame with one row per (fold × model).
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_rows(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    mask = df["flood_label"].isin([0.0, 1.0])
    for c in feature_cols:
        if c in df.columns:
            mask = mask & df[c].notna()
    return df[mask].copy()


def _buffer_filter(
    df: pd.DataFrame,
    test_block_id: int,
    buffer_degrees: float,
) -> pd.DataFrame:
    """
    Remove cells from df that lie within `buffer_degrees` of the boundary of
    the test block, to reduce spatial autocorrelation leakage.

    Parameters
    ----------
    df             : subset that will be used as the test set.
    test_block_id  : ID of the held-out block (for context; all rows in df are
                     already from this block).
    buffer_degrees : geographic buffer in decimal degrees.
    """
    if buffer_degrees <= 0:
        return df
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    # Erode the bounding box by buffer_degrees
    mask = (
        (df["lat"] >= lat_min + buffer_degrees)
        & (df["lat"] <= lat_max - buffer_degrees)
        & (df["lon"] >= lon_min + buffer_degrees)
        & (df["lon"] <= lon_max - buffer_degrees)
    )
    buffered = df[mask].copy()
    logger.debug(
        "Block %d: buffer %.4f° removed %d/%d test cells",
        test_block_id, buffer_degrees, len(df) - len(buffered), len(df)
    )
    return buffered


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
    """Train a model and return (test_probs, optimal_threshold)."""
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


# ── Validator class ───────────────────────────────────────────────────────────

class SpatialKFoldCV:
    """
    Rotating spatial K-fold cross-validation with optional buffer exclusion.

    Parameters
    ----------
    df             : Feature DataFrame for a single region (or multi-event).
    feature_columns: Column names to use as model inputs.
    event_filter   : If set, restrict to a single event_id.
    model_params   : Dict of model name → hyper-parameter dict.
    buffer_degrees : Geographic exclusion buffer around test block boundary.
    val_block_frac : Fraction of remaining blocks reserved for validation.
    output_dir     : If set, saves results CSV here.
    seed           : Random seed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        event_filter: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        buffer_degrees: float = 0.0,
        val_block_frac: float = 0.20,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.model_registry = _build_model_registry(model_params)
        self.buffer_degrees = buffer_degrees
        self.val_block_frac = val_block_frac
        self.output_dir = Path(output_dir) if output_dir else None
        self.seed = seed

        # Filter rows
        working = df.copy()
        if event_filter:
            working = working[working["event_id"] == event_filter]
        self._df = _valid_rows(working, self.feature_columns)
        self._available_feats = [f for f in self.feature_columns if f in self._df.columns]
        self._blocks = sorted(self._df["spatial_block_id"].unique().tolist())

        logger.info(
            "SpatialKFoldCV: %d valid rows, %d blocks, %d features, buffer=%.4f°",
            len(self._df), len(self._blocks), len(self._available_feats), buffer_degrees
        )

    # ── Single fold ───────────────────────────────────────────────────────────

    def _run_fold(self, test_block: int) -> List[Dict[str, Any]]:
        """Run all models for one K-fold rotation."""
        results: List[Dict[str, Any]] = []

        test_df_raw = self._df[self._df["spatial_block_id"] == test_block]
        test_df = _buffer_filter(test_df_raw, test_block, self.buffer_degrees)
        if len(test_df) < 10:
            logger.warning("Block %d: only %d buffered test rows; skipping fold.", test_block, len(test_df))
            return results

        if len(test_df["flood_label"].unique()) < 2:
            logger.warning("Block %d: single class in test set; skipping fold.", test_block)
            return results

        # Remaining blocks → train + val
        remain_blocks = [b for b in self._blocks if b != test_block]
        rng = np.random.RandomState(self.seed + test_block)
        n_val = max(1, int(len(remain_blocks) * self.val_block_frac))
        val_blocks = set(rng.choice(remain_blocks, size=n_val, replace=False).tolist())
        train_blocks = set(remain_blocks) - val_blocks

        train_df = self._df[self._df["spatial_block_id"].isin(train_blocks)]
        val_df = self._df[self._df["spatial_block_id"].isin(val_blocks)]

        X_train = train_df[self._available_feats].values.astype(float)
        y_train = train_df["flood_label"].values.astype(float)
        X_val = val_df[self._available_feats].values.astype(float)
        y_val = val_df["flood_label"].values.astype(float)
        X_test = test_df[self._available_feats].values.astype(float)
        y_test = test_df["flood_label"].values.astype(float)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            logger.warning("Block %d: single class in train/val; skipping fold.", test_block)
            return results

        for model_name, mparams in self.model_registry.items():
            try:
                test_probs, thresh = _fit_predict(
                    model_name, mparams, X_train, y_train, X_val, y_val, X_test
                )
                metrics = evaluate_predictions(y_test, test_probs, threshold=thresh)
                results.append({
                    "protocol": "spatial_kfold",
                    "test_block": int(test_block),
                    "n_train_blocks": len(train_blocks),
                    "n_val_blocks": len(val_blocks),
                    "buffer_degrees": self.buffer_degrees,
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
                    "note": "",
                })
                logger.info(
                    "Kfold block=%-3d model=%-20s ROC-AUC=%.4f  F1=%.4f",
                    test_block, model_name,
                    metrics["roc_auc"] if not np.isnan(metrics["roc_auc"]) else -1,
                    metrics["f1"],
                )
            except Exception as exc:
                logger.error("Kfold block=%d model=%s FAILED: %s", test_block, model_name, exc)
                results.append({
                    "protocol": "spatial_kfold",
                    "test_block": int(test_block),
                    "n_train_blocks": len(train_blocks),
                    "n_val_blocks": len(val_blocks),
                    "buffer_degrees": self.buffer_degrees,
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "brier_score": float("nan"),
                    "false_negative_rate": float("nan"),
                    "prevalence": float(y_test.mean()),
                    "threshold": 0.5,
                    "note": f"ERROR: {exc}",
                })
        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Run all K folds (one per block).

        Returns
        -------
        pd.DataFrame with one row per (fold × model).
        """
        logger.info("=== Spatial K-Fold CV (%d folds, buffer=%.4f°) ===",
                    len(self._blocks), self.buffer_degrees)
        all_results: List[Dict[str, Any]] = []
        for block in self._blocks:
            fold_results = self._run_fold(block)
            all_results.extend(fold_results)

        results_df = pd.DataFrame(all_results)

        if self.output_dir and not results_df.empty:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"spatial_kfold_buffer{int(self.buffer_degrees*1e4):06d}.csv"
            out_path = self.output_dir / fname
            results_df.to_csv(out_path, index=False)
            logger.info("Spatial K-fold results saved to %s", out_path)

        return results_df

    @staticmethod
    def summarise(results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-model mean and std across all folds."""
        if results_df.empty:
            return pd.DataFrame()
        return (
            results_df.groupby("model")[["roc_auc", "pr_auc", "f1", "brier_score", "false_negative_rate"]]
            .agg(["mean", "std"])
            .round(4)
        )
