"""
src/evaluation/cross_region_cv.py

Leave-One-Region-Out (LORO) Cross-Validation Framework
=======================================================

Scientific purpose
------------------
Spatial generalisation is the central research question of GeoFlood-AI 2.0.
Standard within-region block holdouts measure interpolation quality but do NOT
measure whether a model trained in one geographic domain can predict flood
susceptibility in an unseen domain.

LORO validation answers: "How well does a model trained on regions A, B, C
generalise to held-out region D?"

Design decisions
----------------
* Each rotation holds out exactly ONE region's data as the test set.
* Training data comes from ALL other regions that have processed feature files
  available on disk.
* The scaler is always fitted on the training partition only.
* Threshold selection is performed on a within-training spatial-block split
  (no test-region data touches threshold selection).
* If fewer than 2 regions have data, the validator falls back to a
  single-region within-region block holdout and flags the result accordingly.
* XGBoost / LightGBM are included when available; the framework degrades
  gracefully when they are not installed.

Output
------
A pandas DataFrame (and optional CSV) with columns:
  protocol | held_out_region | train_regions | model | n_train | n_test |
  roc_auc | pr_auc | f1 | precision | recall | brier_score |
  false_negative_rate | prevalence | note
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    evaluate_predictions,
    find_optimal_threshold,
)
from src.models.baselines import train_logistic_regression, train_random_forest
from src.models.nn import FloodSusceptibilityNN, predict_proba_nn, train_neural_network

logger = logging.getLogger(__name__)

# ── Optional gradient-boosting imports ───────────────────────────────────────
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


# ── Feature columns agreed across phases ──────────────────────────────────────
FEATURE_COLUMNS = [
    "rain_event_sum",
    "rain_3d",
    "rain_7d",
    "elevation",
    "slope",
    "tpi",
    "relative_elevation",
]

# Columns that identify geography
GEO_COLUMNS = ["lat", "lon", "spatial_block_id", "event_id", "flood_label", "is_perm_water"]

# Validation block fraction (from training region's blocks)
VAL_BLOCK_FRACTION = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_region_features(processed_dir: Path, region_id: str) -> Optional[pd.DataFrame]:
    """
    Load processed features for a region.
    Primary path: {processed_dir}/{region_id}_features.parquet
    Fallback:     {processed_dir}/features.parquet  (for assam/legacy)
    Returns None if no file exists.
    """
    candidate = processed_dir / f"{region_id}_features.parquet"
    if candidate.exists():
        df = pd.read_parquet(candidate)
        df["_region"] = region_id
        return df

    # Legacy fallback for the Assam primary dataset
    if region_id in {"assam", "brahmaputra"}:
        legacy = processed_dir / "features.parquet"
        if legacy.exists():
            df = pd.read_parquet(legacy)
            df["_region"] = "assam"
            return df

    return None


def _valid_land_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only binary-labelled, non-missing feature rows."""
    mask = df["flood_label"].isin([0.0, 1.0])
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            mask = mask & df[col].notna()
    return df[mask].copy()


def _train_val_blocks_split(
    df: pd.DataFrame, val_fraction: float = VAL_BLOCK_FRACTION, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single region's training set into train / val using spatial blocks.
    Val blocks are chosen randomly but never overlap with train blocks.
    """
    blocks = df["spatial_block_id"].unique()
    rng = np.random.RandomState(seed)
    n_val = max(1, int(len(blocks) * val_fraction))
    val_blocks = set(rng.choice(blocks, size=n_val, replace=False).tolist())
    train_blocks = set(blocks) - val_blocks

    return (
        df[df["spatial_block_id"].isin(train_blocks)].copy(),
        df[df["spatial_block_id"].isin(val_blocks)].copy(),
    )


def _build_model_registry(
    default_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the set of candidate models with their default hyper-parameters."""
    params = default_params or {}
    registry: Dict[str, Any] = {
        "logistic_regression": params.get("logistic_regression", {"max_iter": 1000}),
        "random_forest": params.get("random_forest", {"n_estimators": 100, "max_depth": 12}),
        "neural_network": params.get("neural_network", {"hidden_size": 64, "epochs": 30, "lr": 0.001}),
    }
    if _GB_AVAILABLE:
        if xgboost_available():
            registry["xgboost"] = params.get("xgboost", {"n_estimators": 200, "max_depth": 6})
        if lightgbm_available():
            registry["lightgbm"] = params.get("lightgbm", {"n_estimators": 200, "max_depth": 6})
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
    """
    Train one model, find threshold on val, return (test_probs, threshold).
    Neural network receives scaled arrays; tree models receive unscaled arrays.
    Returns (probs_on_test, threshold).
    """
    scaler_needed = model_name in {"logistic_regression", "neural_network", "xgboost", "lightgbm"}

    if scaler_needed:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_vl = scaler.transform(X_val)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_vl, X_te = X_train, X_val, X_test

    if model_name == "logistic_regression":
        m = train_logistic_regression(X_tr, y_train, model_params)
        val_probs = m.predict_proba(X_vl)[:, 1]
        test_probs = m.predict_proba(X_te)[:, 1]

    elif model_name == "random_forest":
        m = train_random_forest(X_tr, y_train, model_params)
        val_probs = m.predict_proba(X_vl)[:, 1]
        test_probs = m.predict_proba(X_te)[:, 1]

    elif model_name == "neural_network":
        m, _ = train_neural_network(X_tr, y_train, X_vl, y_val, model_params)
        val_probs = predict_proba_nn(m, X_vl)
        test_probs = predict_proba_nn(m, X_te)

    elif model_name == "xgboost" and _GB_AVAILABLE:
        m = train_xgboost(X_tr, y_train, model_params, X_vl, y_val)
        val_probs = m.predict_proba(X_vl)[:, 1]
        test_probs = m.predict_proba(X_te)[:, 1]

    elif model_name == "lightgbm" and _GB_AVAILABLE:
        m = train_lightgbm(X_tr, y_train, model_params, X_vl, y_val)
        val_probs = m.predict_proba(X_vl)[:, 1]
        test_probs = m.predict_proba(X_te)[:, 1]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    thresh, _ = find_optimal_threshold(y_val, val_probs)
    return test_probs, thresh


# ── Main validator class ───────────────────────────────────────────────────────

class LoroValidator:
    """
    Leave-One-Region-Out (LORO) cross-validation for flood susceptibility.

    Usage
    -----
    >>> validator = LoroValidator(
    ...     region_ids=["assam", "kerala", "mahanadi", "godavari"],
    ...     processed_dir=Path("data/processed"),
    ... )
    >>> results_df = validator.run()
    >>> results_df.to_csv("reports/loro_results.csv", index=False)
    """

    def __init__(
        self,
        region_ids: List[str],
        processed_dir: Path,
        model_params: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.region_ids = region_ids
        self.processed_dir = Path(processed_dir)
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.model_params = _build_model_registry(model_params)
        self.output_dir = Path(output_dir) if output_dir else None
        self.seed = seed
        self._region_data: Dict[str, pd.DataFrame] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_all_regions(self) -> Dict[str, pd.DataFrame]:
        """Load data for every configured region; skip missing ones with a warning."""
        loaded: Dict[str, pd.DataFrame] = {}
        for rid in self.region_ids:
            df = _load_region_features(self.processed_dir, rid)
            if df is None:
                logger.warning("LORO: No feature file found for region '%s' — skipping.", rid)
                continue
            valid = _valid_land_rows(df)
            available_features = [f for f in self.feature_columns if f in valid.columns]
            if not available_features:
                logger.warning("LORO: Region '%s' has no recognised feature columns — skipping.", rid)
                continue
            if len(valid) < 50:
                logger.warning("LORO: Region '%s' has only %d valid rows — skipping.", rid, len(valid))
                continue
            loaded[rid] = valid
            logger.info("LORO: Loaded region '%s': %d rows, prevalence=%.3f",
                        rid, len(valid), float(valid["flood_label"].mean()))
        return loaded

    # ── Single LORO fold ──────────────────────────────────────────────────────

    def _run_fold(
        self,
        held_out: str,
        train_regions: List[str],
        region_data: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Run all candidate models for one LORO fold.
        Returns a list of result dicts (one per model).
        """
        fold_results: List[Dict[str, Any]] = []

        # Assemble multi-region training dataframe
        train_frames = [region_data[r] for r in train_regions]
        train_df_full = pd.concat(train_frames, ignore_index=True)

        # Available features (intersection across all sources)
        available_feats = [f for f in self.feature_columns if f in train_df_full.columns]
        test_df = region_data[held_out]
        available_feats = [f for f in available_feats if f in test_df.columns]

        if not available_feats:
            logger.warning("LORO fold held_out=%s: no common feature columns; skipping.", held_out)
            return fold_results

        # Val split from training data (spatial-block based)
        train_df, val_df = _train_val_blocks_split(train_df_full, seed=self.seed)

        X_train = train_df[available_feats].values.astype(float)
        y_train = train_df["flood_label"].values.astype(float)
        X_val = val_df[available_feats].values.astype(float)
        y_val = val_df["flood_label"].values.astype(float)
        X_test = test_df[available_feats].values.astype(float)
        y_test = test_df["flood_label"].values.astype(float)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            note = "single_class_in_split"
            logger.warning("LORO held_out=%s: %s — skipping fold.", held_out, note)
            return fold_results

        for model_name, mparams in self.model_params.items():
            try:
                test_probs, thresh = _fit_predict(
                    model_name, mparams, X_train, y_train, X_val, y_val, X_test
                )
                metrics = evaluate_predictions(y_test, test_probs, threshold=thresh)
                fold_results.append({
                    "protocol": "LORO",
                    "held_out_region": held_out,
                    "train_regions": ",".join(sorted(train_regions)),
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "features_used": ",".join(available_feats),
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
                    "LORO held_out=%-10s model=%-20s ROC-AUC=%.4f  F1=%.4f",
                    held_out, model_name,
                    metrics["roc_auc"] if not np.isnan(metrics["roc_auc"]) else -1,
                    metrics["f1"],
                )
            except Exception as exc:
                logger.error("LORO held_out=%s model=%s FAILED: %s", held_out, model_name, exc)
                fold_results.append({
                    "protocol": "LORO",
                    "held_out_region": held_out,
                    "train_regions": ",".join(sorted(train_regions)),
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "features_used": ",".join(available_feats),
                    "roc_auc": float("nan"),
                    "pr_auc": float("nan"),
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "brier_score": float("nan"),
                    "false_negative_rate": float("nan"),
                    "prevalence": float(y_test.mean()) if len(y_test) else float("nan"),
                    "threshold": 0.5,
                    "note": f"ERROR: {exc}",
                })

        return fold_results

    # ── Single-region fallback ────────────────────────────────────────────────

    def _run_single_region_baseline(
        self,
        region_data: Dict[str, pd.DataFrame],
    ) -> List[Dict[str, Any]]:
        """
        Fallback when only one region has data.
        Uses within-region spatial block holdout.
        """
        results: List[Dict[str, Any]] = []
        (only_region, df), = region_data.items()
        logger.info("LORO: only one region available (%s); running within-region block holdout.", only_region)

        available_feats = [f for f in self.feature_columns if f in df.columns]
        train_df, val_df = _train_val_blocks_split(df, val_fraction=0.30, seed=self.seed)
        test_df = val_df   # use val as held-out test when only one split possible

        X_train = train_df[available_feats].values.astype(float)
        y_train = train_df["flood_label"].values.astype(float)
        X_val = val_df[available_feats].values.astype(float)
        y_val = val_df["flood_label"].values.astype(float)
        X_test = test_df[available_feats].values.astype(float)
        y_test = test_df["flood_label"].values.astype(float)

        for model_name, mparams in self.model_params.items():
            try:
                test_probs, thresh = _fit_predict(
                    model_name, mparams, X_train, y_train, X_val, y_val, X_test
                )
                metrics = evaluate_predictions(y_test, test_probs, threshold=thresh)
                results.append({
                    "protocol": "within_region_block_holdout",
                    "held_out_region": only_region,
                    "train_regions": only_region,
                    "model": model_name,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "features_used": ",".join(available_feats),
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "brier_score": metrics["brier_score"],
                    "false_negative_rate": metrics["false_negative_rate"],
                    "prevalence": metrics["prevalence"],
                    "threshold": metrics["threshold"],
                    "note": "single_region_fallback",
                })
            except Exception as exc:
                logger.error("Within-region baseline model=%s FAILED: %s", model_name, exc)

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the full LORO experiment.

        Returns
        -------
        pd.DataFrame with one row per (held-out region × model).
        Saves CSV to output_dir/loro_results.csv when output_dir is set.
        """
        logger.info("=== Leave-One-Region-Out Validation ===")
        region_data = self._load_all_regions()
        self._region_data = region_data

        if len(region_data) == 0:
            logger.error("LORO: No region data found. Cannot proceed.")
            return pd.DataFrame()

        all_results: List[Dict[str, Any]] = []

        if len(region_data) == 1:
            all_results = self._run_single_region_baseline(region_data)
        else:
            available_regions = list(region_data.keys())
            for held_out in available_regions:
                train_regions = [r for r in available_regions if r != held_out]
                logger.info("--- LORO fold: hold out '%s', train on %s ---", held_out, train_regions)
                fold_results = self._run_fold(held_out, train_regions, region_data)
                all_results.extend(fold_results)

        results_df = pd.DataFrame(all_results)

        if self.output_dir and not results_df.empty:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / "loro_results.csv"
            results_df.to_csv(out_path, index=False)
            logger.info("LORO results saved to %s", out_path)

        return results_df

    # ── Summary helpers ───────────────────────────────────────────────────────

    @staticmethod
    def summarise(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate LORO results per model across all held-out regions.
        Returns mean ± std for ROC-AUC, F1, Brier score.
        """
        if results_df.empty:
            return pd.DataFrame()
        return (
            results_df[results_df["protocol"] == "LORO"]
            .groupby("model")[["roc_auc", "pr_auc", "f1", "brier_score", "false_negative_rate"]]
            .agg(["mean", "std"])
            .round(4)
        )
