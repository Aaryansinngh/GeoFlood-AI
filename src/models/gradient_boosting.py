"""
src/models/gradient_boosting.py

Gradient Boosting baselines for GeoFlood-AI 2.0:
  - XGBoost (xgb 3.x)
  - LightGBM (lgbm 4.x)

Both honour class-imbalance via scale_pos_weight / is_unbalance, mirror the
interface of train_logistic_regression / train_random_forest in baselines.py,
and fall back gracefully if either library is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed – XGB model unavailable.")

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    logger.warning("LightGBM not installed – LGB model unavailable.")


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> "xgb.XGBClassifier":
    """
    Train an XGBoost binary classifier.

    Parameters
    ----------
    X_train, y_train : training arrays (already scaled if desired).
    params           : model hyper-parameters (mirrors config yaml).
    X_val, y_val     : optional early-stopping validation set.

    Returns
    -------
    Fitted XGBClassifier.
    """
    if not _XGB_AVAILABLE:
        raise ImportError("XGBoost is not installed.")

    # Compute scale_pos_weight from training prevalence
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    default_spw = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=params.get("scale_pos_weight", default_spw),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )

    fit_kwargs: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ── LightGBM ─────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> "lgb.LGBMClassifier":
    """
    Train a LightGBM binary classifier.

    Parameters
    ----------
    X_train, y_train : training arrays.
    params           : model hyper-parameters.
    X_val, y_val     : optional early-stopping validation set.

    Returns
    -------
    Fitted LGBMClassifier.
    """
    if not _LGB_AVAILABLE:
        raise ImportError("LightGBM is not installed.")

    model = lgb.LGBMClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        is_unbalance=params.get("is_unbalance", True),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", -1),
        verbose=-1,
    )

    fit_kwargs: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = callbacks

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ── Feature importance helper ─────────────────────────────────────────────────

def get_gb_feature_importances(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract sorted feature importances from an XGB or LGB model.
    Uses gain-based importances when available.
    """
    try:
        if _XGB_AVAILABLE and isinstance(model, xgb.XGBClassifier):
            imp = model.get_booster().get_fscore()
            total = sum(imp.values()) or 1.0
            return {f: imp.get(f"f{i}", 0.0) / total for i, f in enumerate(feature_names)}
        elif _LGB_AVAILABLE and isinstance(model, lgb.LGBMClassifier):
            raw = model.feature_importances_
            total = raw.sum() or 1.0
            sorted_pairs = sorted(
                zip(feature_names, (raw / total).tolist()), key=lambda x: x[1], reverse=True
            )
            return {n: float(v) for n, v in sorted_pairs}
    except Exception as exc:
        logger.warning("Could not extract GB feature importances: %s", exc)

    return {f: 0.0 for f in feature_names}


# ── Availability helpers ──────────────────────────────────────────────────────

def xgboost_available() -> bool:
    """Return True if XGBoost is importable."""
    return _XGB_AVAILABLE


def lightgbm_available() -> bool:
    """Return True if LightGBM is importable."""
    return _LGB_AVAILABLE
