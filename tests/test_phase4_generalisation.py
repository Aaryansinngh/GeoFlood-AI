"""
tests/test_phase4_generalisation.py

Tests for Phase 4:
- Gradient boosting model wrappers (XGBoost & LightGBM)
- Spatial K-Fold CV with spatial buffer exclusion
- Leave-One-Region-Out (LORO) validator
- Leave-One-Flood-Event-Out (LOFEO) validator
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.models.gradient_boosting import (
    train_xgboost,
    train_lightgbm,
    xgboost_available,
    lightgbm_available,
    get_gb_feature_importances,
)
from src.evaluation.spatial_kfold_cv import SpatialKFoldCV, _buffer_filter
from src.evaluation.cross_region_cv import LoroValidator
from src.evaluation.temporal_cv import TemporalEventCV


@pytest.fixture
def synthetic_multiblock_df():
    """Create a small multi-block, multi-event synthetic dataset for testing CV rotations."""
    np.random.seed(42)
    n_samples = 400
    blocks = [0, 1, 2, 3]
    events = ["EV_01", "EV_02"]

    df = pd.DataFrame({
        "cell_id": np.arange(n_samples),
        "lat": np.random.uniform(26.0, 27.0, n_samples),
        "lon": np.random.uniform(92.0, 93.0, n_samples),
        "spatial_block_id": np.random.choice(blocks, n_samples),
        "event_id": np.random.choice(events, n_samples),
        "elevation": np.random.uniform(50, 150, n_samples),
        "slope": np.random.uniform(0.1, 15.0, n_samples),
        "tpi": np.random.uniform(-5.0, 5.0, n_samples),
        "relative_elevation": np.random.uniform(1.0, 30.0, n_samples),
        "rain_3d": np.random.uniform(10.0, 100.0, n_samples),
        "rain_7d": np.random.uniform(30.0, 200.0, n_samples),
        "rain_event_sum": np.random.uniform(50.0, 300.0, n_samples),
        "flood_label": np.random.choice([0.0, 1.0], n_samples, p=[0.8, 0.2]),
        "is_perm_water": 0,
    })
    return df


def test_gradient_boosting_wrappers():
    """Verify XGBoost and LightGBM train and predict correctly when installed."""
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100)
    feat_names = [f"f_{i}" for i in range(5)]

    if xgboost_available():
        xgb_m = train_xgboost(X, y, {"n_estimators": 10, "max_depth": 3})
        probs = xgb_m.predict_proba(X)[:, 1]
        assert probs.shape == (100,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        imps = get_gb_feature_importances(xgb_m, feat_names)
        assert len(imps) == 5

    if lightgbm_available():
        lgb_m = train_lightgbm(X, y, {"n_estimators": 10, "max_depth": 3, "num_leaves": 8})
        probs = lgb_m.predict_proba(X)[:, 1]
        assert probs.shape == (100,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        imps = get_gb_feature_importances(lgb_m, feat_names)
        assert len(imps) == 5


def test_buffer_filter():
    """Verify that spatial buffer removes edge points."""
    df = pd.DataFrame({
        "lat": [26.01, 26.50, 26.99],
        "lon": [92.01, 92.50, 92.99],
    })
    # Buffer 0.05 from boundaries [26.01, 26.99] and [92.01, 92.99]
    buffered = _buffer_filter(df, test_block_id=1, buffer_degrees=0.05)
    assert len(buffered) == 1
    assert buffered.iloc[0]["lat"] == 26.50


def test_spatial_kfold_cv_execution(synthetic_multiblock_df, tmp_path):
    """Test full execution of rotating spatial K-fold CV."""
    cv = SpatialKFoldCV(
        df=synthetic_multiblock_df,
        feature_columns=["elevation", "slope", "rain_3d"],
        model_params={
            "logistic_regression": {"max_iter": 100},
            "random_forest": {"n_estimators": 10, "max_depth": 4},
        },
        buffer_degrees=0.0,
        output_dir=tmp_path,
    )
    results = cv.run()
    assert not results.empty
    assert "protocol" in results.columns
    assert set(results["protocol"].unique()) == {"spatial_kfold"}
    assert set(results["model"].unique()).issubset({"logistic_regression", "random_forest", "neural_network", "xgboost", "lightgbm"})
    assert (tmp_path / "spatial_kfold_buffer000000.csv").exists()

    summary = SpatialKFoldCV.summarise(results)
    assert not summary.empty


def test_temporal_event_cv_execution(synthetic_multiblock_df, tmp_path):
    """Test full execution of LOFEO CV."""
    cv = TemporalEventCV(
        df=synthetic_multiblock_df,
        feature_columns=["elevation", "slope", "rain_3d"],
        model_params={
            "logistic_regression": {"max_iter": 100},
            "random_forest": {"n_estimators": 10, "max_depth": 4},
        },
        output_dir=tmp_path,
    )
    results = cv.run()
    assert not results.empty
    assert "protocol" in results.columns
    assert set(results["protocol"].unique()) == {"LOFEO"}
    assert (tmp_path / "temporal_lofeo_results.csv").exists()


def test_loro_validator_execution(tmp_path):
    """Test LORO validator fallback and multi-region behaviour."""
    # Create two synthetic region feature files in tmp_path
    np.random.seed(42)
    n = 200
    for rid in ["region_a", "region_b"]:
        df = pd.DataFrame({
            "cell_id": np.arange(n),
            "lat": np.random.uniform(20.0, 22.0, n),
            "lon": np.random.uniform(80.0, 82.0, n),
            "spatial_block_id": np.random.choice([0, 1, 2], n),
            "event_id": f"EV_{rid}",
            "elevation": np.random.uniform(10, 100, n),
            "slope": np.random.uniform(0.1, 5.0, n),
            "tpi": np.random.uniform(-2, 2, n),
            "relative_elevation": np.random.uniform(1, 10, n),
            "rain_3d": np.random.uniform(20, 80, n),
            "rain_7d": np.random.uniform(50, 150, n),
            "rain_event_sum": np.random.uniform(100, 250, n),
            "flood_label": np.random.choice([0.0, 1.0], n, p=[0.75, 0.25]),
            "is_perm_water": 0,
        })
        df.to_parquet(tmp_path / f"{rid}_features.parquet")

    validator = LoroValidator(
        region_ids=["region_a", "region_b"],
        processed_dir=tmp_path,
        model_params={
            "logistic_regression": {"max_iter": 100},
            "random_forest": {"n_estimators": 10, "max_depth": 4},
        },
        output_dir=tmp_path,
    )
    results = validator.run()
    assert not results.empty
    assert "protocol" in results.columns
    assert set(results["protocol"].unique()) == {"LORO"}
    assert set(results["held_out_region"].unique()) == {"region_a", "region_b"}
    assert (tmp_path / "loro_results.csv").exists()

    summary = LoroValidator.summarise(results)
    assert not summary.empty
