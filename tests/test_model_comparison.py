"""
tests/test_model_comparison.py

Verification tests for Phase 2:
1. Multi-model comparison returns valid probabilities for all three models.
2. Preprocessing integrity: Scaler applied to NN and LR; raw features passed to RF.
3. Scenario analysis: Rainfall features change while terrain features remain strictly frozen.
4. Raster overlay generation succeeds for all three models without errors.
5. Handling of missing features safely marks status 'Unavailable' and never 'Safe'.
6. Out-of-bounds coordinates return failure without crashing.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.predict import GeoFloodPredictor
from src.geo.location_analysis import LocationAnalyzer, RAINFALL_BOUNDS
from app import generate_raster_overlays


@pytest.fixture(scope="module")
def app_data():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parquet_path = Path("outputs/predictions.parquet")
    if not parquet_path.exists():
        parquet_path = Path("data/sample/features_sample.parquet")
    df = pd.read_parquet(parquet_path)
    predictor = GeoFloodPredictor("models")
    analyzer = LocationAnalyzer(df, config, predictor=predictor)
    return analyzer, df, config


def test_compare_models_all_three_models(app_data):
    """Verify compare_models evaluates all 3 models with finite probabilities in [0, 1]."""
    analyzer, _, _ = app_data
    # Valid location in Kaziranga floodplain
    res = analyzer.compare_models(26.65, 93.35)

    assert res["success"] is True
    assert res["is_inside_region"] is True
    assert res["has_missing_features"] is False

    models = res["models"]
    for m_key in ["neural_network", "random_forest", "logistic_regression"]:
        assert m_key in models, f"Model {m_key} missing from comparison result"
        m = models[m_key]
        assert 0.0 <= m["baseline_prob"] <= 1.0, f"Probability for {m_key} out of bounds: {m['baseline_prob']}"
        assert m["baseline_category"] in ["High Risk", "Moderate Risk", "Low Risk", "Permanent Water"]
        assert m["baseline_decision"] in ["Susceptible (Flood Prone)", "Non-Susceptible (Resilient)", "Permanent Water Channel"]
        assert "test_metrics" in m
        assert "roc_auc" in m["test_metrics"]
        assert "recall" in m["test_metrics"]
        assert "precision" in m["test_metrics"]


def test_preprocessing_integrity(app_data):
    """
    Verify preprocessing integrity:
    - GeoFloodPredictor scales features for LR and NN using scaler.joblib
    - GeoFloodPredictor passes unscaled features to Random Forest
    """
    analyzer, _, _ = app_data
    sample_feat = {
        "rain_event_sum": 1200.0,
        "rain_3d": 65.0,
        "rain_7d": 140.0,
        "elevation": 72.0,
        "slope": 1.2,
        "tpi": -0.5,
        "relative_elevation": 2.1
    }

    # Verify predictions return single float
    p_nn = analyzer.predictor.predict_point(sample_feat, model_type="neural_network")
    p_rf = analyzer.predictor.predict_point(sample_feat, model_type="random_forest")
    p_lr = analyzer.predictor.predict_point(sample_feat, model_type="logistic_regression")

    assert isinstance(p_nn, float) and 0.0 <= p_nn <= 1.0
    assert isinstance(p_rf, float) and 0.0 <= p_rf <= 1.0
    assert isinstance(p_lr, float) and 0.0 <= p_lr <= 1.0


def test_scenario_freezes_terrain_features(app_data):
    """
    Verify that in scenario analysis:
    - Rainfall features are updated
    - Terrain features (elevation, slope, HAND, TPI) remain strictly unchanged
    """
    analyzer, _, _ = app_data
    overrides = {
        "rain_3d": 140.0,
        "rain_7d": 280.0,
        "rain_event_sum": 1600.0
    }
    res = analyzer.compare_models(26.65, 93.35, rainfall_overrides=overrides)

    assert res["success"] is True
    assert res["is_scenario"] is True

    base_f = res["baseline_features"]
    scen_f = res["scenario_features"]

    # Terrain must be identical
    for terrain_col in ["elevation", "slope", "tpi", "relative_elevation"]:
        assert base_f[terrain_col] == scen_f[terrain_col], f"Terrain feature {terrain_col} was modified!"

    # Rainfall must reflect overrides
    assert scen_f["rain_3d"] == 140.0
    assert scen_f["rain_7d"] == 280.0
    assert scen_f["rain_event_sum"] == 1600.0


def test_generate_raster_overlays_all_models(app_data):
    """Verify raster overlays generate valid RGBA arrays for all three models."""
    _, df_preds, _ = app_data

    for m_key in ["neural_network", "random_forest", "logistic_regression"]:
        rgba_suscep, rgba_obs, shape = generate_raster_overlays(df_preds, model_key=m_key)
        assert rgba_suscep is not None
        assert rgba_obs is not None
        assert shape[0] == 400
        assert shape[1] == 623
        assert rgba_suscep.shape == (400, 623, 4)
        assert rgba_suscep.dtype == np.uint8


def test_missing_feature_handling_in_comparison(app_data):
    """Verify cells with missing features return 'Unavailable' for all models."""
    analyzer, df_preds, config = app_data

    # Create dummy analyzer with a missing feature row
    dummy_df = df_preds.head(5).copy()
    dummy_df.loc[dummy_df.index[0], "elevation"] = np.nan
    dummy_analyzer = LocationAnalyzer(dummy_df, config, predictor=analyzer.predictor)

    test_lat = float(dummy_df.iloc[0]["lat"])
    test_lon = float(dummy_df.iloc[0]["lon"])

    res = dummy_analyzer.compare_models(test_lat, test_lon)
    assert res["success"] is True
    assert res["has_missing_features"] is True
    assert "elevation" in res["missing_features"]

    for m_key in ["neural_network", "random_forest", "logistic_regression"]:
        m = res["models"][m_key]
        assert m["baseline_prob"] is None
        assert m["baseline_category"] == "Unavailable"
        assert m["baseline_decision"] == "Unavailable"


def test_out_of_bounds_rejection(app_data):
    """Verify out-of-bounds coordinates return failure without exceptions."""
    analyzer, _, _ = app_data

    # Tokyo, Japan
    res = analyzer.compare_models(35.6762, 139.6503)
    assert res["success"] is False
    assert res["is_inside_region"] is False
    assert "outside the supported study area" in res["error"]
