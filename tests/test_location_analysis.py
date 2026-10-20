"""
tests/test_location_analysis.py

Test suite for Coordinate-Based Flood Susceptibility Analysis.
Validates:
1. Coordinate validation (valid, invalid, out-of-bounds)
2. Spatial grid-cell mapping
3. Model selection integrity (NN, RF, LR)
4. Missing feature handling (never labels missing data as safe)
5. Exploratory rainfall scenario evaluation
6. Permanent water identification
"""

import math
import pytest
import numpy as np
import pandas as pd
import yaml

from src.geo.location_analysis import LocationAnalyzer, RAINFALL_BOUNDS


@pytest.fixture(scope="module")
def analyzer():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    df_preds = pd.read_parquet("outputs/predictions.parquet")
    return LocationAnalyzer(df_features=df_preds, config=config)


# Test 1: Valid coordinates inside the study area
def test_valid_coordinates_inside_region(analyzer):
    # Central Kaziranga floodplain point
    res = analyzer.analyze_location(lat=26.65, lon=93.35, model_type="neural_network")
    assert res["success"] is True
    assert res["is_inside_region"] is True
    assert "cell_id" in res
    assert 0.0 <= res["susceptibility_prob"] <= 1.0
    assert res["risk_category"] in ["High Risk", "Moderate Risk", "Low Risk", "Permanent Water"]
    assert "features" in res
    assert res["has_missing_features"] is False


# Test 2: Invalid coordinate inputs (non-numeric, NaN, infinite)
def test_invalid_coordinates(analyzer):
    # NaN
    res_nan = analyzer.analyze_location(lat=float("nan"), lon=93.35)
    assert res_nan["success"] is False
    assert "cannot be NaN" in res_nan["error"]

    # String non-numeric
    res_str = analyzer.analyze_location(lat="not_a_latitude", lon=93.35)
    assert res_str["success"] is False
    assert "valid numeric" in res_str["error"]

    # Global latitude out of bounds (> 90)
    res_lat_oob = analyzer.analyze_location(lat=95.0, lon=93.35)
    assert res_lat_oob["success"] is False
    assert "outside the global range" in res_lat_oob["error"]


# Test 3: Coordinates outside the supported study area
def test_coordinates_outside_study_area(analyzer):
    # Delhi, India coordinates: ~28.6°N, 77.2°E (outside Assam Brahmaputra box)
    res_delhi = analyzer.analyze_location(lat=28.6139, lon=77.2090)
    assert res_delhi["success"] is False
    assert res_delhi["is_inside_region"] is False
    assert "outside the supported study area" in res_delhi["error"]

    # Just north of Assam box
    res_north = analyzer.analyze_location(lat=27.50, lon=93.50)
    assert res_north["success"] is False
    assert "outside the supported study area" in res_north["error"]


# Test 4: Coordinate mapping identifies the nearest grid cell accurately
def test_mapping_to_correct_grid_cell(analyzer):
    # Query cell 000_000 location
    target_cell = analyzer.df_features.iloc[0]
    cell_lat = float(target_cell["lat"])
    cell_lon = float(target_cell["lon"])

    res = analyzer.analyze_location(lat=cell_lat, lon=cell_lon)
    assert res["success"] is True
    assert res["cell_id"] == str(target_cell["cell_id"])
    assert res["cell_lat"] == pytest.approx(cell_lat, abs=1e-4)
    assert res["cell_lon"] == pytest.approx(cell_lon, abs=1e-4)


# Test 5: Missing feature handling preserves integrity and does NOT label location safe
def test_missing_feature_handling():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Synthetic row with NaN elevation
    toy_df = pd.DataFrame([{
        "cell_id": "test_missing_cell",
        "event_id": "DFO_4259",
        "row": 10,
        "col": 10,
        "lat": 26.65,
        "lon": 93.35,
        "elevation": np.nan,  # Missing
        "slope": 1.2,
        "tpi": 0.1,
        "relative_elevation": 2.5,
        "rain_event_sum": 1200.0,
        "rain_3d": 45.0,
        "rain_7d": 90.0,
        "is_perm_water": 0,
        "flood_label": np.nan
    }])

    analyzer_toy = LocationAnalyzer(df_features=toy_df, config=config)
    res = analyzer_toy.analyze_location(lat=26.65, lon=93.35)

    assert res["success"] is True
    assert res["has_missing_features"] is True
    assert "elevation" in res["missing_features"]
    assert res["risk_category"] == "Unavailable"
    assert res["susceptibility_prob"] is None
    # Crucial hard rule check: missing data is NEVER labeled safe!
    assert res["risk_category"] != "Low Risk"
    assert "NOT treated as safe" in res["warning"]


# Test 6: Saved-model inference for all 3 supported models
def test_saved_model_inference_all_models(analyzer):
    models = ["neural_network", "random_forest", "logistic_regression"]
    probs = {}
    for m in models:
        res = analyzer.analyze_location(lat=26.65, lon=93.35, model_type=m)
        assert res["success"] is True
        p = res["susceptibility_prob"]
        assert 0.0 <= p <= 1.0
        assert math.isfinite(p)
        assert res["model_type"] == m
        probs[m] = p

    # Verify that different models produce distinct inferences
    assert len(set(probs.values())) > 1, "Different models should produce model-specific outputs"


# Test 7: Correct threshold selection from metadata
def test_threshold_selection_integrity(analyzer):
    res_nn = analyzer.analyze_location(lat=26.65, lon=93.35, model_type="neural_network")
    res_rf = analyzer.analyze_location(lat=26.65, lon=93.35, model_type="random_forest")
    res_lr = analyzer.analyze_location(lat=26.65, lon=93.35, model_type="logistic_regression")

    assert res_nn["threshold"] == pytest.approx(analyzer.thresholds["neural_network"])
    assert res_rf["threshold"] == pytest.approx(analyzer.thresholds["random_forest"])
    assert res_lr["threshold"] == pytest.approx(analyzer.thresholds["logistic_regression"])


# Test 8: Exploratory rainfall scenario evaluation
def test_rainfall_scenario_evaluation(analyzer):
    baseline = analyzer.analyze_location(lat=26.65, lon=93.35, model_type="neural_network")
    assert baseline["is_scenario"] is False

    # Simulate heavy 3-day rainfall deluge (150 mm)
    scenario = analyzer.analyze_location(
        lat=26.65,
        lon=93.35,
        model_type="neural_network",
        rainfall_overrides={"rain_3d": 150.0, "rain_event_sum": 1800.0}
    )
    assert scenario["is_scenario"] is True
    assert scenario["features"]["rain_3d"] == 150.0
    assert scenario["features"]["rain_event_sum"] == 1800.0
    # Terrain must remain strictly unchanged!
    assert scenario["features"]["elevation"] == baseline["features"]["elevation"]
    assert scenario["features"]["slope"] == baseline["features"]["slope"]
    assert scenario["features"]["relative_elevation"] == baseline["features"]["relative_elevation"]
    assert 0.0 <= scenario["susceptibility_prob"] <= 1.0


# Test 9: Permanent water identification
def test_permanent_water_identification(analyzer):
    # Find a permanent water cell in the master dataframe
    perm_cells = analyzer.df_features[analyzer.df_features["is_perm_water"] == 1]
    if len(perm_cells) > 0:
        p_cell = perm_cells.iloc[0]
        res = analyzer.analyze_location(lat=float(p_cell["lat"]), lon=float(p_cell["lon"]))
        assert res["success"] is True
        assert res["is_perm_water"] == 1
        assert res["risk_category"] == "Permanent Water"
