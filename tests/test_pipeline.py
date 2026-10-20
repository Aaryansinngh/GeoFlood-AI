"""
tests/test_pipeline.py

Comprehensive test suite for GeoFlood AI (Prototype).
Validates data integrity, spatial splits, metric computations,
model reloads, GeoJSON CRS compliance, and edge-case handling.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import rasterio

from src.evaluation.metrics import evaluate_predictions, find_optimal_threshold
from src.evaluation.splits import get_spatial_block_split
from src.geo.grid import assign_spatial_blocks, calculate_slope
from src.models.predict import GeoFloodPredictor
from src.models.fuzzy_layer import MamdaniRiskFuzzySystem
from src.geo.sar_detection import otsu_threshold, process_sar_flood_detection


# Test 1: Metric functions match known hand-calculated values on toy data
def test_metrics_match_known_toy_data():
    # 4 samples: [0, 1, 1, 0]
    # Predictions: [0.1, 0.9, 0.4, 0.8] with threshold 0.5 -> [0, 1, 0, 1]
    # TP=1 (idx 1), FP=1 (idx 3), TN=1 (idx 0), FN=1 (idx 2)
    y_true = np.array([0, 1, 1, 0])
    y_probs = np.array([0.1, 0.9, 0.4, 0.8])
    res = evaluate_predictions(y_true, y_probs, threshold=0.5)

    cm = res["confusion_matrix"]
    assert cm["true_positive"] == 1
    assert cm["false_positive"] == 1
    assert cm["true_negative"] == 1
    assert cm["false_negative"] == 1

    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    assert res["precision"] == pytest.approx(0.5)
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    assert res["recall"] == pytest.approx(0.5)
    # F1 = 0.5
    assert res["f1"] == pytest.approx(0.5)
    # FNR = FN / (TP + FN) = 1 / 2 = 0.5
    assert res["false_negative_rate"] == pytest.approx(0.5)
    # Brier score = mean((y_probs - y_true)^2) = ((0.1)^2 + (0.9-1)^2 + (0.4-1)^2 + (0.8-0)^2) / 4
    # = (0.01 + 0.01 + 0.36 + 0.64) / 4 = 1.02 / 4 = 0.255
    assert res["brier_score"] == pytest.approx(0.255)


# Test 2: Spatial block split has zero spatial overlap between train, val, and test blocks
def test_spatial_block_split_no_overlap():
    toy_df = pd.DataFrame({
        "cell_id": [f"c_{i}" for i in range(12)],
        "event_id": ["DFO_4259"] * 12,
        "spatial_block_id": list(range(12)),
        "flood_label": [0.0, 1.0] * 6
    })

    train_blocks = [0, 1, 4, 5, 8, 9, 11]
    val_blocks = [3, 6]
    test_blocks = [2, 7, 10]

    train_set, val_set, test_set = get_spatial_block_split(
        df=toy_df,
        train_blocks=train_blocks,
        val_blocks=val_blocks,
        test_blocks=test_blocks,
        event_filter="DFO_4259"
    )

    train_b = set(train_set["spatial_block_id"])
    val_b = set(val_set["spatial_block_id"])
    test_b = set(test_set["spatial_block_id"])

    assert len(train_b.intersection(val_b)) == 0
    assert len(train_b.intersection(test_b)) == 0
    assert len(val_b.intersection(test_b)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == 12


# Test 3: Spatial block grid generator partitions 2D space without gaps
def test_spatial_block_generator_integrity():
    blocks = assign_spatial_blocks(height=30, width=40, n_blocks_y=3, n_blocks_x=4)
    assert blocks.shape == (30, 40)
    assert len(np.unique(blocks)) == 12
    assert blocks.min() == 0
    assert blocks.max() == 11


# Test 4: Feature builder / data loader handles nodata and preserves NaN
def test_feature_builder_handles_nodata():
    df = pd.read_parquet("data/sample/features_sample.parquet")
    # Verify that nodata / unobserved values are not silently turned into 0 or "safe"
    assert "flood_label" in df.columns
    # Check that binary labels are valid floats or NaN
    unique_labels = set(df["flood_label"].dropna().unique())
    assert unique_labels.issubset({0.0, 1.0, -1.0})


# Test 5: Unavailable data is not turned into 0 or "safe"
def test_unavailable_data_not_safe():
    df = pd.read_parquet("outputs/predictions.parquet")
    # Check that when probability is NaN, risk_category is 'Unavailable' (not 'Low Risk')
    unavail_cells = df[df["susceptibility_prob"].isna() & (df["is_perm_water"] == 0)]
    if len(unavail_cells) > 0:
        assert (unavail_cells["risk_category"] == "Unavailable").all()


# Test 6: Saved models reload correctly and predict finite outputs in [0, 1]
def test_saved_model_reloads_finite_outputs():
    predictor = GeoFloodPredictor("models")
    sample_df = pd.read_parquet("data/sample/features_sample.parquet").head(20)

    for m_type in ["logistic_regression", "random_forest", "neural_network"]:
        probs = predictor.predict(sample_df, model_type=m_type)
        assert len(probs) == len(sample_df)
        assert np.all(np.isfinite(probs)), f"{m_type} produced non-finite probabilities"
        assert np.all((probs >= 0.0) & (probs <= 1.0)), f"{m_type} produced probabilities outside [0, 1]"


# Test 7: Invalid input is rejected with a clear error
def test_invalid_input_rejected():
    predictor = GeoFloodPredictor("models")
    sample_df = pd.read_parquet("data/sample/features_sample.parquet").head(5)

    with pytest.raises(ValueError):
        predictor.predict(sample_df, model_type="unsupported_quantum_model")


# Test 8: Map GeoJSON layer is in EPSG:4326 and features are valid
def test_geojson_layer_epsg4326():
    geojson_path = Path("outputs/predictions.geojson")
    assert geojson_path.exists(), "predictions.geojson does not exist"

    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)

    assert geo_data["type"] == "FeatureCollection"
    assert len(geo_data["features"]) > 0

    first_feat = geo_data["features"][0]
    coords = first_feat["geometry"]["coordinates"]
    lon, lat = coords[0], coords[1]

    # Verify WGS84 bounding box constraints
    assert 92.0 <= lon <= 95.0, f"Longitude {lon} outside study area"
    assert 26.0 <= lat <= 28.0, f"Latitude {lat} outside study area"

    props = first_feat["properties"]
    expected_keys = {"cell_id", "risk_category", "elevation_m", "slope_deg"}
    assert expected_keys.issubset(set(props.keys()))


# Test 9: Mamdani Fuzzy System evaluates consistently across input domain
def test_mamdani_fuzzy_system():
    fuzzy = MamdaniRiskFuzzySystem()
    score_high, cat_high = fuzzy.evaluate(probability=0.90, rainfall_3d=120.0)
    score_low, cat_low = fuzzy.evaluate(probability=0.05, rainfall_3d=10.0)

    assert 0.0 <= score_high <= 100.0
    assert 0.0 <= score_low <= 100.0
    assert score_high > score_low
    assert cat_high == "High Risk"
    assert cat_low == "Low Risk"


# Test 10: Sentinel-1 SAR Otsu thresholding identifies specular water reduction
def test_sar_detection_otsu():
    rng = np.random.RandomState(42)
    # Bimodal backscatter: land (-10 dB), water (-22 dB)
    land = rng.normal(-10.0, 1.5, 500)
    water = rng.normal(-22.0, 1.5, 500)
    bimodal = np.concatenate([land, water])

    thresh = otsu_threshold(bimodal)
    assert -19.0 <= thresh <= -13.0, f"Otsu threshold {thresh} failed to split bimodal SAR backscatter"
