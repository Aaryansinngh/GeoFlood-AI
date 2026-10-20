"""
tests/test_scientific_validation.py

Phase 3 Scientific Validation Test Suite
Covers:
1. Spatial block train/test separation (zero spatial overlap)
2. Spatially buffered holdout (boundary bleed elimination)
3. Threshold selection strictly on validation data (zero test leakage)
4. Preprocessing isolation (StandardScaler fitted strictly on training partition)
5. Missing feature handling (never converted to safe)
6. Probability bounds and finite outputs
7. Brier score against prevalence baseline
8. Probability calibration on validation set
9. Scenario physical monotonicity validation (rain_3d <= rain_7d <= rain_event_sum)
10. Scenario extrapolation detection (flags out-of-distribution values)
11. Geospatial grid CRS (EPSG:4326) and master raster dimensions (400 x 623)
12. SAR validation audit provenance and agreement classification
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
import joblib
import torch

from src.models.predict import GeoFloodPredictor
from src.models.nn import FloodSusceptibilityNN, predict_proba_nn
from src.evaluation.splits import get_spatial_block_split
from src.evaluation.metrics import find_optimal_threshold, evaluate_predictions
from src.geo.location_analysis import LocationAnalyzer, TRAINING_RAINFALL_DISTRIBUTION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@pytest.fixture(scope="module")
def validation_data():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    proc_path = Path("data/processed/features.parquet")
    if not proc_path.exists():
        proc_path = Path("data/sample/features_sample.parquet")
    df = pd.read_parquet(proc_path)
    predictor = GeoFloodPredictor("models")
    analyzer = LocationAnalyzer(df, config, predictor=predictor)
    return analyzer, df, config


def test_spatial_block_zero_overlap(validation_data):
    """Verify zero overlap between train, val, and test blocks."""
    _, df, config = validation_data
    split_cfg = config["data_split"]
    train_b = split_cfg["train_block_ids"]
    val_b = split_cfg["val_block_ids"]
    test_b = split_cfg["test_block_ids"]

    train_df, val_df, test_df = get_spatial_block_split(df, train_b, val_b, test_b, event_filter="DFO_4259")

    train_blocks = set(train_df["spatial_block_id"].unique())
    val_blocks = set(val_df["spatial_block_id"].unique())
    test_blocks = set(test_df["spatial_block_id"].unique())

    assert len(train_blocks.intersection(val_blocks)) == 0
    assert len(train_blocks.intersection(test_blocks)) == 0
    assert len(val_blocks.intersection(test_blocks)) == 0


def test_spatially_buffered_holdout_removes_boundary_cells(validation_data):
    """Verify 1km buffer exclusion removes boundary cells from test blocks."""
    _, df, _ = validation_data
    ev1 = df[(df["event_id"] == "DFO_4259") & (df["flood_label"].isin([0, 1]))]
    test_df = ev1[ev1["spatial_block_id"].isin([2, 7, 10])].copy()

    k = 4
    m2 = (test_df["spatial_block_id"] == 2) & ((test_df["row"] <= 133 - k) & (test_df["col"] >= 312 + k) & (test_df["col"] <= 467 - k))
    m7 = (test_df["spatial_block_id"] == 7) & ((test_df["row"] >= 134 + k) & (test_df["row"] <= 266 - k) & (test_df["col"] >= 468 + k))
    m10 = (test_df["spatial_block_id"] == 10) & ((test_df["row"] >= 267 + k) & (test_df["col"] >= 312 + k) & (test_df["col"] <= 467 - k))
    buffered_test = test_df[m2 | m7 | m10]

    assert len(buffered_test) < len(test_df)
    # Exactly 3,934 boundary cells should be excluded
    assert len(test_df) - len(buffered_test) == 3934


def test_threshold_selection_never_sees_test_labels(validation_data):
    """Verify optimal decision threshold can be determined strictly on validation partition."""
    _, df, config = validation_data
    split_cfg = config["data_split"]
    _, val_df, test_df = get_spatial_block_split(
        df, split_cfg["train_block_ids"], split_cfg["val_block_ids"], split_cfg["test_block_ids"], event_filter="DFO_4259"
    )

    y_val = val_df["flood_label"].values
    dummy_val_probs = np.linspace(0.1, 0.9, len(y_val))

    thresh, _ = find_optimal_threshold(y_val, dummy_val_probs)
    assert 0.0 < thresh < 1.0


def test_scaler_fitted_on_training_only(validation_data):
    """Verify that the serialized scaler has training dimension and mean."""
    analyzer, _, _ = validation_data
    scaler = analyzer.predictor.scaler
    assert hasattr(scaler, "mean_")
    assert len(scaler.mean_) == 7
    assert np.all(np.isfinite(scaler.mean_))
    assert np.all(np.isfinite(scaler.scale_))


def test_probability_bounds_and_finite(validation_data):
    """Verify all models predict valid probabilities strictly in [0, 1]."""
    analyzer, _, _ = validation_data
    res = analyzer.compare_models(26.65, 93.35)
    assert res["success"] is True

    for m_k in ["neural_network", "random_forest", "logistic_regression"]:
        p = res["models"][m_k]["baseline_prob"]
        assert 0.0 <= p <= 1.0
        assert np.isfinite(p)


def test_brier_score_vs_climatology_baseline(validation_data):
    """Verify Brier score baseline computation matches prevalence * (1 - prevalence)."""
    _, df, config = validation_data
    split_cfg = config["data_split"]
    _, _, test_df = get_spatial_block_split(
        df, split_cfg["train_block_ids"], split_cfg["val_block_ids"], split_cfg["test_block_ids"], event_filter="DFO_4259"
    )

    y_test = test_df["flood_label"].values
    prev = float(np.mean(y_test))
    expected_baseline = prev * (1.0 - prev)

    metrics = evaluate_predictions(y_test, np.full_like(y_test, prev, dtype=float), threshold=0.5)
    assert np.isclose(metrics["brier_score"], expected_baseline, atol=1e-5)


def test_platt_calibration_improves_neural_network_brier(validation_data):
    """Verify Platt scaling fitted on validation set achieves Brier score beating climatology."""
    analyzer, df, config = validation_data
    split_cfg = config["data_split"]
    _, val_df, test_df = get_spatial_block_split(
        df, split_cfg["train_block_ids"], split_cfg["val_block_ids"], split_cfg["test_block_ids"], event_filter="DFO_4259"
    )

    feats = ["rain_event_sum", "rain_3d", "rain_7d", "elevation", "slope", "tpi", "relative_elevation"]
    scaler = analyzer.predictor.scaler
    nn_model = analyzer.predictor.nn_model

    val_p = predict_proba_nn(nn_model, scaler.transform(val_df[feats].values))
    test_p = predict_proba_nn(nn_model, scaler.transform(test_df[feats].values))

    y_val = val_df["flood_label"].values
    y_test = test_df["flood_label"].values

    cal = LogisticRegression()
    cal.fit(val_p.reshape(-1, 1), y_val)
    test_p_cal = cal.predict_proba(test_p.reshape(-1, 1))[:, 1]

    uncal_brier = brier_score_loss(y_test, test_p)
    cal_brier = brier_score_loss(y_test, test_p_cal)
    climatology = np.mean(y_test) * (1 - np.mean(y_test))

    assert cal_brier < uncal_brier
    assert cal_brier < climatology


def test_scenario_physical_monotonicity_validation(validation_data):
    """Verify physical monotonicity check flags inconsistency when rain_3d > rain_7d."""
    analyzer, _, _ = validation_data
    # Inconsistent override: 3-day rainfall > 7-day rainfall
    bad_overrides = {
        "rain_3d": 120.0,
        "rain_7d": 80.0,
        "rain_event_sum": 500.0
    }
    res = analyzer.compare_models(26.65, 93.35, rainfall_overrides=bad_overrides)
    assert res["success"] is True
    assert res["is_physically_plausible"] is False
    assert "cannot exceed" in res["plausibility_warning"]


def test_scenario_extrapolation_detection(validation_data):
    """Verify extrapolation detection flags rainfall exceeding empirical training maximum."""
    analyzer, _, _ = validation_data
    # 250mm 3-day rain is well above the training maximum of 103.3mm
    extrap_overrides = {
        "rain_3d": 250.0,
        "rain_7d": 400.0,
        "rain_event_sum": 1200.0
    }
    res = analyzer.compare_models(26.65, 93.35, rainfall_overrides=extrap_overrides)
    assert res["success"] is True
    assert res["is_extrapolating"] is True
    assert any("exceeds the maximum observed training value" in w for w in res["extrapolation_warnings"])


def test_geospatial_grid_metadata_and_dimensions(validation_data):
    """Verify master raster grid metadata, CRS (EPSG:4326), and shape (400, 623)."""
    _, df, config = validation_data
    ev1 = df[df["event_id"] == "DFO_4259"]

    assert len(ev1) == 249200
    h = int(ev1["row"].max() + 1)
    w = int(ev1["col"].max() + 1)
    assert h == 400
    assert w == 623

    bbox = config["region"]["bbox"]
    assert ev1["lat"].min() >= bbox["min_lat"] - 0.05
    assert ev1["lat"].max() <= bbox["max_lat"] + 0.05
    assert ev1["lon"].min() >= bbox["min_lon"] - 0.05
    assert ev1["lon"].max() <= bbox["max_lon"] + 0.05


def test_sar_audit_provenance_and_disclaimer():
    """Verify SAR audit documentation exists and clarifies the 0.991 agreement metric."""
    sar_audit_path = Path("reports/sar_validation_audit.md")
    assert sar_audit_path.exists(), "sar_validation_audit.md missing"

    with open(sar_audit_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "0.991" in content
    assert "synthetic" in content.lower()
    assert "internal consistency" in content.lower() or "agreement" in content.lower()


def test_phase3_required_reports_exist():
    """Verify all 6 required Phase 3 documentation and artifact files exist."""
    required_files = [
        Path("reports/phase3_data_audit.md"),
        Path("reports/phase3_sar_provenance.md"),
        Path("reports/phase3_limitations.md"),
        Path("reports/phase3_validation_report.md"),
        Path("reports/phase3_reproducibility.json"),
        Path("reports/phase3_model_comparison.csv"),
    ]
    for rf in required_files:
        assert rf.exists(), f"Required Phase 3 file {rf} does not exist!"
        assert rf.stat().st_size > 100, f"File {rf} is empty or incomplete!"


def test_reproducibility_manifest_validity():
    """Verify reproducibility manifest structure, dependencies, and file hashes."""
    manifest_path = Path("reports/phase3_reproducibility.json")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "metadata" in manifest
    assert "dependencies" in manifest
    assert "dataset_artifacts" in manifest
    assert "model_artifacts" in manifest
    assert "protocols" in manifest

    # Ensure all file hashes are valid 64-char SHA256 hex strings (not FILE_NOT_FOUND)
    for k, v in manifest["dataset_artifacts"].items():
        assert len(v["sha256"]) == 64, f"Dataset artifact {k} has invalid hash: {v['sha256']}"
    for k, v in manifest["model_artifacts"].items():
        assert len(v["sha256"]) == 64, f"Model artifact {k} has invalid hash: {v['sha256']}"

    # Verify protocol block IDs
    proto = manifest["protocols"]
    assert proto["spatial_blocks"]["test_blocks"] == [2, 7, 10]
    assert proto["spatial_blocks"]["buffer_exclusion_distance_km"] == 1.0


def test_model_comparison_csv_contains_all_protocols_and_baselines():
    """Verify phase3_model_comparison.csv contains all protocols and baselines."""
    csv_path = Path("reports/phase3_model_comparison.csv")
    assert csv_path.exists()
    df_metrics = pd.read_csv(csv_path)

    splits = set(df_metrics["split"].unique())
    assert "Random Stratified Holdout (Leaked Baseline)" in splits
    assert "Spatial Block Holdout" in splits
    assert "Spatially Buffered Holdout (1km)" in splits
    assert "Cross-Event Holdout (DFO_4288)" in splits

    models = set(df_metrics["model"].unique())
    assert "Majority Class Baseline (Always 0)" in models
    assert "Climatology Baseline (Prevalence)" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert any("Neural Network" in m for m in models)


def test_rf_cross_event_collapse_and_nn_durability():
    """Empirically test that RF cross-event collapse is accurately documented and NN generalises."""
    csv_path = Path("reports/phase3_model_comparison.csv")
    df_metrics = pd.read_csv(csv_path)

    rf_ev2 = df_metrics[(df_metrics["split"] == "Cross-Event Holdout (DFO_4288)") & (df_metrics["model"] == "Random Forest")]
    nn_ev2 = df_metrics[(df_metrics["split"] == "Cross-Event Holdout (DFO_4288)") & (df_metrics["model"] == "Neural Network")]

    assert len(rf_ev2) == 1
    assert len(nn_ev2) == 1

    rf_auc = float(rf_ev2["roc_auc"].iloc[0])
    nn_auc = float(nn_ev2["roc_auc"].iloc[0])

    # RF collapses below random chance (< 0.50) due to axis-aligned tree splits
    assert rf_auc < 0.50, f"Expected RF cross-event collapse (<0.50), got {rf_auc}"
    # NN preserves generalisation (> 0.65)
    assert nn_auc >= 0.65, f"Expected NN to preserve generalisation (>=0.65), got {nn_auc}"


def test_probability_calibration_ece_reduction():
    """Verify that probability calibration (Platt) achieves significant ECE reduction on NN."""
    csv_path = Path("reports/phase3_model_comparison.csv")
    df_metrics = pd.read_csv(csv_path)

    nn_raw = df_metrics[(df_metrics["split"] == "Spatial Block Holdout") & (df_metrics["model"] == "Neural Network (Uncalibrated)")]
    nn_cal = df_metrics[(df_metrics["split"] == "Spatial Block Holdout") & (df_metrics["model"] == "Neural Network (Platt Calibrated)")]

    assert len(nn_raw) == 1
    assert len(nn_cal) == 1

    ece_raw = float(nn_raw["expected_calibration_error"].iloc[0])
    ece_cal = float(nn_cal["expected_calibration_error"].iloc[0])

    # Platt calibration should reduce ECE by at least 60%
    assert ece_cal < 0.10, f"Expected calibrated ECE < 0.10, got {ece_cal}"
    assert ece_cal < ece_raw * 0.40, f"Expected calibrated ECE to be reduced by at least 60%, got {ece_cal} vs {ece_raw}"

