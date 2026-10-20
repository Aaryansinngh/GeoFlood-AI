"""
src/models/train.py

Main training and evaluation pipeline for GeoFlood AI.
Orchestrates:
1. Dataset loading and spatial block holdout partitioning
2. Feature scaling (fitted strictly on train set)
3. Model training: Logistic Regression, Random Forest, PyTorch Neural Network
4. Threshold selection on Validation set ONLY
5. Held-out test set evaluation with real metrics (Precision, Recall, F1, ROC-AUC, PR-AUC, Brier, FNR)
6. Event-holdout cross-event evaluation (DFO_4259 -> DFO_4288)
7. Saving serialized models, scaler, metrics.json, and evaluation curves
8. Full study-area inference and GeoJSON export for the interactive map
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.evaluation.splits import get_spatial_block_split, get_event_holdout_split
from src.evaluation.metrics import find_optimal_threshold, evaluate_predictions, compute_roc_pr_curves
from src.models.baselines import train_logistic_regression, train_random_forest, get_feature_importances
from src.models.nn import train_neural_network, predict_proba_nn, FloodSusceptibilityNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "rain_event_sum",
    "rain_3d",
    "rain_7d",
    "elevation",
    "slope",
    "tpi",
    "relative_elevation"
]


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_error_analysis(df_test: pd.DataFrame, y_true: np.ndarray, y_probs: np.ndarray, threshold: float) -> dict:
    """
    Analyze model failure modes across terrain and geographic features.
    """
    df_eval = df_test.copy()
    df_eval["y_true"] = y_true
    df_eval["y_prob"] = y_probs
    df_eval["y_pred"] = (y_probs >= threshold).astype(int)
    df_eval["is_error"] = (df_eval["y_true"] != df_eval["y_pred"]).astype(int)
    df_eval["is_fn"] = ((df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0)).astype(int)
    df_eval["is_fp"] = ((df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)).astype(int)

    # Elevation strata
    elev_bins = [0, 65, 85, 120, 1000]
    elev_labels = ["Low Floodplain (<65m)", "Terrace (65-85m)", "Piedmont (85-120m)", "Hills (>120m)"]
    df_eval["elev_tier"] = pd.cut(df_eval["elevation"], bins=elev_bins, labels=elev_labels, include_lowest=True)

    elev_analysis = {}
    for tier, grp in df_eval.groupby("elev_tier", observed=False):
        if len(grp) > 0:
            elev_analysis[str(tier)] = {
                "cell_count": int(len(grp)),
                "actual_flooded": int(grp["y_true"].sum()),
                "error_rate": float(grp["is_error"].mean()),
                "false_negatives": int(grp["is_fn"].sum()),
                "false_positives": int(grp["is_fp"].sum()),
            }

    # Slope strata
    slope_bins = [0, 0.5, 2.0, 10.0, 90.0]
    slope_labels = ["Very Flat (<0.5°)", "Gentle Slope (0.5°-2°)", "Moderate (2°-10°)", "Steep (>10°)"]
    df_eval["slope_tier"] = pd.cut(df_eval["slope"], bins=slope_bins, labels=slope_labels, include_lowest=True)

    slope_analysis = {}
    for tier, grp in df_eval.groupby("slope_tier", observed=False):
        if len(grp) > 0:
            slope_analysis[str(tier)] = {
                "cell_count": int(len(grp)),
                "actual_flooded": int(grp["y_true"].sum()),
                "error_rate": float(grp["is_error"].mean()),
                "false_negatives": int(grp["is_fn"].sum()),
                "false_positives": int(grp["is_fp"].sum()),
            }

    return {
        "elevation_breakdown": elev_analysis,
        "slope_breakdown": slope_analysis,
        "overall_error_count": int(df_eval["is_error"].sum()),
        "total_test_cells": int(len(df_eval))
    }


def export_map_geojson(df_full: pd.DataFrame, nn_model: FloodSusceptibilityNN, scaler: StandardScaler,
                       out_path: Path, max_features: int = 12000):
    """
    Generate GeoJSON layer with predicted risk probabilities for the interactive Folium map.
    Downsamples systematically across the spatial grid to maintain high web performance.
    """
    # Filter to primary event
    df_map = df_full[df_full["event_id"] == "DFO_4259"].copy()

    # Predict probabilities for valid land cells
    valid_mask = ~df_map["elevation"].isna()
    X_full = df_map.loc[valid_mask, FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X_full)
    probs = predict_proba_nn(nn_model, X_scaled)

    df_map["susceptibility_prob"] = np.nan
    df_map.loc[valid_mask, "susceptibility_prob"] = np.round(probs, 4)

    # Classify risk category
    def assign_category(p, is_perm):
        if is_perm == 1:
            return "Permanent Water"
        if np.isnan(p):
            return "Unavailable"
        if p >= 0.70:
            return "High Risk"
        elif p >= 0.35:
            return "Moderate Risk"
        else:
            return "Low Risk"

    df_map["risk_category"] = [
        assign_category(p, w) for p, w in zip(df_map["susceptibility_prob"], df_map["is_perm_water"])
    ]

    # Downsample systematically for GeoJSON map layer
    step = max(1, int(len(df_map) / max_features))
    sample_for_geojson = df_map.iloc[::step].copy()

    features = []
    for _, r in sample_for_geojson.iterrows():
        props = {
            "cell_id": str(r["cell_id"]),
            "risk_category": str(r["risk_category"]),
            "susceptibility_prob": None if np.isnan(r["susceptibility_prob"]) else float(r["susceptibility_prob"]),
            "elevation_m": round(float(r["elevation"]), 1),
            "slope_deg": round(float(r["slope"]), 2),
            "rain_sum_mm": round(float(r["rain_event_sum"]), 1),
            "is_observed_flood": int(r["flood_label"] == 1.0) if not np.isnan(r["flood_label"]) else None,
            "is_perm_water": int(r["is_perm_water"])
        }
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon"]), float(r["lat"])]
            },
            "properties": props
        }
        features.append(feature)

    geojson_obj = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)

    logger.info("Saved map GeoJSON to %s (%d features)", out_path, len(features))
    return df_map


def train_and_evaluate():
    config = load_config()
    proc_dir = Path(config["paths"]["processed_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    outputs_dir = Path(config["paths"]["outputs_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    features_file = proc_dir / "features.parquet"
    if not features_file.exists():
        logger.warning("Processed features not found, falling back to data/sample/features_sample.parquet")
        features_file = Path(config["paths"]["sample_dir"]) / "features_sample.parquet"

    logger.info("Loading feature table from %s...", features_file)
    df = pd.read_parquet(features_file)

    split_cfg = config["data_split"]
    train_b = split_cfg["train_block_ids"]
    val_b = split_cfg["val_block_ids"]
    test_b = split_cfg["test_block_ids"]

    # 1. Primary Split: Spatial Block Holdout on DFO_4259
    logger.info("Configuring Spatial Block Holdout: Train=%s, Val=%s, Test=%s", train_b, val_b, test_b)
    train_df, val_df, test_df = get_spatial_block_split(
        df=df, train_blocks=train_b, val_blocks=val_b, test_blocks=test_b, event_filter="DFO_4259"
    )

    logger.info("Train set: %d cells (Flooded: %d)", len(train_df), (train_df["flood_label"] == 1.0).sum())
    logger.info("Validation set: %d cells (Flooded: %d)", len(val_df), (val_df["flood_label"] == 1.0).sum())
    logger.info("Held-out Test set: %d cells (Flooded: %d)", len(test_df), (test_df["flood_label"] == 1.0).sum())

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["flood_label"].values
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["flood_label"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["flood_label"].values

    # 2. Scale features (strictly fitted on Train set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Training
    # Model A: Logistic Regression
    logger.info("Training Baseline 1: Logistic Regression...")
    lr_model = train_logistic_regression(X_train_scaled, y_train, config["models"]["logistic_regression"])

    # Model B: Random Forest
    logger.info("Training Baseline 2: Random Forest Classifier...")
    rf_model = train_random_forest(X_train, y_train, config["models"]["random_forest"])
    rf_feat_imp = get_feature_importances(rf_model, FEATURE_COLUMNS)

    # Model C: PyTorch Neural Network
    logger.info("Training Primary Model: PyTorch Feed-Forward Neural Network...")
    nn_model, nn_history = train_neural_network(
        X_train_scaled, y_train, X_val_scaled, y_val, config["models"]["neural_network"]
    )

    # 4. Predictions on Validation Set & Optimal Threshold Selection
    logger.info("Optimizing decision thresholds on VALIDATION set...")
    lr_val_probs = lr_model.predict_proba(X_val_scaled)[:, 1]
    rf_val_probs = rf_model.predict_proba(X_val)[:, 1]
    nn_val_probs = predict_proba_nn(nn_model, X_val_scaled)

    lr_thresh, _ = find_optimal_threshold(y_val, lr_val_probs)
    rf_thresh, _ = find_optimal_threshold(y_val, rf_val_probs)
    nn_thresh, _ = find_optimal_threshold(y_val, nn_val_probs)

    logger.info("Optimal validation thresholds: LR=%.4f, RF=%.4f, NN=%.4f", lr_thresh, rf_thresh, nn_thresh)

    # 5. Held-Out Test Set Evaluation
    logger.info("Evaluating all models on HELD-OUT TEST SET...")
    lr_test_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_test_probs = rf_model.predict_proba(X_test)[:, 1]
    nn_test_probs = predict_proba_nn(nn_model, X_test_scaled)

    metrics_lr = evaluate_predictions(y_test, lr_test_probs, threshold=lr_thresh)
    metrics_rf = evaluate_predictions(y_test, rf_test_probs, threshold=rf_thresh)
    metrics_nn = evaluate_predictions(y_test, nn_test_probs, threshold=nn_thresh)

    # Secondary Split: Event Holdout (Train on DFO_4259, Test on DFO_4288)
    event_test_df = df[(df["event_id"] == "DFO_4288") & (df["flood_label"].isin([0.0, 1.0]))].copy()
    X_ev_test = event_test_df[FEATURE_COLUMNS].values
    y_ev_test = event_test_df["flood_label"].values
    X_ev_scaled = scaler.transform(X_ev_test)
    nn_ev_probs = predict_proba_nn(nn_model, X_ev_scaled)
    metrics_nn_event_holdout = evaluate_predictions(y_ev_test, nn_ev_probs, threshold=nn_thresh)

    all_metrics = {
        "spatial_block_holdout": {
            "logistic_regression": metrics_lr,
            "random_forest": metrics_rf,
            "neural_network": metrics_nn
        },
        "event_holdout": {
            "train_event": "DFO_4259 (June 2015)",
            "test_event": "DFO_4288 (August 2015)",
            "neural_network": metrics_nn_event_holdout
        },
        "timestamp": datetime.now().isoformat(),
        "features": FEATURE_COLUMNS,
        "flood_prevalence_train": float(np.mean(y_train)),
        "flood_prevalence_test": float(np.mean(y_test))
    }

    # Save metrics JSON
    metrics_json_path = models_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved test metrics to %s", metrics_json_path)

    # 6. Curves and Visualizations JSON
    curves_data = {
        "nn_training_history": nn_history,
        "curves": {
            "logistic_regression": compute_roc_pr_curves(y_test, lr_test_probs),
            "random_forest": compute_roc_pr_curves(y_test, rf_test_probs),
            "neural_network": compute_roc_pr_curves(y_test, nn_test_probs),
        },
        "random_forest_feature_importance": rf_feat_imp,
        "error_analysis": run_error_analysis(test_df, y_test, nn_test_probs, threshold=nn_thresh)
    }

    curves_json_path = outputs_dir / "evaluation_charts.json"
    with open(curves_json_path, "w", encoding="utf-8") as f:
        json.dump(curves_data, f, indent=2)
    logger.info("Saved evaluation curves and error analysis to %s", curves_json_path)

    # 7. Model Serialization
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(lr_model, models_dir / "logistic_regression.joblib")
    joblib.dump(rf_model, models_dir / "random_forest.joblib")
    torch.save(nn_model.state_dict(), models_dir / "neural_network.pt")

    with open(models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    metadata = {
        "project": "GeoFlood AI",
        "created_at": datetime.now().isoformat(),
        "input_features": FEATURE_COLUMNS,
        "split_type": "spatial_block_holdout",
        "study_region": config["region"]["name"],
        "primary_event": config["events"]["primary"]["event_id"],
        "thresholds": {
            "logistic_regression": lr_thresh,
            "random_forest": rf_thresh,
            "neural_network": nn_thresh
        }
    }
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Inference across entire master grid for interactive map
    logger.info("Running spatial inference across study region...")
    predictions_df = export_map_geojson(
        df_full=df,
        nn_model=nn_model,
        scaler=scaler,
        out_path=outputs_dir / "predictions.geojson"
    )
    predictions_parquet_path = outputs_dir / "predictions.parquet"
    predictions_df.to_parquet(predictions_parquet_path, index=False)
    logger.info("Saved full predictions table to %s", predictions_parquet_path)

    logger.info("Training and evaluation completed successfully!")
    return all_metrics


if __name__ == "__main__":
    train_and_evaluate()
