"""
src/evaluation/run_validation.py

Phase 3: Scientific Validation, Generalisation & Research-Grade Evaluation Pipeline
Executes:
1. Spatial Block Holdout Evaluation (disjoint spatial blocks)
2. Spatially Buffered Holdout Evaluation (1 km buffer boundary exclusion to eliminate spatial autocorrelation)
3. Cross-Event Holdout Evaluation (DFO_4259 June 2015 -> DFO_4288 August 2015)
4. Probability Calibration (Platt Scaling & Isotonic Regression fitted strictly on validation set)
5. Prevalence Baseline Brier Comparisons and Brier Skill Scores
6. Permutation Feature Importance on held-out test data
7. High-resolution figure generation (ROC, PR, Confusion Matrices, Reliability Diagrams, Permutation Importance)
8. Export of phase3_metrics.csv and feature_availability.csv
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.models.predict import GeoFloodPredictor
from src.models.nn import FloodSusceptibilityNN, predict_proba_nn
from src.evaluation.splits import get_spatial_block_split, get_random_stratified_split
import hashlib
import platform
import sys
import sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_validation")

FEATURE_COLUMNS = [
    "rain_event_sum",
    "rain_3d",
    "rain_7d",
    "elevation",
    "slope",
    "tpi",
    "relative_elevation"
]

FEATURE_LABELS = {
    "rain_event_sum": "Event Cumulative Rainfall (mm)",
    "rain_3d": "Antecedent 3-Day Rainfall (mm)",
    "rain_7d": "Antecedent 7-Day Rainfall (mm)",
    "elevation": "SRTM Elevation (m)",
    "slope": "Terrain Slope (°)",
    "tpi": "Topographic Position Index (m)",
    "relative_elevation": "Relative Elevation / HAND (m)"
}


def load_dataset():
    proc_path = Path("data/processed/features.parquet")
    sample_path = Path("data/sample/features_sample.parquet")
    if proc_path.exists():
        logger.info("Loading processed dataset from %s", proc_path)
        return pd.read_parquet(proc_path)
    logger.info("Loading sample dataset from %s", sample_path)
    return pd.read_parquet(sample_path)


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) with equal-width probability bins.
    Measures the weighted average absolute difference between predicted confidence
    and empirical accuracy across probability bins.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        in_bin = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i+1] if i < n_bins - 1 else y_probs <= bin_edges[i+1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_probs[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def compute_comprehensive_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float, model_name: str, split_name: str) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_total = len(y_true)
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)
    prevalence = float(n_pos / n_total) if n_total > 0 else 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / n_neg) if n_neg > 0 else 0.0
    fnr = float(fn / n_pos) if n_pos > 0 else 0.0
    fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    balanced_acc = float((rec + specificity) / 2.0)

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = float("nan")

    brier = float(brier_score_loss(y_true, y_probs))
    brier_prevalence_baseline = float(prevalence * (1.0 - prevalence))
    brier_skill_score = float(1.0 - (brier / brier_prevalence_baseline)) if brier_prevalence_baseline > 0 else float("nan")
    ece = compute_ece(y_true, y_probs, n_bins=10)

    return {
        "split": split_name,
        "model": model_name,
        "sample_count": n_total,
        "positive_count": n_pos,
        "prevalence": prevalence,
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": specificity,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "balanced_accuracy": balanced_acc,
        "brier_score": brier,
        "brier_prevalence_baseline": brier_prevalence_baseline,
        "brier_skill_score": brier_skill_score,
        "expected_calibration_error": ece,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def run_permutation_importance(model_predict_fn, X_eval: np.ndarray, y_eval: np.ndarray, feature_names: list, n_repeats: int = 5, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    baseline_probs = model_predict_fn(X_eval)
    baseline_auc = roc_auc_score(y_eval, baseline_probs)
    importances = {}

    for idx, feat in enumerate(feature_names):
        drop_list = []
        for _ in range(n_repeats):
            X_perm = X_eval.copy()
            X_perm[:, idx] = rng.permutation(X_perm[:, idx])
            perm_probs = model_predict_fn(X_perm)
            perm_auc = roc_auc_score(y_eval, perm_probs)
            drop_list.append(baseline_auc - perm_auc)
        importances[feat] = {
            "mean_drop": float(np.mean(drop_list)),
            "std_drop": float(np.std(drop_list))
        }
    return importances


def generate_feature_availability_table() -> pd.DataFrame:
    records = [
        {
            "Feature": "rain_event_sum",
            "Source": "ERA5 Reanalysis / GPM IMERG",
            "Temporal Availability": "Post-event aggregate (cumulative rainfall over full event duration)",
            "Potential Leakage": "HIGH for operational forecasting; zero for retrospective susceptibility mapping",
            "Intended Task": "Retrospective event-based susceptibility / hazard reconstruction",
            "Decision": "Retained for event-based susceptibility; must be EXCLUDED if deployed for operational early-warning forecasting"
        },
        {
            "Feature": "rain_3d",
            "Source": "ERA5 Reanalysis / GPM IMERG",
            "Temporal Availability": "Available at event onset t_0 (3-day pre-event antecedent rainfall)",
            "Potential Leakage": "None (uses strictly historical pre-event window)",
            "Intended Task": "Both operational forecasting and susceptibility mapping",
            "Decision": "Valid across all operational and scientific tasks"
        },
        {
            "Feature": "rain_7d",
            "Source": "ERA5 Reanalysis / GPM IMERG",
            "Temporal Availability": "Available at event onset t_0 (7-day pre-event antecedent rainfall)",
            "Potential Leakage": "None (uses strictly historical pre-event window)",
            "Intended Task": "Both operational forecasting and susceptibility mapping",
            "Decision": "Valid across all operational and scientific tasks"
        },
        {
            "Feature": "elevation",
            "Source": "NASA SRTM GL1 (30m)",
            "Temporal Availability": "Static topographic attribute (immutable physical surface)",
            "Potential Leakage": "None",
            "Intended Task": "Static flood susceptibility, event detection, operational routing",
            "Decision": "Valid across all operational and scientific tasks"
        },
        {
            "Feature": "slope",
            "Source": "Derived from SRTM DEM finite difference gradient",
            "Temporal Availability": "Static topographic attribute",
            "Potential Leakage": "None",
            "Intended Task": "Static flood susceptibility, event detection, operational routing",
            "Decision": "Valid across all operational and scientific tasks"
        },
        {
            "Feature": "tpi",
            "Source": "Topographic Position Index (elevation - local mean)",
            "Temporal Availability": "Static topographic attribute (15-pixel window)",
            "Potential Leakage": "None",
            "Intended Task": "Static flood susceptibility, event detection, operational routing",
            "Decision": "Valid across all operational and scientific tasks"
        },
        {
            "Feature": "relative_elevation",
            "Source": "Height Above Drainage proxy (elevation - local min)",
            "Temporal Availability": "Static topographic attribute (25-pixel window)",
            "Potential Leakage": "None",
            "Intended Task": "Static flood susceptibility, event detection, operational routing",
            "Decision": "Valid across all operational and scientific tasks"
        },
        {
            "Feature": "is_perm_water",
            "Source": "JRC Global Surface Water (permanent river channel)",
            "Temporal Availability": "Long-term historical hydrography",
            "Potential Leakage": "Circular if used as predictive feature for dry-land flood label",
            "Intended Task": "Boundary masking & permanent water channel exclusion",
            "Decision": "Strictly EXCLUDED from feature vector; used exclusively as an evaluation mask (label = -1)"
        },
        {
            "Feature": "flood_label",
            "Source": "Global Flood Database (DFO/MODIS optical water raster)",
            "Temporal Availability": "Observed ground truth inundation",
            "Potential Leakage": "Fatal if leaked into inputs",
            "Intended Task": "Supervised training and validation target ONLY",
            "Decision": "Target label only; zero exposure to feature matrix"
        }
    ]
    return pd.DataFrame(records)


def execute_scientific_validation():
    logger.info("Initializing Research-Grade Scientific Validation Pipeline...")
    figures_dir = Path("reports/figures")
    reports_dir = Path("reports")
    models_dir = Path("models")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_dataset()

    # Load serialized models
    scaler = joblib.load(models_dir / "scaler.joblib")
    lr_model = joblib.load(models_dir / "logistic_regression.joblib")
    rf_model = joblib.load(models_dir / "random_forest.joblib")

    with open(models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    thresholds = metadata.get("thresholds", {
        "logistic_regression": 0.5888,
        "random_forest": 0.0560,
        "neural_network": 0.4394
    })

    nn_model = FloodSusceptibilityNN(len(FEATURE_COLUMNS))
    nn_model.load_state_dict(torch.load(models_dir / "neural_network.pt", weights_only=True))
    nn_model.eval()

    # 2. Partition datasets
    # PROTOCOL 1: Spatial Block Holdout (DFO_4259)
    train_b = [0, 1, 3, 4, 6, 8, 9]
    val_b = [5, 11]
    test_b = [2, 7, 10]
    train_df, val_df, test_df = get_spatial_block_split(df, train_b, val_b, test_b, event_filter="DFO_4259")

    # PROTOCOL 0: Naive Random Stratified Holdout (Baseline to measure spatial autocorrelation leakage)
    train_rnd_df, val_rnd_df, test_rnd_df = get_random_stratified_split(df, event_filter="DFO_4259", train_ratio=0.70, val_ratio=0.15, random_state=42)

    # PROTOCOL 2: Spatially Buffered Holdout (buffer k=4 cells ~1km boundary exclusion)
    k = 4
    m2 = (test_df["spatial_block_id"] == 2) & ((test_df["row"] <= 133 - k) & (test_df["col"] >= 312 + k) & (test_df["col"] <= 467 - k))
    m7 = (test_df["spatial_block_id"] == 7) & ((test_df["row"] >= 134 + k) & (test_df["row"] <= 266 - k) & (test_df["col"] >= 468 + k))
    m10 = (test_df["spatial_block_id"] == 10) & ((test_df["row"] >= 267 + k) & (test_df["col"] >= 312 + k) & (test_df["col"] <= 467 - k))
    buffered_test_df = test_df[m2 | m7 | m10].copy()

    # PROTOCOL 3: Cross-Event Holdout (DFO_4288 August 2015)
    ev2_df = df[(df["event_id"] == "DFO_4288") & (df["flood_label"].isin([0.0, 1.0]))].copy()

    logger.info("Split Sizes: Random Test=%d, Block Train=%d, Block Val=%d, Block Test=%d, Buffered Test=%d, Event 2=%d",
                len(test_rnd_df), len(train_df), len(val_df), len(test_df), len(buffered_test_df), len(ev2_df))

    # Preprocessing
    X_rnd_test = test_rnd_df[FEATURE_COLUMNS].values
    y_rnd_test = test_rnd_df["flood_label"].values
    X_rnd_test_scaled = scaler.transform(X_rnd_test)

    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df["flood_label"].values
    X_val_scaled = scaler.transform(X_val)

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["flood_label"].values
    X_test_scaled = scaler.transform(X_test)

    X_buff = buffered_test_df[FEATURE_COLUMNS].values
    y_buff = buffered_test_df["flood_label"].values
    X_buff_scaled = scaler.transform(X_buff)

    X_ev2 = ev2_df[FEATURE_COLUMNS].values
    y_ev2 = ev2_df["flood_label"].values
    X_ev2_scaled = scaler.transform(X_ev2)

    # Predictions
    # Logistic Regression
    p_lr_rnd = lr_model.predict_proba(X_rnd_test_scaled)[:, 1]
    p_lr_val = lr_model.predict_proba(X_val_scaled)[:, 1]
    p_lr_test = lr_model.predict_proba(X_test_scaled)[:, 1]
    p_lr_buff = lr_model.predict_proba(X_buff_scaled)[:, 1]
    p_lr_ev2 = lr_model.predict_proba(X_ev2_scaled)[:, 1]

    # Random Forest (unscaled raw inputs)
    p_rf_rnd = rf_model.predict_proba(X_rnd_test)[:, 1]
    p_rf_val = rf_model.predict_proba(X_val)[:, 1]
    p_rf_test = rf_model.predict_proba(X_test)[:, 1]
    p_rf_buff = rf_model.predict_proba(X_buff)[:, 1]
    p_rf_ev2 = rf_model.predict_proba(X_ev2)[:, 1]

    # Neural Network (scaled inputs)
    p_nn_rnd = predict_proba_nn(nn_model, X_rnd_test_scaled)
    p_nn_val = predict_proba_nn(nn_model, X_val_scaled)
    p_nn_test = predict_proba_nn(nn_model, X_test_scaled)
    p_nn_buff = predict_proba_nn(nn_model, X_buff_scaled)
    p_nn_ev2 = predict_proba_nn(nn_model, X_ev2_scaled)

    # 3. Probability Calibration (Fitted Strictly on Validation Set)
    logger.info("Fitting probability calibrators on Validation set...")
    # Platt Scaling (Logistic Regression on validation probabilities)
    cal_platt_nn = LogisticRegression()
    cal_platt_nn.fit(p_nn_val.reshape(-1, 1), y_val)
    p_nn_test_platt = cal_platt_nn.predict_proba(p_nn_test.reshape(-1, 1))[:, 1]

    cal_platt_rf = LogisticRegression()
    cal_platt_rf.fit(p_rf_val.reshape(-1, 1), y_val)
    p_rf_test_platt = cal_platt_rf.predict_proba(p_rf_test.reshape(-1, 1))[:, 1]

    cal_platt_lr = LogisticRegression()
    cal_platt_lr.fit(p_lr_val.reshape(-1, 1), y_val)
    p_lr_test_platt = cal_platt_lr.predict_proba(p_lr_test.reshape(-1, 1))[:, 1]

    # Isotonic Regression on validation set
    cal_iso_nn = IsotonicRegression(out_of_bounds="clip")
    cal_iso_nn.fit(p_nn_val, y_val)
    p_nn_test_iso = cal_iso_nn.predict(p_nn_test)

    # 4. Compute Metrics across all evaluations
    metrics_list = []

    # Protocol 0: Random Stratified Holdout (Leaked Baseline)
    p_maj_rnd = np.zeros_like(y_rnd_test, dtype=float)
    p_prev_rnd = np.full_like(y_rnd_test, fill_value=float(np.mean(y_rnd_test)), dtype=float)
    metrics_list.append(compute_comprehensive_metrics(y_rnd_test, p_maj_rnd, 0.5, "Majority Class Baseline (Always 0)", "Random Stratified Holdout (Leaked Baseline)"))
    metrics_list.append(compute_comprehensive_metrics(y_rnd_test, p_prev_rnd, 0.5, "Climatology Baseline (Prevalence)", "Random Stratified Holdout (Leaked Baseline)"))
    metrics_list.append(compute_comprehensive_metrics(y_rnd_test, p_lr_rnd, thresholds["logistic_regression"], "Logistic Regression", "Random Stratified Holdout (Leaked Baseline)"))
    metrics_list.append(compute_comprehensive_metrics(y_rnd_test, p_rf_rnd, thresholds["random_forest"], "Random Forest", "Random Stratified Holdout (Leaked Baseline)"))
    metrics_list.append(compute_comprehensive_metrics(y_rnd_test, p_nn_rnd, thresholds["neural_network"], "Neural Network", "Random Stratified Holdout (Leaked Baseline)"))

    # Protocol 1: Spatial Block Holdout (Primary)
    p_maj_test = np.zeros_like(y_test, dtype=float)
    p_prev_test = np.full_like(y_test, fill_value=float(np.mean(y_test)), dtype=float)
    metrics_list.append(compute_comprehensive_metrics(y_test, p_maj_test, 0.5, "Majority Class Baseline (Always 0)", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_prev_test, 0.5, "Climatology Baseline (Prevalence)", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_lr_test, thresholds["logistic_regression"], "Logistic Regression", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_rf_test, thresholds["random_forest"], "Random Forest", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_nn_test, thresholds["neural_network"], "Neural Network (Uncalibrated)", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_nn_test_platt, thresholds["neural_network"], "Neural Network (Platt Calibrated)", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_nn_test_iso, thresholds["neural_network"], "Neural Network (Isotonic Calibrated)", "Spatial Block Holdout"))
    metrics_list.append(compute_comprehensive_metrics(y_test, p_rf_test_platt, thresholds["random_forest"], "Random Forest (Platt Calibrated)", "Spatial Block Holdout"))

    # Protocol 2: Spatially Buffered Holdout
    p_maj_buff = np.zeros_like(y_buff, dtype=float)
    p_prev_buff = np.full_like(y_buff, fill_value=float(np.mean(y_buff)), dtype=float)
    metrics_list.append(compute_comprehensive_metrics(y_buff, p_maj_buff, 0.5, "Majority Class Baseline (Always 0)", "Spatially Buffered Holdout (1km)"))
    metrics_list.append(compute_comprehensive_metrics(y_buff, p_prev_buff, 0.5, "Climatology Baseline (Prevalence)", "Spatially Buffered Holdout (1km)"))
    metrics_list.append(compute_comprehensive_metrics(y_buff, p_lr_buff, thresholds["logistic_regression"], "Logistic Regression", "Spatially Buffered Holdout (1km)"))
    metrics_list.append(compute_comprehensive_metrics(y_buff, p_rf_buff, thresholds["random_forest"], "Random Forest", "Spatially Buffered Holdout (1km)"))
    metrics_list.append(compute_comprehensive_metrics(y_buff, p_nn_buff, thresholds["neural_network"], "Neural Network", "Spatially Buffered Holdout (1km)"))

    # Protocol 3: Cross-Event Holdout (DFO_4288)
    p_maj_ev2 = np.zeros_like(y_ev2, dtype=float)
    p_prev_ev2 = np.full_like(y_ev2, fill_value=float(np.mean(y_ev2)), dtype=float)
    metrics_list.append(compute_comprehensive_metrics(y_ev2, p_maj_ev2, 0.5, "Majority Class Baseline (Always 0)", "Cross-Event Holdout (DFO_4288)"))
    metrics_list.append(compute_comprehensive_metrics(y_ev2, p_prev_ev2, 0.5, "Climatology Baseline (Prevalence)", "Cross-Event Holdout (DFO_4288)"))
    metrics_list.append(compute_comprehensive_metrics(y_ev2, p_lr_ev2, thresholds["logistic_regression"], "Logistic Regression", "Cross-Event Holdout (DFO_4288)"))
    metrics_list.append(compute_comprehensive_metrics(y_ev2, p_rf_ev2, thresholds["random_forest"], "Random Forest", "Cross-Event Holdout (DFO_4288)"))
    metrics_list.append(compute_comprehensive_metrics(y_ev2, p_nn_ev2, thresholds["neural_network"], "Neural Network", "Cross-Event Holdout (DFO_4288)"))

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(reports_dir / "phase3_metrics.csv", index=False)
    df_metrics.to_csv(reports_dir / "phase3_model_comparison.csv", index=False)
    logger.info("Saved phase3_metrics.csv and phase3_model_comparison.csv with %d evaluation rows", len(df_metrics))

    # Save feature availability table
    df_feat_avail = generate_feature_availability_table()
    df_feat_avail.to_csv(reports_dir / "feature_availability.csv", index=False)
    logger.info("Saved feature_availability.csv")
    logger.info("Saved feature_availability.csv")

    # 5. Permutation Feature Importance on Held-Out Test Set
    logger.info("Computing Permutation Feature Importance on held-out test data...")
    perm_rf = run_permutation_importance(
        model_predict_fn=lambda X: rf_model.predict_proba(X)[:, 1],
        X_eval=X_test,
        y_eval=y_test,
        feature_names=FEATURE_COLUMNS
    )

    perm_nn = run_permutation_importance(
        model_predict_fn=lambda X: predict_proba_nn(nn_model, scaler.transform(X)),
        X_eval=X_test,
        y_eval=y_test,
        feature_names=FEATURE_COLUMNS
    )

    # 6. Generate High-Resolution Scientific Figures
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 11

    # Figure 1: ROC Curves
    plt.figure(figsize=(8, 7), dpi=300)
    for model_name, p, color in [("Logistic Regression", p_lr_test, "#ff7f0e"),
                                 ("Random Forest", p_rf_test, "#2ca02c"),
                                 ("Neural Network", p_nn_test, "#1f77b4")]:
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f"{model_name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) — Spatial Block Holdout (39,749 cells)", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png")
    plt.close()

    # Figure 2: Precision-Recall Curves
    plt.figure(figsize=(8, 7), dpi=300)
    for model_name, p, color in [("Logistic Regression", p_lr_test, "#ff7f0e"),
                                 ("Random Forest", p_rf_test, "#2ca02c"),
                                 ("Neural Network", p_nn_test, "#1f77b4")]:
        prec, rec, _ = precision_recall_curve(y_test, p)
        pr_val = average_precision_score(y_test, p)
        plt.plot(rec, prec, color=color, lw=2.5, label=f"{model_name} (PR-AUC = {pr_val:.3f})")

    prev = np.mean(y_test)
    plt.axhline(prev, color="red", linestyle=":", lw=1.8, label=f"Positive Prevalence Baseline ({prev:.1%})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (True Positive Rate)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title("Precision-Recall Curves — Spatial Block Holdout", fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curves.png")
    plt.close()

    # Figure 3: Reliability Diagrams (Probability Calibration)
    plt.figure(figsize=(8, 7), dpi=300)
    # Perfectly calibrated
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated (Ideal)")

    # Uncalibrated NN
    prob_true_nn, prob_pred_nn = calibration_curve(y_test, p_nn_test, n_bins=10)
    brier_nn = brier_score_loss(y_test, p_nn_test)
    plt.plot(prob_pred_nn, prob_true_nn, "s-", color="#1f77b4", lw=2, label=f"NN Uncalibrated (Brier = {brier_nn:.4f})")

    # Platt Calibrated NN
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, p_nn_test_platt, n_bins=10)
    brier_cal = brier_score_loss(y_test, p_nn_test_platt)
    plt.plot(prob_pred_cal, prob_true_cal, "o-", color="#0077b6", lw=2.5, label=f"NN Platt Calibrated (Brier = {brier_cal:.4f})")

    # Random Forest
    prob_true_rf, prob_pred_rf = calibration_curve(y_test, p_rf_test, n_bins=10)
    brier_rf = brier_score_loss(y_test, p_rf_test)
    plt.plot(prob_pred_rf, prob_true_rf, "^-", color="#2ca02c", lw=2, label=f"Random Forest (Brier = {brier_rf:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Mean Predicted Probability", fontweight="bold")
    plt.ylabel("Empirical Inundation Fraction (Observed)", fontweight="bold")
    plt.title("Reliability Diagram & Calibration Curve (Validation-Fitted)", fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "reliability_diagrams.png")
    plt.close()

    # Figure 4: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    models_cm = [
        ("Neural Network", y_test, p_nn_test, thresholds["neural_network"], axes[0]),
        ("Random Forest", y_test, p_rf_test, thresholds["random_forest"], axes[1]),
        ("Logistic Regression", y_test, p_lr_test, thresholds["logistic_regression"], axes[2]),
    ]
    for m_title, y_t, p_v, th, ax in models_cm:
        pred_b = (p_v >= th).astype(int)
        cm = confusion_matrix(y_t, pred_b)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Predicted Dry", "Predicted Flooded"],
                    yticklabels=["Actual Dry", "Actual Flooded"])
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.7, f"({cm_norm[i, j]:.1%})", ha="center", va="center", color="gray", fontsize=9)
        ax.set_title(f"{m_title}\n(Threshold τ = {th:.4f})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png")
    plt.close()

    # Figure 5: Permutation Feature Importance Comparison
    plt.figure(figsize=(10, 6), dpi=300)
    feat_names_pretty = [FEATURE_LABELS[f] for f in FEATURE_COLUMNS]
    rf_drops = [perm_rf[f]["mean_drop"] for f in FEATURE_COLUMNS]
    nn_drops = [perm_nn[f]["mean_drop"] for f in FEATURE_COLUMNS]

    y_pos = np.arange(len(FEATURE_COLUMNS))
    height = 0.35

    plt.barh(y_pos + height / 2, rf_drops, height, label="Random Forest (ROC-AUC drop)", color="#2ca02c", alpha=0.9)
    plt.barh(y_pos - height / 2, nn_drops, height, label="Neural Network (ROC-AUC drop)", color="#1f77b4", alpha=0.9)

    plt.yticks(y_pos, feat_names_pretty)
    plt.xlabel("Mean Drop in ROC-AUC when Feature is Permuted on Unseen Test Blocks", fontweight="bold")
    plt.title("Permutation Feature Importance on Held-Out Spatial Blocks (n=5 repeats)", fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.axvline(0, color="black", lw=1)
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_importance.png")
    plt.close()

    # Figure 6: Generalisation Protocol Comparison
    plt.figure(figsize=(11, 5.5), dpi=300)
    plot_df = df_metrics[df_metrics["model"].isin(["Neural Network", "Neural Network (Uncalibrated)", "Random Forest", "Logistic Regression"])].copy()
    plot_df["model_clean"] = plot_df["model"].replace({"Neural Network (Uncalibrated)": "Neural Network"})

    sns.barplot(
        data=plot_df,
        x="split",
        y="roc_auc",
        hue="model_clean",
        palette=["#ff7f0e", "#2ca02c", "#1f77b4"]
    )
    plt.ylim([0.5, 1.0])
    plt.ylabel("ROC-AUC Score", fontweight="bold")
    plt.xlabel("Generalisation Evaluation Protocol", fontweight="bold")
    plt.title("Spatial vs Buffered vs Cross-Event Generalisation Performance", fontweight="bold", pad=12)
    plt.legend(title="Model", frameon=True)
    plt.tight_layout()
    plt.savefig(figures_dir / "generalisation_comparison.png")
    plt.close()

    logger.info("All scientific figures generated successfully in %s", figures_dir)

    # 7. Save complete JSON summary for app consumption
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics_list,
        "feature_importance": {
            "random_forest": perm_rf,
            "neural_network": perm_nn
        },
        "brier_comparison": {
            "prevalence_baseline": float(np.mean(y_test) * (1.0 - np.mean(y_test))),
            "uncalibrated_nn": brier_score_loss(y_test, p_nn_test),
            "platt_calibrated_nn": brier_score_loss(y_test, p_nn_test_platt),
            "isotonic_calibrated_nn": brier_score_loss(y_test, p_nn_test_iso),
            "random_forest": brier_score_loss(y_test, p_rf_test)
        },
        "sample_counts": {
            "spatial_block_train": len(train_df),
            "spatial_block_val": len(val_df),
            "spatial_block_test": len(test_df),
            "buffered_block_test": len(buffered_test_df),
            "cross_event_test": len(ev2_df)
        }
    }
    with open(reports_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 8. Save Phase 3 Reproducibility Manifest
    project_root = Path(".")
    manifest = generate_reproducibility_manifest(reports_dir, project_root)
    summary_data["reproducibility_manifest"] = manifest

    logger.info("Scientific validation pipeline finished successfully!")
    return df_metrics, summary_data


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_reproducibility_manifest(reports_dir: Path, project_root: Path) -> dict:
    """Generate and write comprehensive reproducibility manifest."""
    manifest = {
        "metadata": {
            "title": "GeoFlood AI Phase 3 Scientific Validation Reproducibility Manifest",
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "dependencies": {
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__
        },
        "dataset_artifacts": {
            "features_parquet": {
                "path": str(Path("data/processed/features.parquet")),
                "sha256": get_file_sha256(project_root / "data/processed/features.parquet")
            },
            "predictions_parquet": {
                "path": str(Path("outputs/predictions.parquet")),
                "sha256": get_file_sha256(project_root / "outputs/predictions.parquet")
            }
        },
        "model_artifacts": {
            "logistic_regression": {
                "path": str(Path("models/logistic_regression.joblib")),
                "sha256": get_file_sha256(project_root / "models/logistic_regression.joblib")
            },
            "random_forest": {
                "path": str(Path("models/random_forest.joblib")),
                "sha256": get_file_sha256(project_root / "models/random_forest.joblib")
            },
            "neural_network": {
                "path": str(Path("models/neural_network.pt")),
                "sha256": get_file_sha256(project_root / "models/neural_network.pt")
            },
            "scaler": {
                "path": str(Path("models/scaler.joblib")),
                "sha256": get_file_sha256(project_root / "models/scaler.joblib")
            },
            "model_metadata": {
                "path": str(Path("models/model_metadata.json")),
                "sha256": get_file_sha256(project_root / "models/model_metadata.json")
            }
        },
        "protocols": {
            "random_seed": 42,
            "crs_source": "EPSG:4326",
            "crs_display": "EPSG:3857",
            "raster_dimensions": [400, 623],
            "total_cells_per_event": 249200,
            "spatial_blocks": {
                "train_blocks": [0, 1, 3, 4, 6, 8, 9],
                "validation_blocks": [5, 11],
                "test_blocks": [2, 7, 10],
                "buffer_exclusion_distance_km": 1.0
            },
            "events": {
                "primary_event": "DFO_4259 (June 2015)",
                "cross_event_holdout": "DFO_4288 (August 2015)"
            }
        }
    }
    manifest_path = reports_dir / "phase3_reproducibility.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved reproducibility manifest to %s", manifest_path)
    return manifest



if __name__ == "__main__":
    execute_scientific_validation()
