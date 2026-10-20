"""
src/geo/location_analysis.py

Coordinate-based flood susceptibility analysis engine for GeoFlood AI.
Provides:
1. Coordinate validation and study-area boundary checks
2. Grid-cell identification and feature retrieval
3. Model-specific inference with leakage-aware preprocessing
4. Exploratory rainfall scenario analysis
5. Missing-feature handling (preserves integrity; never converts missing data to 'safe')
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from src.models.predict import GeoFloodPredictor
from src.models.fuzzy_layer import MamdaniRiskFuzzySystem

FEATURE_COLUMNS = [
    "rain_event_sum",
    "rain_3d",
    "rain_7d",
    "elevation",
    "slope",
    "tpi",
    "relative_elevation"
]

# Documented physical bounds for monsoonal rainfall in Assam
RAINFALL_BOUNDS = {
    "rain_event_sum": {"min": 0.0, "max": 2500.0, "unit": "mm", "desc": "Event-window cumulative precipitation"},
    "rain_3d": {"min": 0.0, "max": 350.0, "unit": "mm", "desc": "3-day pre-event cumulative precipitation"},
    "rain_7d": {"min": 0.0, "max": 700.0, "unit": "mm", "desc": "7-day pre-event cumulative precipitation"},
}

# Empirical training distribution bounds observed across 498,400 cells
TRAINING_RAINFALL_DISTRIBUTION = {
    "rain_3d": {"min": 4.5, "max": 103.3, "p99": 98.4, "mean": 51.4, "unit": "mm"},
    "rain_7d": {"min": 26.1, "max": 235.0, "p99": 220.1, "mean": 104.5, "unit": "mm"},
    "rain_event_sum": {"min": 381.5, "max": 1300.2, "p99": 1250.0, "mean": 747.9, "unit": "mm"}
}


MODEL_METADATA_INFO = {
    "neural_network": {
        "name": "Neural Network (Primary)",
        "short_name": "Neural Net",
        "arch": "Feed-Forward PyTorch (Linear 7->32->16->1, ReLU, Sigmoid)",
        "input_prep": "StandardScaler normalized (zero mean, unit variance)",
        "non_linear": "Continuous smooth non-linear representation via backpropagation"
    },
    "random_forest": {
        "name": "Random Forest (Baseline 2)",
        "short_name": "Random Forest",
        "arch": "100 Decision Trees with balanced class weights (Scikit-Learn)",
        "input_prep": "Raw unscaled physical features",
        "non_linear": "Piecewise-constant orthogonal decision boundaries via bootstrap aggregation"
    },
    "logistic_regression": {
        "name": "Logistic Regression (Baseline 1)",
        "short_name": "Logistic Reg",
        "arch": "L2-Regularized Generalized Linear Model with Sigmoid link",
        "input_prep": "StandardScaler normalized (zero mean, unit variance)",
        "non_linear": "Monotonic linear log-odds combination (cannot learn spatial non-linear thresholds)"
    }
}


class LocationAnalyzer:
    def __init__(self,
                 df_features: pd.DataFrame,
                 config: dict,
                 predictor: Optional[GeoFloodPredictor] = None,
                 fuzzy_sys: Optional[MamdaniRiskFuzzySystem] = None):
        self.df_features = df_features[df_features["event_id"] == "DFO_4259"].copy() if "event_id" in df_features.columns else df_features.copy()
        self.config = config
        self.bbox = config["region"]["bbox"]
        self.predictor = predictor or GeoFloodPredictor("models")
        self.fuzzy_sys = fuzzy_sys or MamdaniRiskFuzzySystem()
        self.metadata = self.predictor.metadata
        self.thresholds = self.metadata.get("thresholds", {
            "logistic_regression": 0.5888,
            "random_forest": 0.0560,
            "neural_network": 0.4394
        })

        # Load real held-out test evaluation metrics
        metrics_file = self.predictor.models_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                self.saved_metrics = json.load(f)
        else:
            self.saved_metrics = {}

    def validate_coordinates(self, lat: Any, lon: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate latitude and longitude inputs.
        Returns (is_valid, error_message).
        """
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return False, "Coordinates must be valid numeric floating-point values."

        if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
            return False, "Coordinates cannot be NaN or infinite."

        if not (-90.0 <= lat <= 90.0):
            return False, f"Latitude {lat:.4f}° is outside the global range [-90.0, 90.0]."

        if not (-180.0 <= lon <= 180.0):
            return False, f"Longitude {lon:.4f}° is outside the global range [-180.0, 180.0]."

        # Study area bounds check
        min_lat = self.bbox["min_lat"]
        max_lat = self.bbox["max_lat"]
        min_lon = self.bbox["min_lon"]
        max_lon = self.bbox["max_lon"]

        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            region_name = self.config["region"]["name"]
            desc = self.config["region"]["description"]
            return False, (
                f"Location ({lat:.4f}°N, {lon:.4f}°E) is outside the supported study area for {region_name} ({desc}). "
                f"Supported coverage is restricted to: Latitude [{min_lat:.2f}°, {max_lat:.2f}°] and "
                f"Longitude [{min_lon:.2f}°, {max_lon:.2f}°]."
            )

        return True, None

    def find_nearest_cell(self, lat: float, lon: float) -> pd.Series:
        """Find the grid cell closest to the requested coordinate."""
        distances = (self.df_features["lat"] - lat) ** 2 + (self.df_features["lon"] - lon) ** 2
        nearest_idx = distances.idxmin()
        return self.df_features.loc[nearest_idx]

    def analyze_location(self,
                         lat: float,
                         lon: float,
                         model_type: str = "neural_network",
                         rainfall_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Execute full location analysis:
        1. Validate coordinates
        2. Identify grid cell and extract features
        3. Evaluate model inference (with optional rainfall scenario)
        4. Evaluate Mamdani fuzzy risk
        5. Return comprehensive structured result
        """
        is_valid, err_msg = self.validate_coordinates(lat, lon)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg,
                "lat": lat,
                "lon": lon,
                "is_inside_region": False
            }

        lat = float(lat)
        lon = float(lon)
        cell = self.find_nearest_cell(lat, lon)

        # Check for missing features
        missing_features = []
        feature_dict = {}
        for feat in FEATURE_COLUMNS:
            val = cell.get(feat, np.nan)
            if pd.isna(val):
                missing_features.append(feat)
                feature_dict[feat] = np.nan
            else:
                feature_dict[feat] = float(val)

        is_perm_water = int(cell.get("is_perm_water", 0))
        is_scenario = False

        # Apply rainfall scenario overrides if supplied
        if rainfall_overrides:
            for k, v in rainfall_overrides.items():
                if k in RAINFALL_BOUNDS and v is not None:
                    # Validate bounds
                    b_min = RAINFALL_BOUNDS[k]["min"]
                    b_max = RAINFALL_BOUNDS[k]["max"]
                    clamped_v = float(np.clip(float(v), b_min, b_max))
                    feature_dict[k] = clamped_v
                    is_scenario = True

        # Check if missing features prevent inference
        if missing_features:
            return {
                "success": True,
                "is_inside_region": True,
                "cell_id": str(cell.get("cell_id", "Unknown")),
                "cell_lat": float(cell["lat"]),
                "cell_lon": float(cell["lon"]),
                "input_lat": lat,
                "input_lon": lon,
                "model_type": model_type,
                "is_perm_water": is_perm_water,
                "has_missing_features": True,
                "missing_features": missing_features,
                "susceptibility_prob": None,
                "risk_category": "Unavailable",
                "binary_decision": "Unavailable",
                "ground_truth": "Unavailable",
                "features": feature_dict,
                "warning": (
                    f"Required physical feature(s) {missing_features} are missing / unobserved for this cell. "
                    "In adherence to data integrity protocols, this cell is marked 'Unavailable' and is NOT treated as safe."
                ),
                "is_scenario": is_scenario
            }

        # Model Inference
        model_prob = self.predictor.predict_point(feature_dict, model_type=model_type)
        threshold = float(self.thresholds.get(model_type, 0.5))

        # Risk categorization based on project rules
        if is_perm_water == 1:
            risk_cat = "Permanent Water"
            binary_dec = "Permanent Water Channel"
        else:
            if model_prob >= 0.70:
                risk_cat = "High Risk"
            elif model_prob >= 0.35:
                risk_cat = "Moderate Risk"
            else:
                risk_cat = "Low Risk"

            binary_dec = "Susceptible (Flood Prone)" if model_prob >= threshold else "Non-Susceptible (Resilient)"

        # Fuzzy inference
        f_rain = float(feature_dict.get("rain_3d", 45.0))
        fuzzy_score, fuzzy_cat = self.fuzzy_sys.evaluate(model_prob, f_rain)

        # Ground truth status
        f_label = cell.get("flood_label", np.nan)
        if is_perm_water == 1:
            ground_truth = "Permanent River Body (JRC)"
        elif f_label == 1.0:
            ground_truth = "Historically Inundated (DFO_4259)"
        elif f_label == 0.0:
            ground_truth = "Historically Remained Dry (DFO_4259)"
        else:
            ground_truth = "Unobserved / Cloud Gap"

        return {
            "success": True,
            "is_inside_region": True,
            "cell_id": str(cell.get("cell_id", "Unknown")),
            "cell_lat": float(cell["lat"]),
            "cell_lon": float(cell["lon"]),
            "input_lat": lat,
            "input_lon": lon,
            "model_type": model_type,
            "threshold": threshold,
            "susceptibility_prob": round(float(model_prob), 4),
            "risk_category": risk_cat,
            "binary_decision": binary_dec,
            "fuzzy_score": fuzzy_score,
            "fuzzy_category": fuzzy_cat,
            "is_perm_water": is_perm_water,
            "has_missing_features": False,
            "ground_truth": ground_truth,
            "features": feature_dict,
            "is_scenario": is_scenario,
            "baseline_event_rain": float(cell.get("rain_event_sum", 0.0)),
            "baseline_3d_rain": float(cell.get("rain_3d", 0.0)),
            "baseline_7d_rain": float(cell.get("rain_7d", 0.0))
        }

    def compare_models(self,
                       lat: float,
                       lon: float,
                       rainfall_overrides: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compare all three supported models (Neural Network, Random Forest, Logistic Regression)
        for the given geographic location.
        Evaluates:
        1. Baseline susceptibility predictions for each model
        2. Exploratory scenario susceptibility predictions (if rainfall_overrides provided)
        3. Threshold-based decisions and risk categories
        4. Official held-out spatial block evaluation metrics from metrics.json
        5. Preserves terrain features and flags missing data safely
        """
        is_valid, err_msg = self.validate_coordinates(lat, lon)
        if not is_valid:
            return {
                "success": False,
                "error": err_msg,
                "lat": lat,
                "lon": lon,
                "is_inside_region": False
            }

        lat = float(lat)
        lon = float(lon)
        cell = self.find_nearest_cell(lat, lon)

        # Check for missing features
        missing_features = []
        baseline_features = {}
        for feat in FEATURE_COLUMNS:
            val = cell.get(feat, np.nan)
            if pd.isna(val):
                missing_features.append(feat)
                baseline_features[feat] = np.nan
            else:
                baseline_features[feat] = float(val)

        is_perm_water = int(cell.get("is_perm_water", 0))

        # Check ground truth
        f_label = cell.get("flood_label", np.nan)
        if is_perm_water == 1:
            ground_truth = "Permanent River Body (JRC)"
        elif f_label == 1.0:
            ground_truth = "Historically Inundated (DFO_4259)"
        elif f_label == 0.0:
            ground_truth = "Historically Remained Dry (DFO_4259)"
        else:
            ground_truth = "Unobserved / Cloud Gap"

        # If features missing, return structured missing data response
        if missing_features:
            models_res = {}
            for m_key in ["neural_network", "random_forest", "logistic_regression"]:
                meta = MODEL_METADATA_INFO.get(m_key, {})
                thresh = float(self.thresholds.get(m_key, 0.5))
                m_metrics = self.saved_metrics.get("spatial_block_holdout", {}).get(m_key, {})
                models_res[m_key] = {
                    "key": m_key,
                    "name": meta.get("name", m_key),
                    "short_name": meta.get("short_name", m_key),
                    "arch": meta.get("arch", ""),
                    "input_prep": meta.get("input_prep", ""),
                    "non_linear": meta.get("non_linear", ""),
                    "baseline_prob": None,
                    "scenario_prob": None,
                    "delta_prob": None,
                    "baseline_category": "Unavailable",
                    "scenario_category": "Unavailable",
                    "threshold": thresh,
                    "baseline_decision": "Unavailable",
                    "scenario_decision": "Unavailable",
                    "test_metrics": m_metrics
                }

            return {
                "success": True,
                "is_inside_region": True,
                "cell_id": str(cell.get("cell_id", "Unknown")),
                "cell_lat": float(cell["lat"]),
                "cell_lon": float(cell["lon"]),
                "input_lat": lat,
                "input_lon": lon,
                "is_perm_water": is_perm_water,
                "has_missing_features": True,
                "missing_features": missing_features,
                "ground_truth": ground_truth,
                "warning": (
                    f"Required physical feature(s) {missing_features} are missing / unobserved for this cell. "
                    "In adherence to data integrity protocols, this cell is marked 'Unavailable' and is NOT treated as safe."
                ),
                "baseline_features": baseline_features,
                "scenario_features": baseline_features,
                "is_scenario": False,
                "models": models_res
            }

        # Build scenario features if overrides are present
        scenario_features = baseline_features.copy()
        is_scenario = False
        applied_overrides = {}
        if rainfall_overrides:
            for k, v in rainfall_overrides.items():
                if k in RAINFALL_BOUNDS and v is not None:
                    b_min = RAINFALL_BOUNDS[k]["min"]
                    b_max = RAINFALL_BOUNDS[k]["max"]
                    clamped_v = float(np.clip(float(v), b_min, b_max))
                    scenario_features[k] = clamped_v
                    applied_overrides[k] = clamped_v
                    is_scenario = True

        is_physically_plausible = True
        plausibility_warning = None
        is_extrapolating = False
        extrapolation_warnings = []

        if is_scenario:
            r3 = scenario_features.get("rain_3d", 0.0)
            r7 = scenario_features.get("rain_7d", 0.0)
            r_sum = scenario_features.get("rain_event_sum", 0.0)

            # Plausibility check: 3-day rainfall is a subset of 7-day, which is a subset of event total
            if r3 > r7:
                is_physically_plausible = False
                plausibility_warning = f"Physical Inconsistency: 3-day pre-event rainfall ({r3:.1f} mm) cannot exceed 7-day rainfall ({r7:.1f} mm)."
            elif r7 > r_sum:
                is_physically_plausible = False
                plausibility_warning = f"Physical Inconsistency: 7-day rainfall ({r7:.1f} mm) cannot exceed total event-window rainfall ({r_sum:.1f} mm)."

            # Extrapolation check against training data empirical bounds
            for r_k, r_val in [("rain_3d", r3), ("rain_7d", r7), ("rain_event_sum", r_sum)]:
                dist = TRAINING_RAINFALL_DISTRIBUTION.get(r_k, {})
                if dist and r_val < dist["min"]:
                    is_extrapolating = True
                    extrapolation_warnings.append(f"{r_k} ({r_val:.1f} mm) is below the minimum observed training value ({dist['min']:.1f} mm).")
                elif dist and r_val > dist["max"]:
                    is_extrapolating = True
                    extrapolation_warnings.append(f"{r_k} ({r_val:.1f} mm) exceeds the maximum observed training value ({dist['max']:.1f} mm).")

        def _categorize(prob: float, is_water: int, thresh: float):
            if is_water == 1:
                return "Permanent Water", "Permanent River Channel"
            if prob >= 0.70:
                cat = "High Risk"
            elif prob >= 0.35:
                cat = "Moderate Risk"
            else:
                cat = "Low Risk"
            dec = "Susceptible (Flood Prone)" if prob >= thresh else "Non-Susceptible (Resilient)"
            return cat, dec

        models_res = {}
        for m_key in ["neural_network", "random_forest", "logistic_regression"]:
            meta = MODEL_METADATA_INFO.get(m_key, {})
            thresh = float(self.thresholds.get(m_key, 0.5))
            m_metrics = self.saved_metrics.get("spatial_block_holdout", {}).get(m_key, {})

            base_p = float(self.predictor.predict_point(baseline_features, model_type=m_key))
            base_cat, base_dec = _categorize(base_p, is_perm_water, thresh)

            if is_scenario:
                scen_p = float(self.predictor.predict_point(scenario_features, model_type=m_key))
                delta_p = scen_p - base_p
                scen_cat, scen_dec = _categorize(scen_p, is_perm_water, thresh)
            else:
                scen_p = base_p
                delta_p = 0.0
                scen_cat = base_cat
                scen_dec = base_dec

            models_res[m_key] = {
                "key": m_key,
                "name": meta.get("name", m_key),
                "short_name": meta.get("short_name", m_key),
                "arch": meta.get("arch", ""),
                "input_prep": meta.get("input_prep", ""),
                "non_linear": meta.get("non_linear", ""),
                "baseline_prob": round(base_p, 4),
                "scenario_prob": round(scen_p, 4),
                "delta_prob": round(delta_p, 4),
                "baseline_category": base_cat,
                "scenario_category": scen_cat,
                "threshold": thresh,
                "baseline_decision": base_dec,
                "scenario_decision": scen_dec,
                "test_metrics": m_metrics
            }

        return {
            "success": True,
            "is_inside_region": True,
            "cell_id": str(cell.get("cell_id", "Unknown")),
            "cell_lat": float(cell["lat"]),
            "cell_lon": float(cell["lon"]),
            "input_lat": lat,
            "input_lon": lon,
            "is_perm_water": is_perm_water,
            "has_missing_features": False,
            "ground_truth": ground_truth,
            "baseline_features": baseline_features,
            "scenario_features": scenario_features,
            "is_scenario": is_scenario,
            "applied_overrides": applied_overrides,
            "is_physically_plausible": is_physically_plausible,
            "plausibility_warning": plausibility_warning,
            "is_extrapolating": is_extrapolating,
            "extrapolation_warnings": extrapolation_warnings,
            "models": models_res
        }

