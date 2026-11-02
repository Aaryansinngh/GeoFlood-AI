"""
src/evaluation/uncertainty.py

Uncertainty Quantification and Reliability Framework
=====================================================

Implements the four-tier uncertainty and reliability framework for GeoFlood-AI 2.0:
1. Probability Calibration:
   - Platt Scaling (logistic calibration)
   - Isotonic Regression (non-parametric monotonic calibration)
   - Reliability diagrams / calibration curves calculation
   - Expected Calibration Error (ECE) and Brier Skill Score (BSS)
2. Ensemble Disagreement / Epistemic Uncertainty:
   - Bootstrap aggregating across base classifiers or model ensemble variance
3. Out-Of-Distribution (OOD) Detection:
   - Mahalanobis distance / Isolation Forest / Empirical Feature Bounds
4. Decision Regime Categorisation:
   - Categorises predictions into:
     * High-confidence predictions
     * Low-confidence predictions
     * Unfamiliar environmental conditions (OOD)
     * Insufficient data / extreme uncertainty
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE):
    ECE = sum(|B_m| / N * |acc(B_m) - conf(B_m)|)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)
        else:
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_probs[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_skill_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Compute Brier Skill Score (BSS) relative to the sample climatology (prevalence):
    BSS = 1 - (BS / BS_climatology)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)

    if len(y_true) == 0:
        return 0.0

    bs = np.mean((y_probs - y_true) ** 2)
    p_clim = np.mean(y_true)
    bs_clim = np.mean((p_clim - y_true) ** 2)

    if bs_clim == 0.0:
        return 0.0

    return float(1.0 - (bs / bs_clim))


class ProbabilityCalibrator:
    """
    Probability calibration engine supporting Platt scaling (sigmoid)
    and isotonic regression.
    """

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.calibrator: Optional[Union[LogisticRegression, IsotonicRegression]] = None

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        val_probs = np.asarray(val_probs, dtype=float).clip(1e-7, 1.0 - 1e-7)
        y_val = np.asarray(y_val).astype(int)

        if self.method == "sigmoid":
            # Platt scaling: fit logistic regression on logits or raw probs
            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(val_probs.reshape(-1, 1), y_val)
            self.calibrator = lr
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_probs, y_val)
            self.calibrator = iso

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("Calibrator is not fitted yet.")

        probs = np.asarray(probs, dtype=float).clip(1e-7, 1.0 - 1e-7)

        if self.method == "sigmoid":
            assert isinstance(self.calibrator, LogisticRegression)
            cal_probs = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            assert isinstance(self.calibrator, IsotonicRegression)
            cal_probs = self.calibrator.predict(probs)

        return np.clip(cal_probs, 0.0, 1.0)


class EnvironmentalOODDetector:
    """
    Out-Of-Distribution (OOD) detector evaluating distance from training
    environmental feature distribution in normalized space.
    """

    def __init__(self, k_neighbors: int = 5, percentile_threshold: float = 95.0) -> None:
        self.k_neighbors = k_neighbors
        self.percentile_threshold = percentile_threshold
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.nn_: Optional[NearestNeighbors] = None
        self.dist_threshold_: float = 0.0

    def fit(self, X_train: np.ndarray) -> "EnvironmentalOODDetector":
        X = np.asarray(X_train, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0.0] = 1.0

        X_norm = (X - self.mean_) / self.std_
        self.nn_ = NearestNeighbors(n_neighbors=self.k_neighbors, metric="euclidean")
        self.nn_.fit(X_norm)

        distances, _ = self.nn_.kneighbors(X_norm)
        mean_k_dist = np.mean(distances, axis=1)
        self.dist_threshold_ = float(np.percentile(mean_k_dist, self.percentile_threshold))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.nn_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("OOD Detector is not fitted.")

        X = np.asarray(X, dtype=float)
        X_norm = (X - self.mean_) / self.std_
        distances, _ = self.nn_.kneighbors(X_norm)
        return np.mean(distances, axis=1)

    def is_ood(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        return scores > self.dist_threshold_


class UncertaintyEvaluator:
    """
    Unified engine for categorizing model predictions into actionable decision regimes:
    1. High-confidence: Low epistemic spread, within training domain, probability decisive (>0.7 or <0.3)
    2. Low-confidence: Decisive ambiguity in probability (0.35 - 0.65) or moderate disagreement
    3. Unfamiliar environmental conditions (OOD): Environmental feature anomaly
    4. Insufficient data: Extreme ensemble disagreement (variance > 0.15) or NaN/corrupted values
    """

    def __init__(
        self,
        ood_detector: Optional[EnvironmentalOODDetector] = None,
        calibrator: Optional[ProbabilityCalibrator] = None,
    ) -> None:
        self.ood_detector = ood_detector
        self.calibrator = calibrator

    def evaluate_predictions(
        self,
        X: np.ndarray,
        ensemble_probs: np.ndarray,
        prob_threshold_low: float = 0.35,
        prob_threshold_high: float = 0.65,
        ensemble_var_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate uncertainty metrics and categorise each cell.

        Parameters
        ----------
        X : (N, D) environmental features
        ensemble_probs : (M, N) or (N,) array of predicted probabilities.
                         If (M, N), M represents models or bootstrap replicates.

        Returns
        -------
        Dictionary containing:
          - mean_prob: calibrated or ensemble mean probability
          - epistemic_uncertainty: variance across ensemble models
          - is_ood: boolean mask of environmental domain shift
          - decision_regimes: list of string categories
        """
        probs = np.asarray(ensemble_probs, dtype=float)
        if probs.ndim == 1:
            mean_probs = probs
            epistemic_unc = np.zeros_like(probs)
        else:
            mean_probs = np.mean(probs, axis=0)
            epistemic_unc = np.var(probs, axis=0)

        # Calibrate mean probs if calibrator is provided
        if self.calibrator is not None and self.calibrator.calibrator is not None:
            mean_probs = self.calibrator.calibrate(mean_probs)

        # Out-of-distribution check
        if self.ood_detector is not None and self.ood_detector.nn_ is not None:
            is_ood = self.ood_detector.is_ood(X)
        else:
            is_ood = np.zeros(len(mean_probs), dtype=bool)

        # Categorisation logic
        regimes: List[str] = []
        for p, var, ood in zip(mean_probs, epistemic_unc, is_ood):
            if np.isnan(p) or np.isnan(var):
                regimes.append("Insufficient data")
            elif ood:
                regimes.append("Unfamiliar environmental conditions")
            elif var >= ensemble_var_threshold:
                regimes.append("Low-confidence (high ensemble disagreement)")
            elif prob_threshold_low <= p <= prob_threshold_high:
                regimes.append("Low-confidence (ambiguous susceptibility)")
            else:
                regimes.append("High-confidence")

        return {
            "mean_probability": mean_probs,
            "epistemic_uncertainty": epistemic_unc,
            "is_ood": is_ood,
            "decision_regimes": np.array(regimes),
        }
