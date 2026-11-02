"""
tests/test_phase5_uncertainty.py

Unit and integration tests for Phase 5:
- Probability calibration (Platt scaling & Isotonic regression)
- Reliability diagrams & Expected Calibration Error (ECE)
- Brier Skill Score (BSS) calculation
- Out-of-Distribution (OOD) environmental detection
- Four-tier uncertainty decision regime classification
"""

import numpy as np
import pytest

from src.evaluation.uncertainty import (
    ProbabilityCalibrator,
    EnvironmentalOODDetector,
    UncertaintyEvaluator,
    compute_expected_calibration_error,
    compute_brier_skill_score,
)


def test_ece_and_bss_computation():
    # Perfectly calibrated 50/50 test
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8])
    ece = compute_expected_calibration_error(y_true, y_probs, n_bins=5)
    assert 0.0 <= ece <= 1.0

    bss = compute_brier_skill_score(y_true, y_probs)
    assert bss > 0.0  # Decisive predictions should score positive BSS vs base rate

    # Edge case: empty arrays
    assert compute_expected_calibration_error(np.array([]), np.array([])) == 0.0
    assert compute_brier_skill_score(np.array([]), np.array([])) == 0.0


def test_probability_calibrators():
    np.random.seed(42)
    # Generate poorly calibrated overconfident probabilities
    y_val = np.random.choice([0, 1], size=200, p=[0.8, 0.2])
    raw_probs = np.where(y_val == 1, 0.95, 0.05) + np.random.normal(0, 0.1, size=200)
    raw_probs = np.clip(raw_probs, 0.01, 0.99)

    # Sigmoid / Platt scaling
    cal_sig = ProbabilityCalibrator(method="sigmoid").fit(raw_probs, y_val)
    sig_calibrated = cal_sig.calibrate(raw_probs)
    assert sig_calibrated.shape == raw_probs.shape
    assert (sig_calibrated >= 0.0).all() and (sig_calibrated <= 1.0).all()

    # Isotonic calibration
    cal_iso = ProbabilityCalibrator(method="isotonic").fit(raw_probs, y_val)
    iso_calibrated = cal_iso.calibrate(raw_probs)
    assert iso_calibrated.shape == raw_probs.shape
    assert (iso_calibrated >= 0.0).all() and (iso_calibrated <= 1.0).all()


def test_environmental_ood_detector():
    np.random.seed(42)
    # Training distribution centered around 0 with unit variance
    X_train = np.random.normal(0, 1, size=(500, 4))
    ood_detector = EnvironmentalOODDetector(k_neighbors=5, percentile_threshold=95.0)
    ood_detector.fit(X_train)

    # In-distribution test samples from identical distribution
    X_in = np.random.normal(0, 1, size=(100, 4))
    ood_in = ood_detector.is_ood(X_in)
    # For a 95th percentile threshold, expect in-distribution false positive rate ~ 5-15%
    assert np.mean(ood_in) < 0.25

    # Strong out-of-distribution samples (shift by 10 std dev)
    X_out = np.random.normal(10, 1, size=(20, 4))
    ood_out = ood_detector.is_ood(X_out)
    assert np.all(ood_out)


def test_uncertainty_evaluator_decision_regimes():
    np.random.seed(42)
    X_train = np.random.normal(0, 1, size=(200, 3))
    ood_detector = EnvironmentalOODDetector(k_neighbors=3, percentile_threshold=90.0).fit(X_train)

    evaluator = UncertaintyEvaluator(ood_detector=ood_detector)

    # Synthetic test samples representing distinct regimes:
    # 1. High confidence (decisive prob, low variance, in-domain)
    # 2. Low confidence ambiguous (prob around 0.5, low variance)
    # 3. Low confidence disagreement (high variance across models)
    # 4. Out-of-distribution (features far from train)
    X_test = np.array([
        [0.1, 0.2, -0.1],     # in-domain
        [0.0, 0.1, 0.2],      # in-domain
        [0.2, -0.2, 0.0],     # in-domain
        [15.0, 20.0, -10.0],  # obvious OOD
    ])

    ensemble_probs = np.array([
        [0.90, 0.50, 0.90, 0.80],  # Model 1
        [0.92, 0.52, 0.20, 0.85],  # Model 2
        [0.91, 0.48, 0.10, 0.75],  # Model 3
    ])

    results = evaluator.evaluate_predictions(
        X=X_test,
        ensemble_probs=ensemble_probs,
        prob_threshold_low=0.35,
        prob_threshold_high=0.65,
        ensemble_var_threshold=0.08,
    )

    regimes = results["decision_regimes"]
    assert len(regimes) == 4
    assert regimes[0] == "High-confidence"
    assert "ambiguous" in regimes[1]
    assert "disagreement" in regimes[2]
    assert regimes[3] == "Unfamiliar environmental conditions"
