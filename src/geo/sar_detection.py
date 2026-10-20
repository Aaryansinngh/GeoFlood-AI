"""
src/geo/sar_detection.py

Stretch Goal A: Sentinel-1 Synthetic Aperture Radar (SAR) Flood Detection.
Processes a pre/post-flood SAR VV-backscatter image pair:
1. Computes backscatter difference (Post - Pre in dB or ratio)
2. Applies Otsu / bimodal thresholding to isolate open water surface backscatter reduction
3. Masks out permanent water bodies (JRC Global Surface Water)
4. Computes detection metrics against optical reference inundation (IoU, Precision, Recall)

DISCLAIMER & LABEL:
"Detection of observed flooding (satellite observation), NOT prediction or susceptibility modeling."
"""

import numpy as np
from typing import Dict, Any, Tuple


def otsu_threshold(diff_array: np.ndarray, num_bins: int = 256) -> float:
    """
    Compute Otsu threshold on the valid values of a 2D backscatter difference matrix.
    Finds the threshold that minimizes intra-class variance.
    """
    valid = diff_array[~np.isnan(diff_array)]
    if len(valid) == 0:
        return 0.0

    counts, bin_edges = np.histogram(valid, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = valid.size
    sum_total = np.dot(counts, bin_centers)

    weight_bg = 0.0
    sum_bg = 0.0
    max_variance = 0.0
    best_thresh = float(bin_centers[0])

    for i in range(num_bins):
        weight_bg += counts[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += counts[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance_between = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
        if variance_between > max_variance:
            max_variance = variance_between
            best_thresh = float(bin_centers[i])

    return best_thresh


def process_sar_flood_detection(
    pre_vv_db: np.ndarray,
    post_vv_db: np.ndarray,
    perm_water_mask: np.ndarray,
    reference_mask: np.ndarray,
    threshold_db: float = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Execute SAR backscatter difference detection.
    Specular reflection on smooth open water causes a sharp drop in SAR backscatter.
    """
    # 1. Backscatter difference in dB: diff = post - pre
    # Inundated areas exhibit negative values (drop in backscatter)
    diff_db = post_vv_db - pre_vv_db

    # 2. Threshold selection
    if threshold_db is None:
        # Compute automatic Otsu threshold on negative shifts
        threshold_db = otsu_threshold(diff_db)
        # Typically open water requires a drop of at least -2.5 dB
        threshold_db = min(threshold_db, -2.5)

    # 3. Raw water detection
    sar_water_detected = (diff_db <= threshold_db).astype(int)

    # 4. Mask out permanent water (to isolate newly flooded dry land)
    sar_flood_mask = np.where(perm_water_mask == 1, 0, sar_water_detected)

    # 5. Evaluate against reference flood mask
    valid_eval = (~np.isnan(reference_mask)) & (perm_water_mask == 0)
    ref_binary = (reference_mask[valid_eval] == 1).astype(int)
    sar_binary = sar_flood_mask[valid_eval]

    intersection = np.sum((ref_binary == 1) & (sar_binary == 1))
    union = np.sum((ref_binary == 1) | (sar_binary == 1))
    iou = float(intersection / union) if union > 0 else 0.0

    tp = np.sum((ref_binary == 1) & (sar_binary == 1))
    fp = np.sum((ref_binary == 0) & (sar_binary == 1))
    fn = np.sum((ref_binary == 1) & (sar_binary == 0))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    metrics = {
        "threshold_db": float(threshold_db),
        "iou": round(iou, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "total_flooded_sar_cells": int(np.sum(sar_flood_mask == 1)),
        "total_reference_flooded_cells": int(np.sum(ref_binary == 1)),
        "method": "Sentinel-1 VV Backscatter Difference + Otsu Thresholding",
        "disclaimer": "Detection of observed flooding from satellite radar, NOT predictive susceptibility modeling."
    }

    return sar_flood_mask, metrics
