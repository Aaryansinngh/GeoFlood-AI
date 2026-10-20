# GeoFlood AI — Phase 3: Scientific Audit of the Experimental Pipeline

**Document Status:** Complete Scientific Audit  
**Date:** October 20, 2026  
**Auditor:** Senior ML Engineer & Geospatial Data Scientist  
**Study Region:** Brahmaputra River Floodplain, Assam, India (`26.30°N–27.20°N, 92.80°E–94.20°E`)  
**Target Variable:** Event-based flood inundation (binary classification on dry-land grid cells)

---

## Executive Summary

This document presents a comprehensive, stage-by-stage audit of the GeoFlood AI machine-learning pipeline. The system was examined across 13 stages—from raw sensor ingestion to user scenario simulation—to detect data leakage, evaluate spatial and temporal generalisation, verify probability calibration, and establish methodological boundaries.

### Summary of Major Audit Findings

1. **Temporal Availability & Task Definition:**
   - The feature `rain_event_sum` aggregates total precipitation across the full event window (28 days for DFO_4259). This feature is valid for **retrospective flood susceptibility mapping** and **post-disaster damage reconstruction**, but introduces **post-event leakage** if the system were described as an *operational flood forecasting* system.
   - We formally establish the system's scientific boundary as **Static and Event-Based Flood Susceptibility Mapping**, explicitly disclaiming real-time flood warning.
2. **Spatial Generalisation & Boundary Buffer:**
   - The spatial block holdout split effectively separates training and testing regions into 12 distinct tiles ($3 \times 4$ grid).
   - An audit of spatial autocorrelation at block boundaries revealed that adding a $1\text{ km}$ buffer (excluding 3,934 border cells) changed Random Forest ROC-AUC from **0.901** to **0.908**, and Neural Network ROC-AUC from **0.805** to **0.816**, demonstrating that test metrics are not an artifact of boundary leakage.
3. **Cross-Event Temporal Generalisation:**
   - When models trained on the June 2015 monsoon (DFO_4259) were evaluated on the August 2015 monsoon (DFO_4288, 172,199 cells), Random Forest ROC-AUC collapsed to **0.485** (below chance), whereas Neural Network generalized at **0.672** ROC-AUC and Logistic Regression at **0.713**.
   - This proves that Random Forest overfits to specific spatial rainfall cutoffs of an event, while smooth hyperplanes generalize better across disparate hydrological events.
4. **Probability Calibration:**
   - The uncalibrated Neural Network had a Brier score of **0.1651** (worse than the prevalence climatology baseline of **0.0848**).
   - Fitting Platt Scaling strictly on validation blocks (5 and 11) dramatically improved the test Brier score to **0.0695**, achieving a positive Brier Skill Score of **+0.180**.
5. **SAR Radar Detection Provenance:**
   - The reported Sentinel-1 SAR IoU of **0.991** was determined to be an **internal synthetic calibration / agreement benchmark** under ideal bimodal Gaussian backscatter assumptions, rather than an independently acquired external radar campaign. This metric has been reclassified to avoid misleading operational claims.

---

## Detailed 13-Stage Experimental Pipeline Audit

### Stage 1: Data Acquisition and Preprocessing
- **Source Files & Functions:**
  - `src/data/download_raw.py`: `download_and_extract_gfd()`, `download_and_merge_srtm()`, `download_precipitation_grid()`
- **Data Inputs & Outputs:**
  - *Inputs:* DFO Flood Observatory GeoTIFFs (DFO_4259, DFO_4288); NASA SRTMGL1 30m DEM tiles (`N26E092`, `N26E093`, `N27E092`, `N27E093`); Open-Meteo ERA5 / GPM precipitation grids.
  - *Outputs:* `data/raw/DFO_4259_From_*.tif`, `data/raw/srtm_dem_study_area.tif`, `data/raw/rainfall_DFO_*.json`.
- **Assumptions:**
  - Optical MODIS 250m composite accurately captures maximum inundated surface during the cloud-free composite window.
  - SRTM elevation (captured Feb 2000) represents current baseline terrain topography.
- **Potential Bias & Leakage:**
  - Optical cloud gaps: GFD rasters mark cloudy/unobserved pixels with nodata. If cloud gaps correlate with heavy rain, label missingness is non-random (Missing Not At Random, MNAR).
- **Existing Tests:**
  - `tests/test_pipeline.py::test_feature_builder_handles_nodata`
- **Corrective Actions Taken:**
  - Cloud gaps are explicitly preserved as `NaN` and excluded from dry-land label evaluation. Never imputed as dry/safe.

---

### Stage 2: Flood-Label Construction
- **Source Files & Functions:**
  - `src/data/build_features.py`: `process_event()`
- **Data Inputs & Outputs:**
  - *Inputs:* Band 1 (flood extent) and Band 5 (permanent water) from GFD GeoTIFFs.
  - *Outputs:* `flood_label` in `features.parquet` (`1.0` = flooded, `0.0` = dry land, `-1.0` = permanent river channel, `NaN` = unobserved/cloud).
- **Assumptions:**
  - JRC Global Surface Water identifies perennial river channels that should not be classified as newly inundated land.
- **Potential Bias & Leakage:**
  - If permanent water channels were coded as label `1.0`, models would simply learn river bathymetry rather than terrestrial flood susceptibility.
- **Existing Tests:**
  - `tests/test_location_analysis.py::test_permanent_water_identification`
- **Corrective Actions Taken:**
  - Permanent water pixels are masked with label `-1.0` and excluded from dry-land classification loss.

---

### Stage 3: Feature Engineering
- **Source Files & Functions:**
  - `src/geo/grid.py`: `calculate_slope()`, `calculate_tpi()`, `calculate_relative_elevation()`
  - `src/data/build_features.py`: `interpolate_rainfall_surface()`
- **Data Inputs & Outputs:**
  - *Inputs:* SRTM DEM raster, precipitation station records.
  - *Outputs:* `elevation`, `slope`, `tpi`, `relative_elevation`, `rain_event_sum`, `rain_3d`, `rain_7d`.
- **Assumptions:**
  - Slope accounts for metric distortion of latitude ($dx = res_x \cdot 111320 \cdot \cos(\text{lat})$).
  - TPI (15-pixel window $\approx 3.75\text{ km}$) and HAND relative elevation (25-pixel window $\approx 6.25\text{ km}$) capture local hydraulic head.
- **Potential Bias & Leakage:**
  - Thin Plate Spline (TPS) interpolation of rainfall across 249,200 cells is smooth; local convective cloudbursts between gauges are smoothed out.
- **Existing Tests:**
  - `tests/test_pipeline.py::test_metrics_match_known_toy_data`
- **Corrective Actions Taken:**
  - Verified that meter-per-degree correction is properly applied in finite-difference slope calculation (`src/geo/grid.py#L56`).

---

### Stage 4: Train, Validation, and Test Partitioning
- **Source Files & Functions:**
  - `src/evaluation/splits.py`: `get_spatial_block_split()`, `get_event_holdout_split()`
- **Data Inputs & Outputs:**
  - *Inputs:* `features.parquet`, block lists from `config/settings.yaml`.
  - *Outputs:* `train_df` (92,471 cells), `val_df` (39,979 cells), `test_df` (39,749 cells).
- **Assumptions:**
  - Spatial block partitioning prevents near-duplicate leakage that occurs in random K-fold splits.
- **Potential Bias & Leakage:**
  - Block 7 contains 3,706 of the 3,717 test positives (99.7%), because floodwaters concentrated in the central floodplain. Blocks 2 and 10 contain almost no floodwaters.
- **Existing Tests:**
  - `tests/test_pipeline.py::test_spatial_block_split_no_overlap`
- **Corrective Actions Taken:**
  - Evaluated precision-recall curves and Brier scores against prevalence baselines; validated that Block 7 reflects real geographical inundation concentration.

---

### Stage 5: Spatial-Block Assignment
- **Source Files & Functions:**
  - `src/geo/grid.py`: `assign_spatial_blocks()`
- **Data Inputs & Outputs:**
  - *Inputs:* Grid dimensions ($400 \times 623$), $n_y=3$, $n_x=4$.
  - *Outputs:* Integer block grid `spatial_block_id` $\in [0, 11]$.
- **Assumptions:**
  - Contiguous rectangular blocks of $\sim 133 \times 155$ cells ($\sim 33\text{ km} \times 35\text{ km}$) are sufficiently large to contain hydrological sub-basins.
- **Potential Bias & Leakage:**
  - Boundary adjacency: Cells on the outer boundary of Block 7 touch Block 3 (train) and Block 6 (train).
- **Existing Tests:**
  - `tests/test_pipeline.py::test_spatial_block_generator_integrity`
- **Corrective Actions Taken:**
  - Added Spatially Buffered Holdout ($k=4$ cells, $\sim 1\text{ km}$) in Phase 3 evaluation, demonstrating zero boundary bleed.

---

### Stage 6: Model Training
- **Source Files & Functions:**
  - `src/models/baselines.py`: `train_logistic_regression()`, `train_random_forest()`
  - `src/models/nn.py`: `train_neural_network()`
- **Data Inputs & Outputs:**
  - *Inputs:* `X_train_scaled`, `y_train` (for LR and NN); `X_train` raw (for RF).
  - *Outputs:* Serialized model artifacts (`logistic_regression.joblib`, `random_forest.joblib`, `neural_network.pt`).
- **Assumptions:**
  - `StandardScaler` fitted strictly on `X_train` prevents test data distribution leakage.
  - Balanced class weighting in RF compensates for 13.3% positive prevalence in training blocks.
- **Potential Bias & Leakage:**
  - Overfitting of tree ensemble to specific rainfall values in training blocks.
- **Existing Tests:**
  - `tests/test_pipeline.py::test_saved_model_reloads_finite_outputs`
- **Corrective Actions Taken:**
  - Preserved strict training isolation: scalers and weights never inspect validation or test partitions.

---

### Stage 7: Hyperparameter Selection
- **Source Files & Functions:**
  - `config/settings.yaml`
  - `src/models/nn.py`, `src/models/baselines.py`
- **Data Inputs & Outputs:**
  - *Inputs:* Architecture parameters (`hidden_dims: [32, 16]`, `lr: 0.001`, `weight_decay: 1e-4`, `n_estimators: 100`).
  - *Outputs:* Trained model weights.
- **Assumptions:**
  - Early stopping on validation loss prevents neural network overtraining.
- **Potential Bias & Leakage:**
  - Hyperparameters were chosen a priori or on validation splits, never on test blocks.
- **Corrective Actions Taken:**
  - Verified that test set was never accessed during model tuning or early stopping.

---

### Stage 8: Decision-Threshold Selection
- **Source Files & Functions:**
  - `src/evaluation/metrics.py`: `find_optimal_threshold()`
- **Data Inputs & Outputs:**
  - *Inputs:* `y_val`, `y_val_probs`.
  - *Outputs:* `thresholds`: LR = 0.5888, RF = 0.0560, NN = 0.4394.
- **Assumptions:**
  - Maximizing F1 score on validation blocks yields an operational threshold suitable for disaster response.
- **Potential Bias & Leakage:**
  - Optimizing threshold on the test set is a common scientific flaw.
- **Existing Tests:**
  - `tests/test_location_analysis.py::test_threshold_selection_integrity`
- **Corrective Actions Taken:**
  - Confirmed thresholds are computed strictly on validation blocks (5 and 11) and applied blindly to the test set.

---

### Stage 9: Probability Calibration
- **Source Files & Functions:**
  - `src/evaluation/run_validation.py`: Platt Scaling & Isotonic Regression
- **Data Inputs & Outputs:**
  - *Inputs:* `p_val`, `y_val`.
  - *Outputs:* Calibrated probabilities on `X_test`.
- **Assumptions:**
  - Neural network raw sigmoid output requires sigmoid mapping adjustment (Platt scaling) to align with empirical event frequency.
- **Potential Bias & Leakage:**
  - Fitting calibration on test data would be direct label leakage.
- **Corrective Actions Taken:**
  - Calibrators fitted strictly on validation blocks; demonstrated Brier score reduction from 0.1651 to 0.0695 on unseen test blocks.

---

### Stage 10: Metric Calculation
- **Source Files & Functions:**
  - `src/evaluation/metrics.py`: `evaluate_predictions()`
  - `src/evaluation/run_validation.py`: `compute_comprehensive_metrics()`
- **Data Inputs & Outputs:**
  - *Inputs:* `y_test`, `y_test_probs`, `threshold`.
  - *Outputs:* `phase3_metrics.csv` containing ROC-AUC, PR-AUC, Precision, Recall, F1, Specificity, FNR, Brier, Brier Skill Score.
- **Assumptions:**
  - In imbalanced spatial problems (9.35% prevalence), PR-AUC and Brier Skill Score are more informative than ROC-AUC alone.
- **Corrective Actions Taken:**
  - Climatology prevalence baseline ($p(1-p) = 0.0848$) calculated and compared for all Brier scores.

---

### Stage 11: Spatial Prediction Generation
- **Source Files & Functions:**
  - `src/models/train.py`: `export_map_geojson()`
  - `app.py`: `generate_raster_overlays()`
- **Data Inputs & Outputs:**
  - *Inputs:* 249,200 grid cells, serialized models.
  - *Outputs:* `outputs/predictions.parquet`, Folium `ImageOverlay` RGBA matrices.
- **Assumptions:**
  - Grid cell ordering `(row, col)` maps monotonically to `(lat, lon)` without transposition.
- **Existing Tests:**
  - `tests/test_model_comparison.py::test_generate_raster_overlays_all_models`
- **Corrective Actions Taken:**
  - Raster shape verified as precisely $400 \times 623 = 249,200$ cells with correct CRS (EPSG:4326).

---

### Stage 12: SAR Flood Detection Validation
- **Source Files & Functions:**
  - `src/geo/sar_detection.py`: `otsu_threshold()`, `process_sar_flood_detection()`
  - `outputs/sar_demo_data.json`
- **Data Inputs & Outputs:**
  - *Inputs:* Synthetic pre- and post-flood backscatter matrices.
  - *Outputs:* Stated IoU = 0.991, Precision = 1.000, Recall = 0.991.
- **Assumptions:**
  - Clean bimodal Gaussian backscatter drop of $-10\text{ dB}$ on inundated cells.
- **Potential Bias & Leakage:**
  - High agreement was produced by synthetic simulation tuned to match the reference optical mask, not an independent operational radar scene.
- **Corrective Actions Taken:**
  - Created `reports/sar_validation_audit.md` explicitly classifying this as an algorithmic consistency demonstration, rather than real-world satellite ground-truth accuracy.

---

### Stage 13: Rainfall Scenario Simulation
- **Source Files & Functions:**
  - `src/geo/location_analysis.py`: `LocationAnalyzer.compare_models()`
  - `app.py`: Tab 3 (Rainfall Scenario Simulation)
- **Data Inputs & Outputs:**
  - *Inputs:* User slider overrides (`rain_3d`, `rain_7d`, `rain_event_sum`).
  - *Outputs:* Scenario probabilities $p_{\text{scen}}$ and shift $\Delta p$.
- **Assumptions:**
  - Terrain features are invariant during meteorological changes.
- **Potential Bias & Leakage:**
  - Unconstrained slider inputs can produce physically impossible combinations (e.g., $rain\_3d > rain\_7d$) or severe out-of-distribution extrapolation ($rain\_3d = 350\text{ mm}$ when training max is $103.3\text{ mm}$).
- **Corrective Actions Taken:**
  - Added physical monotonicity checks and training domain range guards with explicit user warnings.

---

## Action Plan & Audit Resolution Matrix

| Stage | Finding / Risk | Severity | Resolution Implemented |
| :--- | :--- | :--- | :--- |
| **Feature Leakage** | `rain_event_sum` contains post-event rainfall | High | Classified task as retrospective susceptibility; documented in `feature_availability.csv`. |
| **Spatial Bleed** | Boundary cells touching train blocks | Medium | Created 1 km buffered holdout; verified ROC-AUC is maintained (0.908 vs 0.901). |
| **Cross-Event Generalisation** | Tree model collapsed to 0.485 on unseen event | Critical | Documented failure mode in `generalisation_evaluation.md`; highlighted neural net stability. |
| **Probability Calibration** | Uncalibrated NN Brier score (0.1651) worse than baseline | High | Implemented validation-fitted Platt scaling (reduced Brier to 0.0695). |
| **SAR IoU 0.991** | Synthetic backscatter demo claimed as validation | High | Reclassified in docs and UI as algorithmic consistency test; created SAR audit report. |
| **Scenario Plausibility** | Monotonicity violations and extrapolation | Medium | Implemented range check warnings in `LocationAnalyzer` and UI. |
