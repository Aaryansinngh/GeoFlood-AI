# GeoFlood AI — Phase 3: Comprehensive Data and Feature Audit

**Document Status:** Research-Grade Audit & Provenance Verification  
**Date:** October 2026  
**Study Region:** Brahmaputra River Floodplain, Assam, India (26.30°N–27.20°N, 92.80°E–94.20°E)  
**Evaluated Dataset:** `data/processed/features.parquet` (498,400 grid cells across 2 historical flood events)

---

## 1. Executive Summary & Purpose

This audit establishes the empirical provenance, feature distributions, temporal alignment, and data leakage safeguards for the GeoFlood AI platform. Machine learning models deployed in Earth Observation and flood susceptibility mapping frequently suffer from inflated performance metrics due to unaddressed spatial autocorrelation, circular feature definitions, and post-event feature leakage. 

This document traces all 10 model features and the binary ground-truth label, categorises each feature by its physical sensor origin and prediction-time availability, and documents the architectural safeguards implemented to ensure scientific defensibility.

---

## 2. Dataset Architecture & Geospatial Specifications

The GeoFlood AI spatial domain is structured as an orthogonal raster grid over the central Brahmaputra basin, spanning Kaziranga National Park, Jorhat, Tezpur, and surrounding agricultural floodplains.

* **Grid Dimensions:** 400 rows $\times$ 623 columns ($N = 249,200$ cells per event)
* **Spatial Resolution:** $\approx 250$ metres per cell ($\Delta\text{lat} \approx 0.002251^\circ$, $\Delta\text{lon} \approx 0.002246^\circ$)
* **Coordinate Reference System (Source):** EPSG:4326 (WGS 84 geographic latitude/longitude)
* **Coordinate Reference System (Display):** EPSG:3857 (Web Mercator projected coordinates)
* **Geographic Extent (Edge Bounds):**
  * West: $92.800002^\circ\text{E}$
  * South: $26.301684^\circ\text{N}$
  * East: $94.199129^\circ\text{E}$
  * North: $27.200000^\circ\text{N}$
* **Events Catalogued:**
  * **Event 1 (Primary / Training & Validation):** DFO_4259 (Dartmouth Flood Observatory, June–July 2015 monsoonal flood). 249,200 cells.
  * **Event 2 (Cross-Event Holdout):** DFO_4288 (Dartmouth Flood Observatory, August–September 2015 subsequent monsoonal surge). 249,200 cells.
  * **Total Samples:** 498,400 rows.

---

## 3. Data Provenance & Feature Catalog

| Feature Name | Primary Data Source | Physical Sensor / Platform | Spatial Res. | Temporal Scope | Units | Physical Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `elevation` | NASA SRTM v3 | C-band Radar Interferometry (Shuttle) | 30m $\rightarrow$ 250m | Static (Feb 2000) | Metres (a.s.l.) | Absolute terrain elevation above EGM96 geoid |
| `slope` | Derived from SRTM | Geospatial gradient operator | 250m | Static | Degrees ($^\circ$) | Maximum rate of elevation change across cell neighborhood |
| `aspect` | Derived from SRTM | Geospatial directional gradient | 250m | Static | Degrees ($^\circ$) | Downslope compass direction of steepest descent ($0^\circ–360^\circ$) |
| `curvature` | Derived from SRTM | 2nd derivative of elevation surface | 250m | Static | $\text{m}^{-1}$ | Profile/planform convexity ($>0$) or concavity ($<0$) |
| `twi` | Computed: $\ln(a / \tan \beta)$ | Hydro-conditioned SRTM flow routing | 250m | Static | Dimensionless | Topographic Wetness Index; tendency of water to accumulate |
| `dist_river` | Derived from JRC GSW | Euclidean distance transform | 250m | Static | Metres (m) | Planar proximity to permanent Brahmaputra river channel |
| `flow_accum` | D8 Flow Routing on DEM | HydroSHEDS / SRTM hydro-conditioned | 250m | Static | Upstream cells | Cumulative number of upslope cells draining into cell |
| `relative_elevation` | HAND Proxy | DEM minus local minimum in 25-cell window | 250m | Static | Metres (m) | Height Above Nearest Drainage; vertical distance to drainage base |
| `rain_7d_pre` | NASA GPM IMERG Final Run | Multi-satellite precipitation radar/radiometer | 0.1$^\circ \rightarrow$ 250m | 7-day Antecedent | mm | Cumulative rainfall in the 7 days prior to flood crest |
| `rain_event_sum` | NASA GPM IMERG Final Run | Multi-satellite precipitation radar/radiometer | 0.1$^\circ \rightarrow$ 250m | Full Event Window | mm | Cumulative storm total rainfall during the flood event |
| `flood_label` (Target) | Dartmouth Flood Observatory / MODIS | Optical surface reflectance (Terra/Aqua) | 250m | Event Observation | Binary (0 / 1) | 1 = Inundated during event; 0 = Dry; -1 = Permanent water |
| `is_perm_water` (Mask) | JRC Global Surface Water | Landsat 30-year multi-temporal consensus | 30m $\rightarrow$ 250m | Multi-decadal | Binary (0 / 1) | 1 = Perennial river channel; 0 = Land / seasonal flood plain |

---

## 4. Empirical Feature Distributions & Missingness Audit

Summary statistics computed across the primary dataset (DFO_4259, $N = 249,200$ cells, unmasked valid cells $N = 172,199$ excluding perennial water and nodata):

| Feature Name | Min | 25th %ile | Median | 75th %ile | Max | Mean | Std Dev | Missing / NaN Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `elevation` | 42.00 | 73.00 | 88.00 | 108.00 | 1,364.00 | 104.28 | 78.54 | 0 (0.00%) |
| `slope` | 0.00 | 0.42 | 1.15 | 2.87 | 48.61 | 2.45 | 3.82 | 0 (0.00%) |
| `aspect` | 0.00 | 84.21 | 178.45 | 267.12 | 359.99 | 176.82 | 103.11 | 0 (0.00%) |
| `curvature` | -0.18 | -0.01 | 0.00 | 0.01 | 0.22 | 0.00 | 0.02 | 0 (0.00%) |
| `twi` | 4.12 | 8.65 | 11.24 | 14.88 | 24.31 | 11.92 | 3.84 | 0 (0.00%) |
| `dist_river` | 0.00 | 1,250.00 | 3,750.00 | 8,500.00 | 36,250.00 | 5,842.10 | 5,914.30 | 0 (0.00%) |
| `flow_accum` | 1.00 | 4.00 | 18.00 | 142.00 | 248,110.00 | 1,842.30 | 12,410.20 | 0 (0.00%) |
| `relative_elevation` | 0.00 | 1.20 | 3.50 | 8.40 | 842.10 | 12.18 | 34.62 | 0 (0.00%) |
| `rain_7d_pre` | 45.20 | 88.40 | 114.20 | 142.80 | 210.50 | 116.35 | 34.12 | 0 (0.00%) |
| `rain_event_sum` | 142.50 | 280.10 | 342.60 | 412.00 | 610.80 | 348.90 | 88.45 | 0 (0.00%) |
| `flood_label` (Valid) | 0 | 0 | 0 | 0 | 1 | 0.108 | 0.310 | 0 (0.00%) |

**Data Integrity Verification:**
1. Zero NaN, null, or infinite values exist in the processed dataset.
2. The prevalence of inundation in the unmasked study region is $\approx 10.8\%$, presenting a realistic class imbalance typical of large monsoonal flood basins.
3. Permanent river cells ($N = 22,418$ cells, representing the perennial Brahmaputra braided channel) are coded as `label = -1` and flagged by `is_perm_water = 1`. These are completely excluded from supervised model training and evaluation loss functions to prevent trivial classification artifacts.

---

## 5. Investigation of Eight Potential Data Leakage Vectors

### Risk 1: Spatial Autocorrelation Leakage
* **Threat:** In continuous geospatial fields (DEM, rainfall, river distance), adjacent $250\text{ m}$ cells share near-identical feature values and flood status due to Tobler's First Law of Geography. Standard random train/test splitting samples immediately adjacent grid cells into both partitions, allowing models (especially Random Forests) to memorize local spatial coordinates rather than learning generalisable hydrology.
* **Audit Finding:** Naive random stratified splitting inflates model metrics artificially.
* **Safeguard:** Implementation of **Spatial Block Holdout** (dividing the basin into disjoint rectangular geographic blocks) and **Spatially Buffered Holdout** (enforcing a $1.0\text{ km}$ boundary buffer around test blocks).

### Risk 2: Temporal & Post-Event Target Leakage
* **Threat:** The feature `rain_event_sum` measures the cumulative precipitation over the full multi-week storm period. In an operational forecasting setting, this variable is not observable prior to the event peak.
* **Audit Finding:** Utilizing `rain_event_sum` without qualification creates the false impression that GeoFlood AI is an operational flash-flood early warning system.
* **Safeguard:** Formal reclassification of the system as a **Retrospective Flood Susceptibility / Inundation Mapping Platform**. For retrospective damage mapping and insurance assessment, `rain_event_sum` is fully available and physically valid.

### Risk 3: Preprocessing & Scaling Leakage
* **Threat:** Standardizing features using statistics ($\mu, \sigma$) computed across the pooled dataset (including validation and test partitions).
* **Audit Finding:** In early prototypes, `StandardScaler` was fitted globally.
* **Safeguard:** In Phase 3, `scaler.joblib` was verified to be fitted strictly on the designated spatial training blocks ($0, 1, 3, 4, 6, 8, 9$). Validation and test partitions are transformed using training statistics without updating the scaler parameters.

### Risk 4: Decision-Threshold Optimization Leakage
* **Threat:** Tuning the classification decision threshold to maximize F1-score or balanced accuracy on the final test set.
* **Audit Finding:** The evaluation pipeline optimizes thresholds exclusively on the disjoint validation partition (`val_blocks = [5, 11]`), then freezes the thresholds before evaluating on the held-out test blocks (`test_blocks = [2, 7, 10]`) and cross-event data (`DFO_4288`).
* **Safeguard:** Zero test-set exposure during threshold selection.

### Risk 5: Probability Calibration Leakage
* **Threat:** Fitting Platt scaling (logistic sigmoid) or Isotonic Regression on the held-out test set, artificially driving Brier Score down.
* **Audit Finding:** Calibration curves and models are fitted strictly on `val_df`. Calibrated probabilities on the test set are generated using the pre-fit calibrator object.
* **Safeguard:** Verified in `src/evaluation/run_validation.py` (lines 330–348).

### Risk 6: Permanent Water Circularity
* **Threat:** Allowing the model to use `is_perm_water` as an input feature to predict flooding. Since permanent water is always wet, this results in $100\%$ accuracy for river cells while concealing poor performance on floodplain dry land.
* **Audit Finding:** `is_perm_water` is completely excluded from the feature matrix `FEATURE_COLUMNS`. It is used exclusively as a post-prediction evaluation mask.
* **Safeguard:** Verified in `src/models/predict.py`.

### Risk 7: Geographic Boundary Bleed
* **Threat:** Pixels immediately adjacent to spatial block partition borders sharing hydrological continuity.
* **Audit Finding:** Cells within $1.0\text{ km}$ of any training block boundary were identified in Protocol 2.
* **Safeguard:** A $1.0\text{ km}$ buffer was applied using Euclidean coordinate distance, excluding $3,934$ boundary cells and forming the 35,815-cell `buffered_test_df`.

### Risk 8: Feature Redundancy & Multicollinearity
* **Threat:** Collinear features such as `elevation` and `relative_elevation`, or `rain_7d_pre` and `rain_event_sum`, destabilizing Logistic Regression coefficients.
* **Audit Finding:** While tree ensembles and neural networks handle nonlinear interactions robustly, Logistic Regression shows sign instability under collinearity. Permutation importance tests on held-out blocks demonstrate that `relative_elevation` and `dist_river` dominate terrain susceptibility.

---

## 6. Prediction-Time Feature Availability Classification

| Task Context | Feature Set Valid | Operational Feasibility | Intended Application |
| :--- | :--- | :--- | :--- |
| **Static Susceptibility Mapping** | Topographic features only (`elevation`, `slope`, `aspect`, `curvature`, `twi`, `dist_river`, `flow_accum`, `relative_elevation`) | Fully Available 100% offline | Long-term urban planning, zoning, infrastructure risk rating |
| **Retrospective Inundation Mapping** | Topographic features + `rain_7d_pre` + `rain_event_sum` | Fully Available post-event | Post-disaster damage assessment, insurance claims, emergency response review |
| **Near-Real-Time Forecasting** | Topographic features + `rain_7d_pre` + Numerical Weather Forecasts | Partially Available (requires GPM early-run or ECMWF forecast) | 24h–72h early warning (requires separate forecast model calibration) |

---

## 7. Audit Conclusion

The GeoFlood AI processed feature store (`data/processed/features.parquet`) is structurally sound, free of missing values, and adheres to strict geospatial integrity constraints. The feature definitions, spatial blocking protocols, and post-event availability boundaries are fully documented and reproducible via `reports/phase3_reproducibility.json`.
