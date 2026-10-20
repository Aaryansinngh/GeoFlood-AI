# GeoFlood AI — Phase 3: Known Scientific & Operational Limitations

**Document Status:** Research Transparency & Limitations Disclosure  
**Date:** October 2026  
**Project:** GeoFlood AI — Geospatial Flood Susceptibility Platform  
**Target Audience:** Scientific Reviewers, Hydrologists, Disaster Management Authorities  

---

## 1. Scope & Purpose

Scientific integrity in artificial intelligence requires candid documentation of model boundaries, failure modes, and domain assumptions. This report outlines the documented limitations of the GeoFlood AI system across geographic, temporal, hydrological, and machine learning dimensions.

---

## 2. Geographic & Geomorphological Limitations

### 1. Single River Basin Specialization
* **Domain:** GeoFlood AI is calibrated and evaluated exclusively on the central **Brahmaputra River floodplain in Assam, India** ($26.3^\circ\text{N}–27.2^\circ\text{N}, 92.8^\circ\text{E}–94.2^\circ\text{E}$).
* **Geomorphological Context:** The Brahmaputra is a high-discharge, highly braided, sediment-heavy alluvial river system with wide, low-gradient floodplains.
* **Limitation:** The learned feature relationships (e.g., the dominant influence of `relative_elevation` and `dist_river`) **do not transfer directly** to:
  * Steep mountainous catchments (where flash floods are velocity-driven rather than inundation-driven).
  * Arid/semi-arid regions (where wadi flash floods occur in dry channels).
  * Highly urbanized metropolitan zones (where artificial stormwater drainage and impervious surface fractions govern inundation).
* **Mitigation for Transfer:** The pipeline architecture is modular, but applying the models to another basin (e.g., Ganga, Godavari, Mekong) requires retraining on local DEM and historical inundation records.

### 2. Spatial Resolution Constraints ($250\text{ m}$ Grid)
* **Raster Scale:** The master prediction grid operates at $\approx 250\text{ m}$ spatial resolution ($400 \times 623$ cells).
* **Limitation:** A $250\text{ m}$ cell represents an area of $\approx 6.25\text{ hectares}$. Consequently, the model cannot resolve:
  * Flood embankments, dykes, and levees narrower than $250\text{ m}$.
  * Road culverts, bridge spans, and drainage channels.
  * Street-level urban flooding or individual building vulnerability.
* **Operational Implication:** The system is appropriate for regional risk zoning, emergency evacuation planning, and macro-scale damage assessment, but is **unsuitable for micro-infrastructure engineering**.

---

## 3. Hydrological & Temporal Data Limitations

### 1. Retrospective vs Forecasting Classification
* **Feature Nature:** The model uses `rain_event_sum` (total cumulative storm rainfall over the duration of the event).
* **Limitation:** Because `rain_event_sum` is only known in full *after* the storm has concluded, GeoFlood AI is an **event-based susceptibility and inundation mapping system**, not a real-time forecasting or early warning system.
* **Correction:** Presenting the application as an "early-warning forecaster" would be scientifically inaccurate. For early warning, the feature would need to be replaced with multi-day numerical weather prediction (NWP) forecasts from IMD or ECMWF.

### 2. Limited Historical Events Catalogued
* **Events Evaluated:** The dataset contains two historical events:
  * **DFO_4259:** Primary flood event (June–July 2015, monsoonal onset).
  * **DFO_4288:** Secondary flood event (August–September 2015, late-monsoon surge).
* **Limitation:** Two events do not capture the full multi-decadal hydrological variability of the Brahmaputra basin, such as 100-year recurrence interval catastrophic floods or pre-monsoon convective flash floods.

### 3. Dynamic River Channel Morphology & Sandbar Avulsion
* **Hydrological Dynamics:** The Brahmaputra is one of the world's most dynamic braided rivers, shifting sediment bars (*chars*) and eroding riverbanks by hundreds of metres in a single monsoon season.
* **Limitation:** The JRC permanent water mask is a multi-decadal consensus composite. In localized reaches, the active river channel may have shifted since the baseline period, leading to occasional misclassification along the immediate river margin.

---

## 4. Machine Learning Model Generalisation & Failure Modes

### 1. Random Forest Cross-Event Generalisation Collapse
One of the most critical empirical findings of Phase 3 is the catastrophic generalisation collapse of the Random Forest model across temporal events:

* **Spatial Block Holdout Performance (Within Event 1):** $\text{ROC-AUC} = \mathbf{0.901}$, $\text{PR-AUC} = \mathbf{0.507}$
* **Cross-Event Holdout Performance (Event 2 — DFO_4288):** $\text{ROC-AUC} = \mathbf{0.485}$, $\text{PR-AUC} = \mathbf{0.131}$

#### Root Cause Analysis:
1. Decision trees partition feature space using orthogonal axis-aligned splits (e.g., `if rain_event_sum > 340.5 mm`).
2. In DFO_4259, rainfall was concentrated in the northern sub-basins with high local values, allowing deep tree ensembles to split on specific rainfall thresholds as proxies for geographic location.
3. In DFO_4288, the spatial distribution and storm totals shifted. Because tree models cannot extrapolate or learn smooth functional representations outside their training bounding boxes, the Random Forest's predictions degraded to worse than random guessing ($\text{ROC-AUC} < 0.50$).

#### Model Contrast:
The **PyTorch Neural Network** demonstrated substantially superior generalisation:
* **Spatial Block Holdout:** $\text{ROC-AUC} = \mathbf{0.805}$
* **Cross-Event Holdout:** $\text{ROC-AUC} = \mathbf{0.672}$
The continuous sigmoidal activations and smooth weight space enabled the neural network to preserve physical monotonicity (lower elevation + higher rainfall = higher flood risk) across different flood events.

### 2. Scenario Simulation Extrapolation Bounds
* **Simulator Operation:** The scenario tool allows users to modify `rain_7d_pre` and `rain_event_sum` up to $+50\%$ or $+200\text{ mm}$.
* **Limitation:** 
  * Random Forests produce flat step functions when inputs exceed training maximums.
  * Neural networks continue linear extrapolation beyond their normalized feature range $[-3.0, +3.0]$, which may produce extreme probability saturation.
* **Mitigation:** The application explicitly bounds scenario modifications within physically plausible meteorological ranges and displays warning badges when inputs exceed historical maximums.

---

## 5. Satellite Ground Truth Uncertainties

### 1. Optical Cloud Contamination in Reference Masks
* The reference flood polygons from the Dartmouth Flood Observatory (DFO) are derived from MODIS Terra and Aqua optical sensors ($250\text{ m}$ surface reflectance).
* During monsoonal storms, heavy cloud cover, cloud shadows, and dense mist frequently obscure the ground during satellite passes.
* While multi-day compositing reduces missing pixels, edge boundaries of flood extents may contain temporal compositing artifacts.

### 2. SAR Module Algorithmic Nature
* As detailed in `reports/phase3_sar_provenance.md`, the SAR module's $0.991$ IoU metric is an algorithmic verification on a bimodal distribution, not empirical satellite validation against raw radar imagery.
* Operational SAR processing requires handling radar speckle filter artifacts, Bragg wind waves, and dense vegetation penetration.

---

## 6. Summary of Operational Recommendations

| Intended Use Case | Feasibility Status | Governing Condition |
| :--- | :--- | :--- |
| **Brahmaputra Regional Susceptibility Zoning** | **Approved** | Use Neural Network (Platt Calibrated); verified ROC-AUC 0.805 |
| **Post-Disaster Inundation Mapping** | **Approved** | Valid when post-event rainfall totals are integrated |
| **Near-Real-Time Early Warning System** | **Not Approved** | Requires integration of predictive numerical weather models |
| **Micro-Zoning (< 100m) / Culvert Sizing** | **Not Approved** | Requires fine-scale LiDAR DEM ($< 5\text{ m}$) and 2D hydraulic modeling (HEC-RAS / TUFLOW) |
| **Direct Application to Mountain/Arid Catchments** | **Not Approved** | Requires complete retraining on local geomorphology and hydrology |
