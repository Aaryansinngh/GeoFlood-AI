# GeoFlood AI â€” Event-Based Flood Susceptibility Mapping

## 🚀 Live Demo

**Try GeoFlood-AI here:**

[🌊 Launch GeoFlood-AI](https://geoflood-ai.streamlit.app/)

Explore the interactive flood intelligence, geospatial analysis, and evacuation decision-support dashboard.

---
> **What this system does:** For each grid cell in a defined study region, it
> estimates the **probability that the cell was inundated during a documented
> historical flood event**, given terrain and antecedent/concurrent rainfall
> features derived from satellite data.
>
> **What this system does NOT do:** Real-time flood forecasting, early-warning,
> operational risk assessment, or advance prediction of future floods.
> These are listed under [Future Scope](#9-future-scope).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Study Region & Events](#2-study-region--events)
3. [Data Sources](#3-data-sources)
4. [Feature Engineering](#4-feature-engineering)
5. [Models](#5-models)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Known Limitations](#7-known-limitations)
8. [Reproduction Commands](#8-reproduction-commands)
9. [Future Scope](#9-future-scope)

---

## 1. Overview

GeoFlood AI is a geospatial machine-learning research prototype built as an
academic PBL (Project-Based Learning) demonstration. It implements one complete,
honest pipeline:

```
satellite-derived data â†’ feature table â†’ trained models â†’ evaluation â†’ interactive map
```

The application provides:

| Feature | Description |
|---|---|
| Interactive flood susceptibility map | Folium raster overlay on OpenStreetMap basemap for 3 models |
| Coordinate-based location analysis | Single-cell inspection, feature retrieval, inference |
| Multi-model comparison | LR vs RF vs NN side-by-side at a location |
| Rainfall scenario simulation | Vary antecedent rainfall, observe probability shift |
| SAR flood detection | Otsu-threshold flood mask on synthetic radar demo |
| Mamdani fuzzy layer | Rule-based interpretability overlay |
| **Scientific Validation tab** | Empirical metrics, ROC/PR/reliability figures, cross-event results |

---

## 2. Study Region & Events

| Property | Value |
|---|---|
| Region | Brahmaputra River basin, Assam, India |
| Bounding box | Approx. 25.5Â°â€“27.5Â°N, 91.5Â°â€“95.5Â°E |
| Grid | 400 rows Ã— 623 columns = **249,200 cells** |
| Cell size | ~500 m Ã— ~500 m |
| CRS | EPSG:4326 (WGS84 geographic) |
| Training event | DFO_4259 â€” June 2012 Assam flood |
| Cross-event test | DFO_4288 â€” August 2015 Assam flood |

---

## 3. Data Sources

| Layer | Source | Resolution | Notes |
|---|---|---|---|
| Flood inundation labels | Global Flood Database (GFD) â€” MODIS-derived | ~500 m | Cloud-masked binary raster per event |
| Elevation | SRTM GL1 (NASA) | 30 m â†’ resampled to 500 m | |
| Slope, TPI, Relative elevation | Derived from SRTM DEM | 500 m | Computed via rasterio / numpy |
| 3-day antecedent rainfall | ERA5 / GPM IMERG | ~10 km â†’ interpolated | Pre-event accumulation |
| 7-day antecedent rainfall | ERA5 / GPM IMERG | ~10 km â†’ interpolated | Pre-event accumulation |
| Event-sum rainfall | ERA5 / GPM IMERG | ~10 km â†’ interpolated | Full 28-day event window (post-event â€” see Limitations) |

---

## 4. Feature Engineering

Nine features per grid cell:

| Feature | Type | Pre/Post event |
|---|---|---|
| `elevation` | Terrain | Static |
| `slope` | Terrain | Static |
| `tpi` | Terrain (topographic position index) | Static |
| `relative_elevation` | Terrain | Static |
| `distance_to_river` | Terrain | Static |
| `flow_accumulation` | Terrain | Static |
| `rain_3d` | Hydro-meteorological | **Pre-event** |
| `rain_7d` | Hydro-meteorological | **Pre-event** |
| `rain_event_sum` | Hydro-meteorological | âš ï¸ **Post-event** â€” see Limitations |

---

## 5. Models

| Model | Implementation | Hyperparameter selection |
|---|---|---|
| Logistic Regression | scikit-learn `LogisticRegression` | Default C=1.0 |
| Random Forest | scikit-learn `RandomForestClassifier` | n_estimators=200, max_depth=12 |
| Neural Network | PyTorch MLP (3 hidden layers, BatchNorm, Dropout) | Trained 50 epochs, Adam optimizer |

Decision thresholds and probability calibrators (Platt / Isotonic) are fitted
on **validation blocks 5 & 11** only â€” the test set is never seen during threshold
or calibrator selection.

---

## 6. Evaluation Methodology

Three nested evaluation protocols are used, each progressively more stringent:

### 6.1 Spatial Block Holdout

Grid divided into 12 spatial blocks (3Ã—4). Blocks 2, 7, 10 form the test set
(~39,749 cells). Threshold selected on validation blocks 5 & 11.

| Model | ROC-AUC | PR-AUC | F1 | Brier Skill Score |
|---|---|---|---|---|
| Logistic Regression | 0.684 | 0.320 | 0.398 | âˆ’0.549 |
| Random Forest | **0.901** | **0.507** | 0.481 | +0.141 |
| Neural Network (Uncalibrated) | 0.805 | 0.448 | 0.323 | âˆ’0.947 |
| Neural Network (Platt Calibrated) | 0.805 | 0.448 | 0.415 | **+0.180** |

### 6.2 Spatially Buffered Holdout (1 km)

1 km buffer applied around block boundaries. 3,934 boundary cells excluded.
Test set: ~35,815 cells. This verifies block holdout metrics are NOT inflated
by spatial autocorrelation boundary bleed.

| Model | ROC-AUC | Change vs Block Holdout |
|---|---|---|
| Random Forest | 0.908 | +0.007 (metrics *improve* â†’ no bleed) |
| Neural Network | 0.816 | +0.011 (metrics *improve* â†’ no bleed) |

### 6.3 Cross-Event Holdout (DFO_4288)

Models trained on June 2012 flood evaluated on August 2015 flood (~172,199 cells).
**True temporal generalisation test.**

| Model | ROC-AUC | Change vs Block Holdout |
|---|---|---|
| Logistic Regression | 0.713 | âˆ’0.029 (moderate degradation) |
| **Random Forest** | **0.485** | **âˆ’0.416 (near-random collapse)** |
| Neural Network | 0.672 | âˆ’0.133 (significant but not catastrophic) |

> âš ï¸ **Critical finding:** Random Forest collapses to near-random on temporal
> cross-event validation. It must not be used for cross-season or operational
> claims. See `reports/generalisation_evaluation.md`.

---

## 7. Known Limitations

| Limitation | Impact |
|---|---|
| `rain_event_sum` is post-event | System is valid only for retrospective susceptibility analysis, NOT operational forecasting |
| Single training basin | Spatial transfer to other river systems is untested |
| Two events only | Cross-event collapse of RF is likely partially due to small event-level sample |
| SAR 0.991 IoU is synthetic | The IoU figure measures algorithmic consistency on simulated data, NOT real Sentinel-1 accuracy |
| No independent label validation | GFD MODIS labels have their own cloud-masking and resolution limitations |
| Uncalibrated NN probabilities are poorly calibrated | Platt calibration is required before treating NN outputs as true probabilities |

---

## 8. Reproduction Commands

```bash
# Step 1 â€” Download raw satellite data
python -m src.data.download_raw

# Step 2 â€” Build feature table (features.parquet)
python -m src.data.build_features

# Step 3 â€” Train all models (saves to models/)
python -m src.models.train

# Step 4 â€” Run scientific validation pipeline
#          (generates reports/phase3_metrics.csv, reports/figures/, etc.)
python -m src.evaluation.run_validation

# Step 5 â€” Launch interactive application
python -m streamlit run app.py

# Step 6 â€” Run full test suite (36 tests)
python -m pytest tests/ -v
```

### Environment Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.12. Key dependencies: `streamlit`, `streamlit-folium`, `folium`,
`torch`, `scikit-learn`, `scikit-fuzzy`, `rasterio`, `geopandas`, `plotly`.

---

## 9. Future Scope

The following capabilities are explicitly **out of scope** for this prototype and
are listed here for academic completeness:

- **Real-time flood forecasting** â€” requires NWP model coupling and sub-hourly data
- **Operational early warning** â€” requires lead-time validation, false-alarm analysis, and institutional approval
- **Multi-basin generalisation** â€” requires representative training data across climatic zones
- **Real SAR integration** â€” requires Copernicus/ESA API access and atmospheric correction
- **Ensemble uncertainty quantification** â€” requires calibrated multi-model posterior sampling
- **Dynamic risk index** â€” requires temporal updating and event-specific threshold tuning

---

## Project Structure

```
GeoFlood-AI/
â”œâ”€â”€ app.py                          # Main Streamlit application (7 tabs)
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ config/settings.yaml
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                        # Raw GFD, SRTM, ERA5 downloads
â”‚   â”œâ”€â”€ processed/features.parquet  # 498,400 rows (2 events Ã— 249,200 cells)
â”‚   â””â”€â”€ sample/                     # Sample data (labelled SAMPLE DATA)
â”œâ”€â”€ models/                         # Saved model artifacts
â”‚   â”œâ”€â”€ nn_model.pt
â”‚   â”œâ”€â”€ random_forest.pkl
â”‚   â”œâ”€â”€ logistic_regression.pkl
â”‚   â”œâ”€â”€ scaler.pkl
â”‚   â””â”€â”€ metrics.json
â”œâ”€â”€ outputs/predictions.parquet     # 249,200 cells with per-model probabilities
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ phase3_metrics.csv          # 12-row empirical evaluation table
â”‚   â”œâ”€â”€ evaluation_summary.json     # Full JSON with feature importance & Brier
â”‚   â”œâ”€â”€ feature_availability.csv    # Feature leakage audit
â”‚   â”œâ”€â”€ phase3_scientific_audit.md  # 13-stage pipeline audit
â”‚   â”œâ”€â”€ generalisation_evaluation.md
â”‚   â”œâ”€â”€ sar_validation_audit.md
â”‚   â””â”€â”€ figures/                    # ROC, PR, reliability, confusion, permutation importance PNGs
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/                       # Data download & feature engineering
â”‚   â”œâ”€â”€ models/                     # Training, inference, fuzzy layer
â”‚   â”œâ”€â”€ geo/                        # SAR detection, location analysis
â”‚   â””â”€â”€ evaluation/                 # run_validation.py â€” Phase 3 pipeline
â””â”€â”€ tests/                          # 36 tests across 4 files (all passing)
```


## Project Presentation

[View GeoFlood-AI Final Pitch Deck](docs/GeoFlood_AI_Pitch_Deck.pptx)
---

*GeoFlood AI â€” Academic PBL Prototype. Not for operational use.*
*All displayed metrics come from real pipeline runs. No numbers are fabricated.*

