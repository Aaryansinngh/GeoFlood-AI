# GeoFlood AI: Flood Susceptibility Mapping from Satellite-Derived Data with a Neural Network and an Interactive Map

## Academic PBL Project Technical Report

---

### Abstract
Flood susceptibility mapping plays an essential role in disaster risk reduction, land-use zoning, and emergency preparedness. Traditional hydrodynamic models, while physically rigorous, require extensive river bathymetry, calibrated cross-sections, and substantial computational power, often making regional deployment challenging. Machine learning provides an effective complementary paradigm. This project presents **GeoFlood AI (Prototype)**, an end-to-end geospatial machine learning pipeline that maps event-conditioned flood susceptibility across the Brahmaputra River floodplain in Assam, India (Kaziranga–Jorhat corridor). Using authentic satellite-derived observations from the Shuttle Radar Topography Mission (SRTM 30m DEM) and high-resolution precipitation reanalysis, models are trained to predict historical inundation extent documented in the Dartmouth Flood Observatory (DFO) and MODIS Global Flood Database (GFD). To prevent spatial autocorrelation leakage, models are benchmarked using a rigorous **Spatial Block Holdout split** and an **Event Holdout split**. The Random Forest baseline achieved an ROC-AUC of **0.901** and Recall of **81.2%** on 39,749 held-out test cells, while a PyTorch Feed-Forward Neural Network achieved an ROC-AUC of **0.805** and Recall of **60.9%** (generalizing to **80.4% Recall** across a separate monsoon event). An offline-first 3-page Streamlit application integrates the model outputs with interactive geospatial maps, validation analytics, and a Mamdani fuzzy decision layer.

---

### 1. Problem Statement & Scope Boundary

Flooding along the Brahmaputra River in Assam causes annual devastation to agriculture, critical infrastructure, and wildlife sanctuaries such as Kaziranga National Park. 

#### Target Definition
> For each ~250m grid cell in the study region, estimate the probability that the cell was inundated during a documented historical flood event, conditioned on antecedent rainfall and terrain characteristics.

#### Boundary Taxonomy
This project is strictly scoped as **flood susceptibility / event-based risk mapping**. It is **not** an early warning system or advance real-time forecast. Inundation estimates are conditioned on observed meteorological footprints and physical topographic properties.

---

### 2. Datasets & Feature Pipeline

The study area encompasses a 400 × 623 grid (249,200 total cells, ~172,199 valid land cells) covering `[92.80°E, 26.30°N, 94.20°E, 27.20°N]`:

1. **Ground Truth Inundation Labels:**
   - **Source:** Global Flood Database (Tellman et al., *Nature* 2021) / Dartmouth Flood Observatory (DFO).
   - **Events Analyzed:**
     - Event 1: `DFO_4259` (June 2–29, 2015; 19,522 inundated cells, 11.34% prevalence).
     - Event 2: `DFO_4288` (August 13 – September 11, 2015; 21,322 inundated cells, 12.38% prevalence).
   - **Permanent Surface Water:** Perennial channels of the Brahmaputra were masked out using the European Commission JRC Global Surface Water dataset (Band 5 of GFD) to ensure models learn dry-land inundation rather than permanent water optics.
   - **Missing Data:** Cloud-obscured or unobserved pixels are retained as `NaN` (Unavailable) and never converted to 0 ("safe").

2. **Digital Elevation Model (DEM) & Terrain Derivatives:**
   - **Source:** NASA Shuttle Radar Topography Mission (SRTM GL1, 30m resolution, ESA STEP public mirror).
   - **Features:** Ground elevation (m), slope (° via 2nd-order finite differences), Topographic Position Index (TPI, 15-cell moving window), and Relative Elevation / HAND proxy (25-cell neighborhood).

3. **Precipitation Metrics:**
   - **Source:** Daily satellite and reanalysis precipitation interpolated across the master grid using Thin Plate Splines.
   - **Features:** Event-window cumulative precipitation (`rain_event_sum`), pre-event 3-day rainfall (`rain_3d`), and pre-event 7-day rainfall (`rain_7d`).

---

### 3. Methodology & Leakage-Aware Validation

#### 3.1 Spatial Block Holdout Split
Random k-fold splits on geospatial raster grids cause severe test set leakage because neighboring cells share near-identical topography and rainfall (spatial autocorrelation). To enforce strict physical independence:
- The study region was tiled into **12 contiguous spatial blocks** (3 rows × 4 columns).
- Training Blocks: `[0, 1, 4, 5, 8, 9, 11]` (91,442 cells).
- Validation Blocks: `[3, 6]` (41,008 cells).
- Test Blocks: `[2, 7, 10]` (39,749 cells).
- Train, validation, and test blocks share **zero spatial intersection**.

#### 3.2 Decision Threshold Optimization
Decision thresholds were optimized **exclusively on the validation set** to maximize F1-score under class imbalance, preventing test set snooping:
- Logistic Regression: $\tau = 0.5888$
- Random Forest: $\tau = 0.0560$
- Neural Network: $\tau = 0.4394$

#### 3.3 Model Architectures
- **Logistic Regression:** Scaled features, $L_2$ regularization, balanced class weights.
- **Random Forest:** 100 estimators, max depth 12, balanced class weights.
- **PyTorch Neural Network:** 3 hidden layers (64 $\rightarrow$ 32 $\rightarrow$ 16 units), Batch Normalization, ReLU activations, Dropout ($p=0.2$), BCEWithLogitsLoss with `pos_weight = 12.1`, Adam optimizer, early stopping on validation PR-AUC.

---

### 4. Experimental Results

#### 4.1 Spatial Block Holdout Benchmark (Held-out Test Set, N = 39,749)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | False Negative Rate | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.684 | 0.320 | 53.5% | 31.7% | 0.398 | 46.5% | 0.131 |
| **Random Forest** | **0.901** | **0.507** | **81.2%** | **34.1%** | **0.481** | **18.8%** | **0.073** |
| **PyTorch Neural Network** | 0.805 | 0.448 | 60.9% | 22.0% | 0.323 | 39.1% | 0.165 |

#### 4.2 Why Recall Dominates in Disaster Analytics
In hazard susceptibility modeling, the asymmetry of error costs is extreme:
- A **False Positive** produces an unnecessary precautionary advisory.
- A **False Negative** leaves an unwarned population in harm's way.
The Random Forest achieved the lowest False Negative Rate (**18.8%**), accurately flagging over 3,000 of the 3,717 test-block flooded cells.

#### 4.3 Feature Importance Breakdown (Random Forest Gini)
1. **Elevation:** 34.2%
2. **Event Rainfall Sum:** 25.4%
3. **Pre-Event 3-Day Rain:** 17.5%
4. **Pre-Event 7-Day Rain:** 10.3%
5. **Relative Elevation (HAND):** 8.9%
6. **Topographic Position Index (TPI):** 2.3%
7. **Slope:** 1.4%

#### 4.4 Error Analysis across Physical Strata
- **Terraces & Flat Plains (<0.5° slope, 65–85m elevation):** Error rates averaged 29.7%–42.1%, driven by shallow agricultural depressions and backwater ponding not captured at 250m resolution.
- **Moderate & Steep Slopes (>2.0° slope, >120m elevation):** Error rates dropped to 5.3%–9.8% with zero false negatives in steep terrain, proving elevation provides an unequivocal physical barrier to flood pooling.

#### 4.5 Temporal Event Holdout Transfer
When the Neural Network trained on June 2015 (`DFO_4259`) was evaluated on August 2015 (`DFO_4288`, 172,199 cells), it achieved **80.4% Recall** and **ROC-AUC of 0.672**, demonstrating solid generalization across independent flood pulses.

---

### 5. Stretch Goals & Extensions

1. **Sentinel-1 SAR Satellite Flood Detection:**
   - Implemented VV-backscatter difference analysis with automatic Otsu thresholding (-4.78 dB) and permanent water masking.
   - Evaluated against MODIS optical reference: **IoU = 0.991, Precision = 1.000, Recall = 0.991**.
2. **Mamdani Fuzzy Inference Layer:**
   - 8-rule Mamdani system mapping model probability + antecedent 3-day rainfall into linguistic risk tiers (`Low`, `Moderate`, `High`) for operational clarity.

---

### 6. Limitations

1. **Spatial Granularity:** A 250m grid cannot resolve localized drainage structures, culverts, or road embankments.
2. **Monsoon Cloud Gaps:** Optical MODIS imagery cannot observe inundation beneath heavy overcast skies; peak flood peaks may be missed if short-lived.
3. **Absence of 2D Hydrodynamic Conservation:** The model does not enforce mass conservation equations ($\partial h/\partial t + \nabla \cdot q = 0$) or river discharge momentum.

---

### 7. Future Scope

1. Multi-basin cross-regional transfer across the Indo-Gangetic and Brahmaputra plains.
2. Integration with Numerical Weather Prediction (NWP) precipitation forecasts for lead-time early warning.
3. Spatio-temporal deep learning (U-Net CNNs on high-resolution Sentinel-1 SAR time series).

---

### 8. References

1. Tellman, B. et al. (2021). *Satellite imaging reveals increased proportion of population at risk of floods.* **Nature**, 596, 355–359.
2. Farr, T. G. et al. (2007). *The Shuttle Radar Topography Mission.* **Reviews of Geophysics**, 45(2).
3. Pekel, J. F. et al. (2016). *High-resolution mapping of global surface water and its long-term changes.* **Nature**, 540, 418–422.
4. Huffman, G. J. et al. (2020). *Integrated Multi-satellitE Retrievals for GPM (IMERG) Technical Documentation.* NASA GES DISC.
