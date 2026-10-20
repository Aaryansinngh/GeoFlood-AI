# Phase 0: Feasibility Study & Dataset Selection

## 1. Executive Summary

This feasibility report establishes the study region, data sources, prediction target, and methodological parameters for **GeoFlood AI (Prototype)**, an academic PBL project for flood susceptibility mapping. 

All datasets and endpoints have been empirically verified. The selected region is the **Brahmaputra River Floodplain in Assam, India (Kaziranga–Golaghat–Jorhat region)**, utilizing two documented historical flood events from 2015 for leakage-aware spatial block and event holdout validation.

---

## 2. Target Definition & Problem Framing

- **Fixed Prediction Target:** For each ~250m grid cell in the study region, estimate the probability that the cell was inundated during a documented historical flood event, conditioned on antecedent rainfall and terrain characteristics.
- **Classification:** Binary classification ($y_i \in \{0, 1\}$).
  - $y_i = 1$: Cell inundated during the event window.
  - $y_i = 0$: Cell remained dry during the event window.
- **Category:** Flood Susceptibility / Event-based Risk Mapping (static/event-conditioned susceptibility).
- **Explicit Distinction:** This prototype is **NOT** a real-time early warning or advance forecasting system. It maps terrain- and rainfall-driven susceptibility given an event's meteorological footprint.

---

## 3. Candidate Regions Evaluated

| Region | Candidate Details | Documented Events | Topography & Feasibility Assessment | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **Candidate 1: Assam (Brahmaputra Valley)** | Bounding Box: `[92.8°E, 26.3°N, 94.2°E, 27.2°N]` (Kaziranga–Jorhat corridor) | **DFO 4259** (June 2–29, 2015)<br>**DFO 4288** (Aug 13–Sep 11, 2015) | Floodplain elevation 50–90m bordered by Karbi Anglong / Mikir Hills (>500m). Rich topographic gradients, strong monsoonal pulse. Total cells: 249,823 (~173k valid land cells, ~12% flood ratio). | **SELECTED (Primary)** |
| **Candidate 2: North Bihar (Kosi Megafan)** | Bounding Box: `[85.5°E, 25.8°N, 87.0°E, 26.8°N]` (Darbhanga–Saharsa) | **DFO 4174** (Aug 14–Sep 8, 2014) | Extremely flat alluvial fan (<0.2° slope everywhere). High flood recurrence, but minimal terrain variance causes elevation models to collapse into geographic coordinate proxies. | Deprioritized |
| **Candidate 3: Eastern UP (Ghaghara/Rapti)** | Bounding Box: `[83.0°E, 26.0°N, 84.0°E, 27.0°N]` (Gorakhpur) | **DFO 1789** (Aug 20–Sep 11, 2001) | Fluvial floodplain. Older 2001 event lacks high-resolution multi-satellite optical and radar coverage for stretch comparisons. | Deprioritized |

---

## 4. Verified Data Sources & Access Mechanisms

### 4.1 Flood Extent Labels (Ground Truth)
- **Primary Source:** Global Flood Database (GFD) / Cloud to Street & Dartmouth Flood Observatory (DFO)
  - Nature Reference: *Tellman et al., "Satellite imaging reveals increased proportion of population at risk of floods", Nature 596, 355–359 (2021).*
  - GEE Catalog ID: `GLOBAL_FLOOD_DB/MODIS_EVENTS/V1`
  - Direct Mirror: Google Cloud Storage public archive (`gs://gfd_v1_4/` via `https://storage.googleapis.com/gfd_v1_4/`)
  - Resolution: 250m MODIS Terra/Aqua composite
  - Verified Bands:
    1. `flooded`: Binary inundation flag (1 = flooded, 0 = non-flooded)
    2. `duration`: Flood duration (days)
    3. `clear_views`: Count of cloud-free observations
    4. `clear_perc`: Cloud-free percentage (quality assurance)
    5. `jrc_perm_water`: Permanent surface water mask (from European Commission JRC Global Surface Water dataset)
  - Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

### 4.2 Terrain Data (Digital Elevation Model & Slope)
- **Primary Source:** NASA Shuttle Radar Topography Mission (SRTM) 1 Arc-Second Global (SRTMGL1, ~30m resolution).
  - GEE Catalog ID: `USGS/SRTMGL1_003`
  - Direct Mirror: ESA STEP auxdata mirror (`https://step.esa.int/auxdata/dem/SRTMGL1/` tiles `N26E092`, `N26E093`, `N27E092`, `N27E093`)
  - Derived Features:
    - Elevation (meters above sea level)
    - Slope (degrees, calculated using finite difference gradient)
    - Relative Elevation / Height Above Minimum Drainage (HAND proxy)
  - Licence: Public Domain (USGS / NASA).

### 4.3 Precipitation Data (Rainfall Features)
- **Primary Source:** Daily precipitation from satellite-derived observation and reanalysis:
  - CHIRPS Daily 0.05° (~5 km) satellite infrared & station blended precipitation (Climate Hazards Center, UC Santa Barbara).
  - Open-Meteo High-Resolution Historical Reanalysis API (0.1° / ~9 km, ERA5-Land & satellite calibrated, zero-key REST endpoint).
  - GEE Reference: `NASA/GPM_L3/IMERG_V06`
  - Derived Features:
    - `rain_event_sum`: Total cumulative rainfall during the flood event window.
    - `rain_7d`: Cumulative rainfall in the 7 days prior to peak flooding.
    - `rain_3d`: Cumulative rainfall in the 3 days prior to peak flooding.
  - Licence: Open data / Creative Commons.

---

## 5. Technical Grid Specifications

- **Coordinate Reference System (CRS):** WGS 84 (EPSG:4326)
- **Study Area Extent:**
  - Min Longitude: 92.80° E
  - Max Longitude: 94.20° E
  - Min Latitude: 26.30° N
  - Max Latitude: 27.20° N
- **Raster Dimensions:** 401 rows × 623 columns = 249,823 cells
- **Pixel Resolution:** ~0.002245° (~250 meters at 26.5°N)
- **Class Distribution (DFO 4259):**
  - Valid Land Cells: 173,772
  - Inundated Cells: 20,630 (11.87%)
  - Permanent Water Cells: 1,361 (masked to prevent positive label leakage)
  - Nodata / Cloud Obs: 76,051 (explicitly masked as "unavailable", never imputed to "safe" or 0)

---

## 6. Known Risks, Biases & Mitigation Strategies

| Risk / Bias | Specific Impact on Prototype | Mitigation Strategy |
| :--- | :--- | :--- |
| **Spatial Autocorrelation (Leakage)** | Random train/test split causes test cells adjacent to training cells to leak labels, inflating accuracy artificially. | Implement **Spatial Block Holdout** (divide region into spatial tiles, holding out entire spatial blocks) and **Event Holdout** (train on Event 1, test on Event 2). |
| **Permanent Water Conflation** | Rivers and perennial lakes always look inundated, biasing the model into learning water bodies rather than flooded dry land. | Exclude permanent water pixels using the JRC Global Surface Water mask (Band 5 of GFD). |
| **Cloud Contamination** | Optical MODIS data cannot penetrate monsoon clouds, causing false negatives in peak rain. | GFD utilizes 3-day multi-temporal compositing and quality filtering (`clear_views`, `clear_perc >= 0.8`). Unclear cells are kept as `NaN`/unavailable. |
| **Offline Reproducibility** | GEE requires interactive OAuth login, which fails in headless automated or offline environments. | Provide both GEE export script (`src/data/export_gee.py`) and direct satellite downloader (`src/data/download_raw.py`) fetching authentic raw GeoTIFFs from verified public mirrors, plus a precomputed offline sample in `data/sample/`. |
