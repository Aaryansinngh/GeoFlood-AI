# GeoFlood-AI 2.0: Environmental & Geospatial Data Sources Catalog

**Document Status:** Official Reference & Specification Document  
**Version:** 2.0 (Phase 2 Architecture)  
**Date:** November 2026  

---

## 1. Overview and Acquisition Philosophy

GeoFlood-AI 2.0 operates under strict scientific data provenance standards:
1. **Authenticity Guarantee:** Only authentic satellite observations and official reanalyses are ingested. Synthetic demonstrations are never substituted for real sensor observations.
2. **Transparent Provenance Tracking:** Every acquired dataset generates a persistent JSON record recording source URL, exact temporal window, bounding box, CRS, spatial/temporal resolution, missing value percentage, and SHA-256 artifact verification checksum.
3. **Resilient Network Client Architecture:** Built on `BaseAcquisitionClient` with deterministic disk caching, exponential backoff retries, rate-limiting compliance, and standardized error handling.
4. **Leakage-Aware Partitioning:** Clear temporal delineation between static baseline terrain, pre-event antecedent rainfall, during-event hydrometeorology, and post-event satellite composites.

---

## 2. Integrated Data Sources Matrix

| Dataset | Provider / Source | Spatial Resolution | Temporal Resolution | Native CRS | Primary Variables Acquired | Authentication |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **NASA POWER** | NASA Langley Research Center | $0.5^\circ \times 0.625^\circ$ ($\sim 50\text{ km}$) | Daily | EPSG:4326 | `PRECTOTCORR` (precipitation), `T2M` (temperature), `RH2M` (humidity), `WS10M` (wind speed) | None (Public API) |
| **CHIRPS v2.0** | UCSB Climate Hazards Center | $0.05^\circ$ ($\sim 5.3\text{ km}$) | Daily | EPSG:4326 | Daily quasi-global precipitation rasters | None (Public FTP/HTTP) |
| **ERA5-Land** | ECMWF (via Open-Meteo Gateway / CDS) | $0.1^\circ \times 0.1^\circ$ ($\sim 9\text{ km}$) | Hourly / Daily Aggregates | EPSG:4326 | Precipitation sum, 2m mean temperature, surface solar radiation | Optional CDS API token; Open-Meteo gateway keyless |
| **SRTM GL1** | NASA / USGS / ESA STEP Mirror | 1 arc-second ($\sim 30\text{ m}$) | Static (Feb 2000) | EPSG:4326 | Orthometric surface elevation ($z$ in meters) | None (Public STEP mirror) |
| **Copernicus DEM** | ESA / Airbus / OpenTopography | 1 arc-second (GLO-30, $\sim 30\text{ m}$) | Static (2011–2015 baseline) | EPSG:4326 | Digital surface model elevation | Free OpenTopography API key / AWS open data |
| **Global Flood DB** | Cloud to Street / DFO (Tellman et al. 2021) | $\sim 250\text{ m}$ (MODIS composite) | Event Composite (multi-day) | EPSG:4326 | Band 1: `flooded` extent, Band 2: `duration`, Band 5: `jrc_perm_water` | None (Public GCS bucket) |
| **JRC Surface Water** | European Commission Joint Research Centre | $30\text{ m}$ | Long-term Seasonality (1984–present) | EPSG:4326 | Water occurrence, permanence, seasonality | None (Public JRC mirror) |

---

## 3. Detailed Client Specifications & API Protocols

### 3.1 NASA POWER Daily Meteorology Client
* **Implementation:** `src/data/acquisition/nasa_power.py` (`NasaPowerClient`)
* **Endpoint:** `https://power.larc.nasa.gov/api/temporal/daily/point`
* **Query Format:**
  ```http
  GET /api/temporal/daily/point?parameters=PRECTOTCORR,T2M,RH2M,WS10M&community=AG&longitude=93.35&latitude=26.65&start=20150601&end=20150615&format=JSON HTTP/1.1
  Host: power.larc.nasa.gov
  User-Agent: GeoFlood-AI-Research/2.0
  ```
* **Missing Value Protocol:** NASA POWER codes missing values as `-999.0`. The client flags values $< -900.0$, converts them to IEEE `NaN`, and increments the provenance missing count.
* **Rate Limits:** Maximum 30 concurrent requests per IP. Exponential backoff is automatically triggered upon HTTP 429.

### 3.2 CHIRPS Daily Precipitation Client
* **Implementation:** `src/data/acquisition/chirps.py` (`ChirpsClient`)
* **Endpoint:** `https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{YEAR}/chirps-v2.0.{YYYY.MM.DD}.tif.gz`
* **Handling:** Ingests daily gzip-compressed GeoTIFFs, decompresses them into local cache, and extracts spatial sub-grids over regional bounding boxes.
* **Missing Value Protocol:** Unobserved cells (oceans/polar zones) use $-9999.0$ and are masked with transparency.

### 3.3 ERA5-Land Reanalysis Client
* **Implementation:** `src/data/acquisition/era5.py` (`Era5Client`)
* **Primary Gateway:** `https://archive-api.open-meteo.com/v1/archive`
* **Direct ECMWF CDS Setup Instructions (Optional Alternative):**
  1. Register at `https://cds.climate.copernicus.eu/`
  2. Create configuration file `~/.cdsapirc`:
     ```ini
     url: https://cds.climate.copernicus.eu/api
     key: <USER-UID>:<API-KEY>
     ```
  3. Install `cdsapi`: `pip install cdsapi`

### 3.4 Multi-Source DEM Client (SRTMGL1 & Copernicus GLO-30)
* **Implementation:** `src/data/acquisition/dem_client.py` (`DemClient`)
* **Tile Calculation Engine:** `RegionManager.get_srtm_tiles_for_bbox(bbox)` dynamically computes all required $1^\circ \times 1^\circ$ geographic tiles (`NxxEyyy`).
* **Mirror:** `https://step.esa.int/auxdata/dem/SRTMGL1/{TILE}.SRTMGL1.hgt.zip`
* **Processing Chain:**
  1. Downloads and decompresses `.hgt` raster archives.
  2. Mosaics contiguous tiles using `rasterio.merge.merge()`.
  3. Crops precisely to the regional bounding box.
  4. Nodata value $-32768$ is handled via `UnifiedRasterProcessor.handle_missing_dem()`, imputing local median elevation without erasing the missing mask.

### 3.5 Global Flood Database (GFD) Client
* **Implementation:** `src/data/acquisition/gfd_client.py` (`GfdClient`)
* **Mirror URL:** `https://storage.googleapis.com/gfd_v1_4/{EVENT_ID}_From_{START}_to_{END}.zip`
* **Multi-Band Extraction:**
  - **Band 1 (Inundated Land):** Pixel value `1.0` indicates verified water on land during the optical satellite overpass composite; `0.0` indicates dry land; `NaN` indicates cloud obstruction.
  - **Band 5 (Permanent Water):** Sourced from the JRC Global Surface Water dataset; pixel value `1.0` identifies perennial rivers and permanent reservoirs. These pixels are assigned label `-1.0` in the feature matrix and strictly excluded from terrestrial flood susceptibility loss calculation.

---

## 4. Regional Target Data Availability

| Target Region | Primary Basin | DEM Tile Set | Primary Benchmark Flood Event | Secondary Benchmark Event | Data Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Assam** | Brahmaputra | N26E092, N26E093, N27E092, N27E093 | DFO_4259 (June 2015 Wave 1) | DFO_4288 (August 2015 Wave 2) | Full Raw & Processed Ingested |
| **Kerala** | Periyar-Pamba | N09E076, N09E077, N10E076, N10E077 | DFO_4684 (August 2018 Deluge) | DFO_4803 (August 2019 Wave) | Configured & Client Validated |
| **Mahanadi** | Mahanadi Delta | N20E085, N20E086 | DFO_3847 (September 2011 Super Flood) | DFO_4945 (August 2020 Wave) | Configured & Client Validated |
| **Godavari** | Lower Godavari | N16E080, N16E081, N17E080, N17E081 | DFO_5182 (July 2022 Historic Deluge) | DFO_3112 (August 2006 Flood) | Configured & Client Validated |
