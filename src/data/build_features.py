"""
src/data/build_features.py

Aligns raw satellite rasters (GFD flood labels, SRTM elevation, precipitation grids)
to a unified ~250m grid (EPSG:4326), computes terrain derivatives (slope, TPI, relative elevation),
interpolates rainfall features, assigns spatial block IDs, and saves:
1. data/processed/features.parquet (full dataset for both events)
2. data/sample/features_sample.parquet (lightweight representative sample for tests & offline runs)
"""

import os
import json
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.interpolate import Rbf

from src.geo.grid import (
    get_grid_coordinates,
    resample_raster_to_grid,
    calculate_slope,
    calculate_tpi,
    calculate_relative_elevation,
    assign_spatial_blocks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def interpolate_rainfall_surface(rainfall_json_path: Path, grid_lats: np.ndarray, grid_lons: np.ndarray):
    """
    Interpolate station precipitation records to the full 2D master grid using Thin Plate Splines.
    """
    with open(rainfall_json_path, "r", encoding="utf-8") as f:
        stations = json.load(f)

    st_lats = np.array([s["lat"] for s in stations])
    st_lons = np.array([s["lon"] for s in stations])
    st_3d = np.array([s["rain_3d"] for s in stations])
    st_7d = np.array([s["rain_7d"] for s in stations])
    st_sum = np.array([s["rain_event_sum"] for s in stations])

    rbf_3d = Rbf(st_lons, st_lats, st_3d, function="thin_plate")
    rbf_7d = Rbf(st_lons, st_lats, st_7d, function="thin_plate")
    rbf_sum = Rbf(st_lons, st_lats, st_sum, function="thin_plate")

    grid_rain_3d = rbf_3d(grid_lons, grid_lats).astype(np.float32)
    grid_rain_7d = rbf_7d(grid_lons, grid_lats).astype(np.float32)
    grid_rain_sum = rbf_sum(grid_lons, grid_lats).astype(np.float32)

    return grid_rain_sum, grid_rain_3d, grid_rain_7d


def process_event(event_key: str, config: dict, raw_dir: Path,
                  master_transform: rasterio.Affine, master_shape: tuple,
                  elev_grid: np.ndarray, slope_grid: np.ndarray,
                  tpi_grid: np.ndarray, rel_elev_grid: np.ndarray,
                  spatial_blocks: np.ndarray,
                  grid_lats: np.ndarray, grid_lons: np.ndarray) -> pd.DataFrame:
    """
    Extract labels and construct feature rows for one flood event.
    """
    event_cfg = config["events"][event_key]
    event_id = event_cfg["event_id"]
    bbox = config["region"]["bbox"]

    # 1. Read GFD flood raster
    tif_candidates = list(raw_dir.glob(f"{event_id}*.tif"))
    if not tif_candidates:
        raise FileNotFoundError(f"GFD GeoTIFF for {event_id} not found in {raw_dir}")
    gfd_tif = tif_candidates[0]

    with rasterio.open(gfd_tif) as src:
        window = from_bounds(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"], src.transform)
        flooded_raw = src.read(1, window=window)
        perm_water_raw = src.read(5, window=window)

    logger.info("Event %s raw cropped shape: %s", event_id, flooded_raw.shape)

    # 2. Interpolate rainfall
    rain_json = raw_dir / f"rainfall_{event_id}.json"
    rain_sum, rain_3d, rain_7d = interpolate_rainfall_surface(rain_json, grid_lats, grid_lons)

    # 3. Create DataFrame rows
    h, w = master_shape
    rows = []
    
    # Check for label leakage: Ensure NO flood-derived metrics enter the feature matrix
    for r in range(h):
        for c in range(w):
            f_val = flooded_raw[r, c]
            p_val = perm_water_raw[r, c]

            # Missing or unobserved pixel: label as NaN/unavailable
            if np.isnan(f_val) or f_val < 0:
                label = np.nan
            elif p_val == 1.0:
                # Permanent water: mark label as -1 or exclude from dry-land inundation modeling
                label = -1.0
            else:
                label = float(f_val)

            cell_id = f"{r:03d}_{c:03d}"
            rows.append({
                "cell_id": cell_id,
                "row": r,
                "col": c,
                "lat": float(grid_lats[r, c]),
                "lon": float(grid_lons[r, c]),
                "event_id": event_id,
                "spatial_block_id": int(spatial_blocks[r, c]),
                "rain_event_sum": float(rain_sum[r, c]),
                "rain_3d": float(rain_3d[r, c]),
                "rain_7d": float(rain_7d[r, c]),
                "elevation": float(elev_grid[r, c]),
                "slope": float(slope_grid[r, c]),
                "tpi": float(tpi_grid[r, c]),
                "relative_elevation": float(rel_elev_grid[r, c]),
                "is_perm_water": int(p_val == 1.0),
                "flood_label": label
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Event %s total cells: %d, Valid binary land cells: %d, Flooded: %d (%.2f%%)",
        event_id,
        len(df),
        (df["flood_label"].isin([0.0, 1.0])).sum(),
        (df["flood_label"] == 1.0).sum(),
        100.0 * (df["flood_label"] == 1.0).sum() / (df["flood_label"].isin([0.0, 1.0])).sum()
    )
    return df


def build_all_features():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    proc_dir = Path(config["paths"]["processed_dir"])
    sample_dir = Path(config["paths"]["sample_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    bbox = config["region"]["bbox"]
    primary_id = config["events"]["primary"]["event_id"]
    primary_tif = list(raw_dir.glob(f"{primary_id}*.tif"))[0]

    # Establish master grid reference from the cropped primary GFD raster
    with rasterio.open(primary_tif) as src:
        window = from_bounds(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"], src.transform)
        master_transform = rasterio.windows.transform(window, src.transform)
        master_shape = (int(window.height), int(window.width))
        master_crs = src.crs

    logger.info("Master Grid Shape: %s, Transform: %s", master_shape, master_transform)

    # 1. Coordinate grids
    grid_lats, grid_lons = get_grid_coordinates(master_transform, master_shape[0], master_shape[1])

    # 2. Resample SRTM DEM to master grid
    dem_path = raw_dir / "srtm_dem_study_area.tif"
    logger.info("Resampling SRTM DEM to master grid...")
    elev_grid = resample_raster_to_grid(
        src_path=str(dem_path),
        dst_crs=master_crs.to_string(),
        dst_transform=master_transform,
        dst_shape=master_shape
    )

    # Clean any DEM nodata
    elev_grid[elev_grid < -100] = np.nan
    nan_mask = np.isnan(elev_grid)
    if nan_mask.any():
        logger.warning("Imputing %d missing DEM cells with local mean", nan_mask.sum())
        mean_val = np.nanmean(elev_grid)
        elev_grid[nan_mask] = mean_val

    # 3. Terrain derivatives
    logger.info("Calculating slope, TPI, and relative elevation...")
    slope_grid = calculate_slope(elev_grid, master_transform)
    tpi_grid = calculate_tpi(elev_grid, window_size=15)
    rel_elev_grid = calculate_relative_elevation(elev_grid, window_size=25)

    # 4. Spatial Block Partitioning
    split_cfg = config["data_split"]
    spatial_blocks = assign_spatial_blocks(
        height=master_shape[0],
        width=master_shape[1],
        n_blocks_y=split_cfg.get("n_blocks_y", 3),
        n_blocks_x=split_cfg.get("n_blocks_x", 4)
    )

    # 5. Process both events
    df_primary = process_event(
        "primary", config, raw_dir, master_transform, master_shape,
        elev_grid, slope_grid, tpi_grid, rel_elev_grid,
        spatial_blocks, grid_lats, grid_lons
    )
    df_secondary = process_event(
        "secondary", config, raw_dir, master_transform, master_shape,
        elev_grid, slope_grid, tpi_grid, rel_elev_grid,
        spatial_blocks, grid_lats, grid_lons
    )

    df_full = pd.concat([df_primary, df_secondary], ignore_index=True)

    # Output paths
    features_parquet = proc_dir / "features.parquet"
    logger.info("Writing full feature dataset to %s (%d rows)...", features_parquet, len(df_full))
    df_full.to_parquet(features_parquet, index=False)

    # Create stratified compact sample (e.g. 8,000 rows, preserving class & spatial block distribution)
    # Filter to valid binary cells for the primary event to give a clean offline sample
    valid_primary = df_primary[df_primary["flood_label"].isin([0.0, 1.0])].copy()
    sample_df = valid_primary.groupby(["spatial_block_id", "flood_label"], group_keys=False).apply(
        lambda g: g.sample(frac=0.06, random_state=42)
    )
    sample_parquet = sample_dir / "features_sample.parquet"
    sample_csv = sample_dir / "features_sample.csv"
    logger.info("Writing sample dataset to %s and %s (%d rows)...", sample_parquet, sample_csv, len(sample_df))
    sample_df.to_parquet(sample_parquet, index=False)
    sample_df.to_csv(sample_csv, index=False)

    logger.info("Feature engineering complete!")
    return df_full


def build_features_for_region(region_id: str = "assam", mode: str = "retrospective") -> pd.DataFrame:
    """
    GeoFlood-AI 2.0 multi-regional feature pipeline entrypoint.
    """
    from src.data.preprocessing.feature_pipeline import RegionalFeaturePipeline
    from src.data.validation.leakage_guard import ModelingMode

    m = ModelingMode.PRE_EVENT_FORECASTING if mode == "forecasting" else ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY
    pipeline = RegionalFeaturePipeline(region_id=region_id, mode=m)
    df, _ = pipeline.run_pipeline()
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GeoFlood-AI 2.0 Feature Pipeline")
    parser.add_argument("--region", type=str, default="assam", help="Target region: assam, kerala, mahanadi, godavari")
    parser.add_argument("--mode", type=str, default="retrospective", choices=["retrospective", "forecasting"], help="Modeling mode")
    args = parser.parse_args()

    if args.region == "assam" and args.mode == "retrospective":
        build_all_features()
    else:
        build_features_for_region(region_id=args.region, mode=args.mode)

