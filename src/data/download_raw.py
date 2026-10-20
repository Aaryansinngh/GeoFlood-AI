"""
src/data/download_raw.py

Downloads verified authentic raw satellite-derived datasets for the study area
when direct GEE export is not authenticated:
1. Flood extent labels & JRC water from Global Flood Database (Cloud to Street / DFO)
2. Digital Elevation Model (SRTM 30m / SRTMGL1) from ESA STEP public mirror
3. Daily precipitation (7-day, 3-day, and event cumulative rainfall)
"""

import os
import io
import zipfile
import urllib.request
import logging
from pathlib import Path
import yaml
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, chunk_size: int = 1024 * 1024):
    """Download a file with user-agent headers and progress logging."""
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("File already exists: %s", output_path.name)
        return

    logger.info("Downloading %s -> %s", url, output_path.name)
    req = urllib.request.Request(url, headers={"User-Agent": "GeoFlood-AI-Academic-Research/1.0"})
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as out:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
    logger.info("Completed download: %s (%.2f MB)", output_path.name, output_path.stat().st_size / (1024 * 1024))


def download_and_extract_gfd(event_cfg: dict, raw_dir: Path) -> Path:
    """Download GFD zip archive and extract the GeoTIFF."""
    zip_url = event_cfg["gfd_zip_url"]
    event_id = event_cfg["event_id"]
    zip_path = raw_dir / f"{event_id}.zip"
    download_file(zip_url, zip_path)

    tif_name = None
    with zipfile.ZipFile(zip_path, "r") as zf:
        for fname in zf.namelist():
            if fname.endswith(".tif"):
                tif_name = fname
                out_tif = raw_dir / fname
                if not out_tif.exists():
                    logger.info("Extracting %s from zip...", fname)
                    with open(out_tif, "wb") as f:
                        f.write(zf.read(fname))
                else:
                    logger.info("Extracted raster already present: %s", fname)
                return out_tif
    raise FileNotFoundError(f"No GeoTIFF found inside {zip_path}")


def download_and_merge_srtm(terrain_cfg: dict, bbox_cfg: dict, raw_dir: Path) -> Path:
    """Download SRTM .hgt.zip tiles and merge them into a cropped DEM."""
    output_dem = raw_dir / "srtm_dem_study_area.tif"
    if output_dem.exists():
        logger.info("Merged SRTM DEM already exists: %s", output_dem.name)
        return output_dem

    dem_dir = raw_dir / "srtm_tiles"
    dem_dir.mkdir(parents=True, exist_ok=True)

    hgt_files = []
    base_url = terrain_cfg["srtm_mirror_url"]
    for tile in terrain_cfg["srtm_tiles"]:
        zip_url = f"{base_url}/{tile}.SRTMGL1.hgt.zip"
        tile_zip = dem_dir / f"{tile}.SRTMGL1.hgt.zip"
        download_file(zip_url, tile_zip)

        hgt_name = f"{tile}.hgt"
        extracted_hgt = dem_dir / hgt_name
        if not extracted_hgt.exists():
            with zipfile.ZipFile(tile_zip, "r") as zf:
                with open(extracted_hgt, "wb") as f:
                    f.write(zf.read(hgt_name))
        hgt_files.append(extracted_hgt)

    logger.info("Merging %d SRTM tiles...", len(hgt_files))
    src_files = [rasterio.open(f) for f in hgt_files]
    mosaic, out_trans = merge(src_files)

    for src in src_files:
        src.close()

    # Crop precisely to bounding box
    min_lon = bbox_cfg["min_lon"]
    max_lon = bbox_cfg["max_lon"]
    min_lat = bbox_cfg["min_lat"]
    max_lat = bbox_cfg["max_lat"]

    # Write merged DEM
    out_meta = {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": mosaic.dtype,
        "nodata": -32768
    }

    temp_merged = dem_dir / "temp_merged.tif"
    with rasterio.open(temp_merged, "w", **out_meta) as dst:
        dst.write(mosaic)

    with rasterio.open(temp_merged) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        cropped_data = src.read(1, window=window)
        cropped_trans = rasterio.windows.transform(window, src.transform)
        cropped_meta = src.meta.copy()
        cropped_meta.update({
            "height": cropped_data.shape[0],
            "width": cropped_data.shape[1],
            "transform": cropped_trans
        })
        with rasterio.open(output_dem, "w", **cropped_meta) as dst:
            dst.write(cropped_data, 1)

    if temp_merged.exists():
        temp_merged.unlink()

    logger.info("Saved cropped SRTM DEM: %s (shape: %s)", output_dem.name, cropped_data.shape)
    return output_dem


def download_precipitation_grid(event_cfg: dict, bbox_cfg: dict, raw_dir: Path) -> Path:
    """
    Download daily precipitation for the event window and 7 preceding days
    across a representative spatial grid over the study region using the open archive API.
    """
    event_id = event_cfg["event_id"]
    out_json = raw_dir / f"rainfall_{event_id}.json"
    if out_json.exists():
        logger.info("Rainfall records already exist: %s", out_json.name)
        return out_json

    from datetime import datetime, timedelta
    began_dt = datetime.strptime(event_cfg["began"], "%Y-%m-%d")
    ended_dt = datetime.strptime(event_cfg["ended"], "%Y-%m-%d")
    pre_start_dt = began_dt - timedelta(days=7)

    start_str = pre_start_dt.strftime("%Y-%m-%d")
    end_str = ended_dt.strftime("%Y-%m-%d")

    # Sample a grid of 5x5 stations across the bounding box
    lats = np.linspace(bbox_cfg["min_lat"], bbox_cfg["max_lat"], 5)
    lons = np.linspace(bbox_cfg["min_lon"], bbox_cfg["max_lon"], 5)

    records = []
    logger.info("Fetching daily precipitation grid for %s (%s to %s)...", event_id, start_str, end_str)
    for lat in lats:
        for lon in lons:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat:.4f}&longitude={lon:.4f}&start_date={start_str}&end_date={end_str}"
                f"&daily=precipitation_sum&timezone=UTC"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "GeoFlood-AI/1.0"})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                daily_dates = data["daily"]["time"]
                daily_precip = [p if p is not None else 0.0 for p in data["daily"]["precipitation_sum"]]

                # Map dates
                date_to_rain = dict(zip(daily_dates, daily_precip))
                
                # Pre-event 3-day sum
                p3_dates = [(began_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 4)]
                rain_3d = sum(date_to_rain.get(d, 0.0) for d in p3_dates)

                # Pre-event 7-day sum
                p7_dates = [(began_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
                rain_7d = sum(date_to_rain.get(d, 0.0) for d in p7_dates)

                # Event-window sum
                event_days = (ended_dt - began_dt).days + 1
                ev_dates = [(began_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(event_days)]
                rain_event_sum = sum(date_to_rain.get(d, 0.0) for d in ev_dates)

                records.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "rain_3d": float(rain_3d),
                    "rain_7d": float(rain_7d),
                    "rain_event_sum": float(rain_event_sum)
                })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info("Saved precipitation grid (%d stations) for %s to %s", len(records), event_id, out_json.name)
    return out_json


def download_all_raw_data(config: dict, raw_dir: Path):
    """Orchestrate downloading all raw satellite assets."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    bbox = config["region"]["bbox"]

    logger.info("1. Downloading Global Flood Database rasters...")
    download_and_extract_gfd(config["events"]["primary"], raw_dir)
    download_and_extract_gfd(config["events"]["secondary"], raw_dir)

    logger.info("2. Downloading SRTM Elevation tiles...")
    download_and_merge_srtm(config["terrain"], bbox, raw_dir)

    logger.info("3. Downloading Precipitation time series...")
    download_precipitation_grid(config["events"]["primary"], bbox, raw_dir)
    download_precipitation_grid(config["events"]["secondary"], bbox, raw_dir)
    logger.info("All raw satellite data successfully acquired.")


if __name__ == "__main__":
    cfg_path = Path("config/settings.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    raw = Path(cfg["paths"]["raw_dir"])
    download_all_raw_data(cfg, raw)
