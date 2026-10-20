"""
src/data/export_gee.py

Google Earth Engine (GEE) export pipeline for GeoFlood AI.
Exports satellite rasters (GPM IMERG rainfall, SRTM DEM, and GFD flood extent)
for the study region to data/raw/.

NOTE: GEE authentication requires running `earthengine authenticate` in your shell
before executing this script. Credentials must NEVER be committed to code.
If GEE credentials are not detected, this script logs the blocker and delegates to
the direct public satellite repository mirror (gs://gfd_v1_4 and ESA STEP SRTMGL1).
"""

import os
import sys
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_from_gee(config: dict, output_dir: Path):
    """
    Export raw rasters from Google Earth Engine when authenticated.
    """
    try:
        import ee
        ee.Initialize()
        logger.info("Successfully authenticated with Google Earth Engine.")
    except Exception as e:
        logger.warning(
            "GEE Authentication failed or credentials missing: %s\n"
            "To authenticate GEE, run in terminal:\n"
            "    earthengine authenticate\n"
            "Proceeding with the verified direct public satellite mirror alternative...",
            e
        )
        return False

    import ee
    bbox = config["region"]["bbox"]
    roi = ee.Geometry.Rectangle([bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]])

    logger.info("Setting up GEE exports for ROI: %s", bbox)

    # 1. Flood Extent Labels: GLOBAL_FLOOD_DB/MODIS_EVENTS/V1
    primary_event = config["events"]["primary"]["event_id"].replace("DFO_", "")
    logger.info("Querying GFD collection for DFO Event %s", primary_event)
    gfd_col = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").filter(
        ee.Filter.eq("id", int(primary_event))
    )
    if gfd_col.size().getInfo() > 0:
        gfd_img = gfd_col.first().clip(roi)
        task = ee.batch.Export.image.toDrive(
            image=gfd_img.select(["flooded", "duration", "jrc_perm_water"]),
            description=f"GFD_Event_{primary_event}_Raw",
            folder="GeoFlood_Raw",
            fileNamePrefix=f"gfd_{primary_event}_raw",
            region=roi,
            scale=250,
            crs=config["region"]["target_crs"],
            maxPixels=1e9
        )
        task.start()
        logger.info("Started GEE Drive Export task: %s", task.id)

    # 2. SRTM DEM: USGS/SRTMGL1_003
    srtm = ee.Image("USGS/SRTMGL1_003").clip(roi)
    slope = ee.Terrain.slope(srtm)
    terrain_img = srtm.addBands(slope)
    task_dem = ee.batch.Export.image.toDrive(
        image=terrain_img,
        description="SRTM_Elevation_Slope_Raw",
        folder="GeoFlood_Raw",
        fileNamePrefix="srtm_raw",
        region=roi,
        scale=250,
        crs=config["region"]["target_crs"],
        maxPixels=1e9
    )
    task_dem.start()
    logger.info("Started GEE DEM Export task: %s", task_dem.id)

    # 3. GPM IMERG Precipitation: NASA/GPM_L3/IMERG_V06
    began = config["events"]["primary"]["began"]
    ended = config["events"]["primary"]["ended"]
    gpm = ee.ImageCollection("NASA/GPM_L3/IMERG_V06").filterDate(began, ended).filterBounds(roi)
    rain_sum = gpm.select("precipitationCal").sum().clip(roi)
    task_gpm = ee.batch.Export.image.toDrive(
        image=rain_sum,
        description="GPM_Event_Precipitation_Raw",
        folder="GeoFlood_Raw",
        fileNamePrefix="gpm_raw",
        region=roi,
        scale=250,
        crs=config["region"]["target_crs"],
        maxPixels=1e9
    )
    task_gpm.start()
    logger.info("Started GEE GPM Export task: %s", task_gpm.id)
    return True


def run_pipeline():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    gee_success = export_from_gee(config, raw_dir)
    if not gee_success:
        logger.info("Triggering direct satellite mirror downloader...")
        from src.data.download_raw import download_all_raw_data
        download_all_raw_data(config, raw_dir)


if __name__ == "__main__":
    run_pipeline()
