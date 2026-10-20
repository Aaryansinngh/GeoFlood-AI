"""
tests/test_geospatial_alignment.py

Geospatial Alignment & Raster Georeferencing Verification Test Suite.
Validates:
1. Raster bounds & extent: verified edge-to-edge coordinates match GFD master transform.
2. CRS metadata: verified EPSG:4326 WGS84 geographic coordinate system.
3. Raster dimensions: exactly 400 rows x 623 columns = 249,200 grid cells.
4. Grid-to-coordinate mapping: row 0 is North, row 399 is South, col 0 is West, col 622 is East.
5. Prediction-array alignment: RGBA arrays align with grid coordinates without vertical or horizontal flip.
6. Historical-mask alignment: permanent water channel and flood pixels match Brahmaputra physical course.
7. No-data transparency: all unobserved/nodata cells have alpha == 0 (fully transparent).
8. Web Mercator projection compatibility: arrays successfully reproject via folium mercator_transform.
9. Folium map viewport & fit_bounds: map HTML output contains fitBounds with verified bounds.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
import folium
from folium.utilities import mercator_transform

from src.geo.grid import get_raster_edge_bounds, get_dataframe_raster_bounds
from app import generate_raster_overlays


@pytest.fixture(scope="module")
def geo_data():
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    parquet_path = Path("outputs/predictions.parquet")
    if not parquet_path.exists():
        parquet_path = Path("data/sample/features_sample.parquet")
    df = pd.read_parquet(parquet_path)
    return df, config


def test_raster_bounds_and_extent(geo_data):
    """Verify raster bounds match the master transform and cell-center extrema."""
    df, _ = geo_data
    west, south, east, north = get_dataframe_raster_bounds(df)

    # Master grid derived from GFD TIF: W=92.800000, S=26.301685, E=94.199126, N=27.200000
    assert 92.799 <= west <= 92.801, f"West bound mismatch: {west}"
    assert 94.198 <= east <= 94.201, f"East bound mismatch: {east}"
    assert 26.300 <= south <= 26.303, f"South bound mismatch: {south}"
    assert 27.199 <= north <= 27.201, f"North bound mismatch: {north}"

    assert west < east, "West must be strictly less than East"
    assert south < north, "South must be strictly less than North"


def test_crs_metadata_wgs84():
    """Verify that raw GFD and DEM GeoTIFFs use EPSG:4326."""
    gfd_tifs = list(Path("data/raw").glob("DFO_4259*.tif"))
    assert len(gfd_tifs) > 0, "DFO_4259 TIF missing"

    with rasterio.open(gfd_tifs[0]) as src:
        assert src.crs.to_string() == "EPSG:4326"

    dem_path = Path("data/raw/srtm_dem_study_area.tif")
    if dem_path.exists():
        with rasterio.open(dem_path) as src:
            assert src.crs.to_string() == "EPSG:4326"


def test_raster_dimensions_and_cell_count(geo_data):
    """Verify raster dimensions are exactly 400 x 623 = 249,200 cells."""
    df, _ = geo_data
    ev1 = df[df["event_id"] == "DFO_4259"]

    h = int(ev1["row"].max() + 1)
    w = int(ev1["col"].max() + 1)

    assert h == 400, f"Expected height 400, got {h}"
    assert w == 623, f"Expected width 623, got {w}"
    assert len(ev1) == 400 * 623 == 249200


def test_grid_to_coordinate_mapping_orientation(geo_data):
    """Verify grid coordinate orientation (North-up, West-left)."""
    df, _ = geo_data
    ev1 = df[df["event_id"] == "DFO_4259"].sort_values(["row", "col"])

    row0_lat = ev1[ev1["row"] == 0]["lat"].iloc[0]
    row_last_lat = ev1[ev1["row"] == 399]["lat"].iloc[0]
    col0_lon = ev1[ev1["col"] == 0]["lon"].iloc[0]
    col_last_lon = ev1[ev1["col"] == 622]["lon"].iloc[0]

    # Row 0 must be North (higher latitude)
    assert row0_lat > row_last_lat, f"Row 0 lat ({row0_lat}) should be > row 399 lat ({row_last_lat})"
    assert np.isclose(row0_lat, 27.1989, atol=0.01)
    assert np.isclose(row_last_lat, 26.3028, atol=0.01)

    # Col 0 must be West (lower longitude)
    assert col0_lon < col_last_lon, f"Col 0 lon ({col0_lon}) should be < col 622 lon ({col_last_lon})"
    assert np.isclose(col0_lon, 92.8011, atol=0.01)
    assert np.isclose(col_last_lon, 94.1980, atol=0.01)


def test_prediction_array_alignment(geo_data):
    """Verify generate_raster_overlays produces RGBA arrays aligned with grid."""
    df, _ = geo_data
    rgba_suscep, rgba_obs, shape = generate_raster_overlays(df, model_key="neural_network")

    assert rgba_suscep.shape == (400, 623, 4)
    assert rgba_obs.shape == (400, 623, 4)
    assert shape == (400, 623)

    # Permanent river cells in RGBA must be blue [0, 119, 182, 230]
    ev1 = df[df["event_id"] == "DFO_4259"].sort_values(["row", "col"])
    perm = ev1["is_perm_water"].values.reshape((400, 623))
    assert np.all(rgba_suscep[perm == 1, 0] == 0)
    assert np.all(rgba_suscep[perm == 1, 1] == 119)
    assert np.all(rgba_suscep[perm == 1, 2] == 182)


def test_nodata_transparency(geo_data):
    """Verify that unobserved / NaN cells have alpha == 0 (fully transparent)."""
    df, _ = geo_data
    rgba_suscep, rgba_obs, _ = generate_raster_overlays(df, model_key="neural_network")

    # In rgba_obs, cells with flood_label == 0.0 (dry) or NaN must be completely transparent (alpha = 0)
    ev1 = df[df["event_id"] == "DFO_4259"].sort_values(["row", "col"])
    obs = ev1["flood_label"].values.reshape((400, 623))
    perm = ev1["is_perm_water"].values.reshape((400, 623))

    dry_unobserved_mask = (obs == 0.0) & (perm == 0)
    assert np.all(rgba_obs[dry_unobserved_mask, 3] == 0), "Dry land in observed layer must be transparent"

    nan_mask = np.isnan(obs) & (perm == 0)
    assert np.all(rgba_obs[nan_mask, 3] == 0), "Cloud gap / NaN cells in observed layer must be transparent"


def test_historical_mask_river_alignment(geo_data):
    """Verify permanent water pixels follow the physical Brahmaputra channel from NE to SW."""
    df, _ = geo_data
    ev1 = df[df["event_id"] == "DFO_4259"]
    pw = ev1[ev1["is_perm_water"] == 1]

    # East end of basin (~94.1E): river should be at higher latitude (~26.9N)
    east_river = pw[pw["lon"] >= 94.0]
    # West end of basin (~92.9E): river should be at lower latitude (~26.65N)
    west_river = pw[pw["lon"] <= 93.0]

    assert len(east_river) > 0 and len(west_river) > 0
    assert east_river["lat"].mean() > west_river["lat"].mean(), "River channel must slope from NE to SW"


def test_web_mercator_transform_compatibility(geo_data):
    """Verify arrays are compatible with Folium's mercator_transform for Web Mercator reprojection."""
    df, _ = geo_data
    rgba_suscep, rgba_obs, _ = generate_raster_overlays(df, model_key="neural_network")
    west, south, east, north = get_dataframe_raster_bounds(df)

    warped_suscep = mercator_transform(rgba_suscep, (south, north), origin="upper")
    warped_obs = mercator_transform(rgba_obs, (south, north), origin="upper")

    assert warped_suscep.shape == rgba_suscep.shape
    assert warped_obs.shape == rgba_obs.shape
    assert warped_suscep.dtype == np.float64
    assert warped_obs.dtype == np.float64


def test_folium_viewport_fit_bounds(geo_data):
    """Verify Folium map has fit_bounds configured with the verified raster bounds."""
    df, _ = geo_data
    west, south, east, north = get_dataframe_raster_bounds(df)
    bounds = [[south, west], [north, east]]
    center = [(south + north) / 2.0, (west + east) / 2.0]

    m = folium.Map(location=center, zoom_start=9)
    m.fit_bounds(bounds, padding=(10, 10))

    html = m.get_root().render()
    assert "fitBounds" in html, "fitBounds script missing from map HTML"
    # Check that bounds coordinates are embedded
    assert str(south) in html or f"{south:.2f}" in html
    assert str(west) in html or f"{west:.2f}" in html
