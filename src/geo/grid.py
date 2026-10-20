"""
src/geo/grid.py

Geospatial grid and terrain computation utilities.
Handles coordinate transformation, finite-difference slope calculation,
Topographic Position Index (TPI), and spatial block partitioning.
"""

import numpy as np
from scipy.ndimage import uniform_filter
import rasterio
from rasterio.warp import reproject, Resampling


def get_grid_coordinates(transform: rasterio.Affine, height: int, width: int):
    """
    Generate 2D arrays of latitudes and longitudes for each cell in the raster grid.
    Returns (lats, lons) as float32 arrays of shape (height, width).
    """
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lons = np.array(xs, dtype=np.float32).reshape((height, width))
    lats = np.array(ys, dtype=np.float32).reshape((height, width))
    return lats, lons


def resample_raster_to_grid(src_path: str, dst_crs: str, dst_transform: rasterio.Affine,
                            dst_shape: tuple, resampling=Resampling.bilinear) -> np.ndarray:
    """
    Resample an external raster (e.g., DEM) onto the master grid.
    """
    with rasterio.open(src_path) as src:
        destination = np.empty(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling
        )
    return destination


def calculate_slope(elevation: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """
    Calculate terrain slope in degrees using 2nd-order central finite differences.
    Accounts for latitude-dependent meter-per-degree scaling.
    """
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    # Estimate latitude at center of grid for longitude degree-to-meter conversion
    center_y = transform.f + (elevation.shape[0] / 2.0) * transform.e
    dx_meters = res_x * 111320.0 * np.cos(np.radians(center_y))
    dy_meters = res_y * 111320.0

    # Gradient computation: np.gradient returns [df/dy, df/dx]
    dy, dx = np.gradient(elevation, dy_meters, dx_meters)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return slope_deg


def calculate_tpi(elevation: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute Topographic Position Index (TPI) = elevation - local_mean_elevation.
    Negative TPI indicates topographic hollows / valleys; positive indicates ridges.
    """
    local_mean = uniform_filter(elevation, size=window_size, mode="nearest")
    tpi = (elevation - local_mean).astype(np.float32)
    return tpi


def calculate_relative_elevation(elevation: np.ndarray, window_size: int = 25) -> np.ndarray:
    """
    Compute local relative elevation (elevation - local_min_elevation)
    as a proxy for Height Above Drainage (HAND) within a moving window.
    """
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(elevation, size=window_size, mode="nearest")
    rel_elev = np.clip(elevation - local_min, 0.0, None).astype(np.float32)
    return rel_elev


def assign_spatial_blocks(height: int, width: int, n_blocks_y: int = 3, n_blocks_x: int = 4) -> np.ndarray:
    """
    Partition the 2D grid into n_blocks_y * n_blocks_x contiguous rectangular spatial tiles.
    Returns an integer array of shape (height, width) with block IDs in [0, n_blocks_y * n_blocks_x - 1].
    Used to guarantee zero spatial overlap between train and test blocks.
    """
    y_splits = np.array_split(np.arange(height), n_blocks_y)
    x_splits = np.array_split(np.arange(width), n_blocks_x)

    block_grid = np.zeros((height, width), dtype=np.int32)
    block_id = 0
    for by_idx, y_indices in enumerate(y_splits):
        for bx_idx, x_indices in enumerate(x_splits):
            block_grid[np.ix_(y_indices, x_indices)] = block_id
            block_id += 1

    return block_grid


def get_raster_edge_bounds(transform: rasterio.Affine, height: int, width: int):
    """
    Compute the exact outer geographic bounding box (west, south, east, north)
    of a raster grid from its affine transform and dimensions.
    Returns (west, south, east, north) in CRS units (degrees for EPSG:4326).
    """
    west = float(transform.c)
    north = float(transform.f)
    east = float(west + width * transform.a)
    south = float(north + height * transform.e)
    return west, south, east, north


def get_dataframe_raster_bounds(df):
    """
    Derive the exact outer geographic bounding box (west, south, east, north)
    from a gridded feature or prediction DataFrame.
    Accounts for cell-center to edge half-pixel offsets.
    Returns (west, south, east, north).
    """
    if "event_id" in df.columns:
        ev1 = df[df["event_id"] == "DFO_4259"]
    else:
        ev1 = df

    h = int(ev1["row"].max() + 1)
    w = int(ev1["col"].max() + 1)

    min_lon = float(ev1["lon"].min())
    max_lon = float(ev1["lon"].max())
    min_lat = float(ev1["lat"].min())
    max_lat = float(ev1["lat"].max())

    res_x = (max_lon - min_lon) / (w - 1) if w > 1 else 0.002245788
    res_y = (max_lat - min_lat) / (h - 1) if h > 1 else 0.002245788

    west = min_lon - res_x / 2.0
    east = max_lon + res_x / 2.0
    south = min_lat - res_y / 2.0
    north = max_lat + res_y / 2.0

    return west, south, east, north
