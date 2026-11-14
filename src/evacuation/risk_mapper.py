"""Historical flood screening for the Guwahati evacuation study area.

Important:
- DFO observations are historical event evidence, not future probabilities.
- Terrain vulnerability is unavailable until a Guwahati DEM is supplied.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
from pyproj import Transformer


@dataclass
class RiskMapResult:
    risk: np.ndarray
    historical_flood: np.ndarray
    flood_duration: np.ndarray
    terrain_vulnerability: np.ndarray
    permanent_water: np.ndarray
    transform: object
    crs: str
    bounds: tuple
    source_label: str = "HISTORICAL FLOOD SCREENING — NOT A PROBABILITY"

    @property
    def shape(self):
        return self.risk.shape


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max normalisation that preserves NaN values."""
    values = np.asarray(values, dtype=np.float32)
    result = np.full(values.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(values)
    if not valid.any():
        return result

    minimum = float(np.min(values[valid]))
    maximum = float(np.max(values[valid]))

    if maximum == minimum:
        result[valid] = 0.0
    else:
        result[valid] = (values[valid] - minimum) / (maximum - minimum)

    return result


def _projected_grid(src, destination_crs):
    transform, width, height = calculate_default_transform(
        src.crs,
        destination_crs,
        src.width,
        src.height,
        *src.bounds,
    )
    return transform, width, height


def _reproject_band(
    path: str,
    band: int,
    destination_transform,
    destination_crs,
    width: int,
    height: int,
    resampling=Resampling.nearest,
):
    """Reproject one raster band while preserving invalid pixels."""
    with rasterio.open(path) as src:
        if band > src.count:
            raise ValueError(
                f"{path} has {src.count} bands; requested band {band}"
            )

        source = src.read(band, masked=True)
        source_data = source.astype(np.float32).filled(np.nan)

        destination = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )

        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

        return destination


def build_guwahati_risk_map(
    dem_path: Optional[str] = None,
    historical_event_paths: Optional[Sequence[str]] = None,
    latitude: float = 26.1445,
    longitude: float = 91.7362,
    radius_m: float = 5000,
    projected_crs: str = "EPSG:32646",
    historical_band: int = 1,
    duration_band: int = 2,
    clear_views_band: int = 3,
    clear_percentage_band: int = 4,
    permanent_water_band: int = 5,
    minimum_clear_views: float = 1,
    minimum_clear_percentage: float = 0.0,
) -> RiskMapResult:
    """Build a historical flood screening surface around Guwahati.

    dem_path is retained for interface compatibility but is deliberately
    not used until a DEM covering the Guwahati study area is available.

    The risk array represents the fraction of valid historical event
    observations classified as flooded. It is not a forecast probability.
    """

    del dem_path  # Avoid accidentally using the unrelated Assam DEM.

    if not historical_event_paths:
        raise ValueError("At least one historical DFO raster is required.")

    for path in historical_event_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Historical raster not found: {path}")

    geographic_crs = "EPSG:4326"
    transformer = Transformer.from_crs(
        geographic_crs,
        projected_crs,
        always_xy=True,
    )

    centre_x, centre_y = transformer.transform(longitude, latitude)

    # Use the first DFO event as the target grid.
    with rasterio.open(historical_event_paths[0]) as reference:
        transform, width, height = _projected_grid(
            reference,
            projected_crs,
        )

    # Reprojected raster bounds.
    left, bottom, right, top = array_bounds(
        height,
        width,
        transform,
    )

    # Crop the projected grid to the study-area bounding square.
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    col_start = max(
        0,
        int(np.floor((centre_x - radius_m - left) / pixel_width)),
    )
    col_end = min(
        width,
        int(np.ceil((centre_x + radius_m - left) / pixel_width)),
    )

    row_start = max(
        0,
        int(np.floor((top - (centre_y + radius_m)) / pixel_height)),
    )
    row_end = min(
        height,
        int(np.ceil((top - (centre_y - radius_m)) / pixel_height)),
    )

    if col_end <= col_start or row_end <= row_start:
        raise ValueError(
            "The Guwahati study area does not overlap the DFO raster."
        )

    cropped_transform = rasterio.Affine(
        transform.a,
        transform.b,
        transform.c + col_start * transform.a,
        transform.d,
        transform.e,
        transform.f + row_start * transform.e,
    )

    crop_width = col_end - col_start
    crop_height = row_end - row_start

    def crop(array):
        return array[
            row_start:row_end,
            col_start:col_end,
        ].copy()

    # Circular mask in projected metres.
    rows, cols = np.indices((crop_height, crop_width))
    x = (
        cropped_transform.c
        + (cols + 0.5) * cropped_transform.a
        + (rows + 0.5) * cropped_transform.b
    )
    y = (
        cropped_transform.f
        + (cols + 0.5) * cropped_transform.d
        + (rows + 0.5) * cropped_transform.e
    )

    study_mask = (
        (x - centre_x) ** 2 + (y - centre_y) ** 2
    ) <= radius_m ** 2

    flood_observations = []
    duration_observations = []
    permanent_observations = []

    for path in historical_event_paths:
        flood = crop(
            _reproject_band(
                path,
                historical_band,
                transform,
                projected_crs,
                width,
                height,
            )
        )

        duration = crop(
            _reproject_band(
                path,
                duration_band,
                transform,
                projected_crs,
                width,
                height,
            )
        )

        clear_views = crop(
            _reproject_band(
                path,
                clear_views_band,
                transform,
                projected_crs,
                width,
                height,
            )
        )

        clear_percentage = crop(
            _reproject_band(
                path,
                clear_percentage_band,
                transform,
                projected_crs,
                width,
                height,
            )
        )

        permanent = crop(
            _reproject_band(
                path,
                permanent_water_band,
                transform,
                projected_crs,
                width,
                height,
            )
        )

        valid = (
            np.isfinite(flood)
            & np.isfinite(duration)
            & np.isfinite(clear_views)
            & np.isfinite(clear_percentage)
            & study_mask
            & (clear_views >= minimum_clear_views)
            & (clear_percentage >= minimum_clear_percentage)
        )

        flood_valid = np.where(valid, flood, np.nan)
        duration_valid = np.where(valid, duration, np.nan)

        permanent_valid = np.where(
            valid & np.isfinite(permanent),
            permanent,
            np.nan,
        )

        flood_observations.append(flood_valid)
        duration_observations.append(duration_valid)
        permanent_observations.append(permanent_valid)

    flood_stack = np.stack(flood_observations)
    duration_stack = np.stack(duration_observations)
    permanent_stack = np.stack(permanent_observations)

    valid_count = np.sum(np.isfinite(flood_stack), axis=0)
    flooded_count = np.nansum(flood_stack == 1, axis=0)

    historical_flood = np.full(
        (crop_height, crop_width),
        np.nan,
        dtype=np.float32,
    )

    observed = valid_count > 0
    historical_flood[observed] = (
        flooded_count[observed] / valid_count[observed]
    )

    flood_duration = np.full_like(historical_flood, np.nan)
    duration_valid_count = np.sum(
        np.isfinite(duration_stack),
        axis=0,
    )
    duration_observed = duration_valid_count > 0

    duration_sum = np.nansum(duration_stack, axis=0)
    flood_duration[duration_observed] = (
        duration_sum[duration_observed]
        / duration_valid_count[duration_observed]
    )

    permanent_water = np.any(
        permanent_stack == 1,
        axis=0,
    ) & study_mask

    # Outside the circular study area is always invalid.
    historical_flood[~study_mask] = np.nan
    flood_duration[~study_mask] = np.nan

    # No valid Guwahati DEM is currently available.
    terrain_vulnerability = np.full(
        (crop_height, crop_width),
        np.nan,
        dtype=np.float32,
    )

    return RiskMapResult(
        risk=historical_flood.copy(),
        historical_flood=historical_flood,
        flood_duration=flood_duration,
        terrain_vulnerability=terrain_vulnerability,
        permanent_water=permanent_water,
        transform=cropped_transform,
        crs=projected_crs,
        bounds=(
            centre_x - radius_m,
            centre_y - radius_m,
            centre_x + radius_m,
            centre_y + radius_m,
        ),
    )
