"""
CHIRPS v3 Regional Daily Rainfall Acquisition.
Reads regional pixels directly from remote COG files.
"""

import csv
import hashlib
import json
import logging
import math
import time

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from rasterio.windows import from_bounds
from rasterio.transform import xy
from affine import Affine

from src.geospatial.region_manager import BoundingBox
from src.data.acquisition.base_client import (
    BaseAcquisitionClient,
    AcquisitionResult,
    AcquisitionStatus,
)
from src.data.provenance.metadata_logger import ProvenanceTracker

logger = logging.getLogger(__name__)


class ChirpsV3RegionalClient(BaseAcquisitionClient):

    BASE_URL = "https://data.source.coop/wfp/chirps-rnl-daily/v3.0"
    DATASET_VERSION = "3.0-RNL"
    RESOLUTION = "0.05 degree (~5 km)"

    ROLLING_WINDOWS = {
        3: "rainfall_3day.csv",
        7: "rainfall_7day.csv",
        30: "rainfall_30day.csv",
    }

    def __init__(
        self,
        cache_dir=Path("data/cache/chirps_v3"),
        provenance_tracker: Optional[ProvenanceTracker] = None,
        max_retries=3,
        timeout_seconds=60,
        minimum_coverage=0.8,
    ):
        super().__init__(
            cache_dir=cache_dir,
            provenance_tracker=provenance_tracker,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )

        if not 0 < minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1].")

        self.minimum_coverage = minimum_coverage

    def _build_url(self, day):
        year = day.strftime("%Y")
        month = day.strftime("%m")
        date_string = day.strftime("%Y.%m.%d")

        return (
            f"{self.BASE_URL}/{year}/{month}/{day.strftime('%d')}/"
            f"chirps-v3.0.rnl.{date_string}.tif"
        )

    def _cache_path(self, region_id, bbox, day):
        key = hashlib.sha256(
            f"{region_id}_{bbox.as_tuple()}_{day.date()}".encode()
        ).hexdigest()[:16]

        return self.cache_dir / region_id / f"{key}.npz"

    def _read_remote_day(self, url, bbox):
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
            GDAL_HTTP_TIMEOUT=str(self.timeout_seconds),
        ):
            with rasterio.open(url) as src:

                if src.crs is None or src.crs.to_epsg() != 4326:
                    raise ValueError(f"Unexpected CHIRPS CRS: {src.crs}")

                window = from_bounds(
                    *bbox.as_tuple(),
                    transform=src.transform,
                ).round_offsets().round_lengths()

                if window.width <= 0 or window.height <= 0:
                    raise ValueError("Regional raster window is empty.")

                data = src.read(1, window=window, masked=True)

                rainfall = np.asarray(
                    data.filled(np.nan),
                    dtype=np.float32,
                )

                rainfall[~np.isfinite(rainfall)] = np.nan
                rainfall[rainfall <= -9990] = np.nan
                rainfall[rainfall < 0] = np.nan

                return (
                    rainfall,
                    src.window_transform(window),
                    int(window.row_off),
                    int(window.col_off),
                )

    def _load_day(self, region_id, bbox, day):
        cache_file = self._cache_path(region_id, bbox, day)

        if cache_file.exists():
            try:
                with np.load(cache_file, allow_pickle=False) as cached:
                    rainfall = cached["rainfall"]
                    transform = Affine(*cached["transform"].tolist())

                    if rainfall.ndim != 2 or rainfall.size == 0:
                        raise ValueError("Invalid cached raster.")

                    return (
                        rainfall.astype(np.float32),
                        transform,
                        int(cached["row_off"]),
                        int(cached["col_off"]),
                        True,
                    )
            except Exception as exc:
                logger.warning("Invalid cache %s: %s", cache_file, exc)
                cache_file.unlink(missing_ok=True)

        url = self._build_url(day)

        for attempt in range(1, self.max_retries + 1):
            try:
                rainfall, transform, row_off, col_off = (
                    self._read_remote_day(url, bbox)
                )

                cache_file.parent.mkdir(parents=True, exist_ok=True)
                temporary_file = cache_file.with_suffix(".tmp")

                with open(temporary_file, "wb") as handle:
                    np.savez_compressed(
                        handle,
                        rainfall=rainfall,
                        transform=np.array([
                            transform.a, transform.b, transform.c,
                            transform.d, transform.e, transform.f,
                        ]),
                        row_off=row_off,
                        col_off=col_off,
                    )

                temporary_file.replace(cache_file)

                return rainfall, transform, row_off, col_off, False

            except Exception as exc:
                logger.warning(
                    "Failed %s, attempt %d/%d: %s",
                    day.date(), attempt, self.max_retries, exc,
                )

                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))

        return None

    @staticmethod
    def _write_csv(path, fieldnames, records):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def _generate_daily_records(self, dates, daily_arrays, transform):
        records = []

        height, width = next(iter(daily_arrays.values())).shape

        coordinates = [
            (row, col, *xy(transform, row, col, offset="center"))
            for row in range(height)
            for col in range(width)
        ]

        for day in dates:
            rainfall = daily_arrays[day]

            for row, col, longitude, latitude in coordinates:
                value = float(rainfall[row, col])

                records.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "row": row,
                    "col": col,
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                    "rainfall_mm": (
                        round(value, 4)
                        if np.isfinite(value) else ""
                    ),
                })

        return records

    def _generate_rolling_records(
        self, dates, daily_arrays, transform, window_days
    ):
        height, width = next(iter(daily_arrays.values())).shape

        required_days = math.ceil(
            window_days * self.minimum_coverage
        )

        coordinates = [
            (row, col, *xy(transform, row, col, offset="center"))
            for row in range(height)
            for col in range(width)
        ]

        records = []

        for index, day in enumerate(dates):
            if index + 1 < window_days:
                continue

            window_dates = dates[index-window_days+1:index+1]

            window_data = np.stack([
                daily_arrays[d] for d in window_dates
            ])

            valid_counts = np.sum(np.isfinite(window_data), axis=0)
            rainfall_sums = np.nansum(window_data, axis=0)

            for row, col, longitude, latitude in coordinates:
                valid_days = int(valid_counts[row, col])
                coverage = valid_days / window_days

                value = (
                    round(float(rainfall_sums[row, col]), 4)
                    if valid_days >= required_days else ""
                )

                records.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "period_start": window_dates[0].strftime("%Y-%m-%d"),
                    "row": row,
                    "col": col,
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                    "rainfall_sum_mm": value,
                    "valid_days": valid_days,
                    "window_days": window_days,
                    "coverage_fraction": round(coverage, 3),
                })

        return records

    def acquire(
        self, region_id, bbox, start_date, end_date, output_dir
    ):
        start_time = time.time()

        valid, error = self.validate_inputs(
            bbox, start_date, end_date
        )

        if not valid:
            return AcquisitionResult(
                dataset_name="CHIRPS_V3_Regional_Rainfall",
                status=AcquisitionStatus.INVALID_INPUT,
                local_path=None,
                records_count=0,
                missing_count=0,
                spatial_resolution=self.RESOLUTION,
                temporal_resolution="Daily",
                provenance=None,
                error_message=error,
            )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        dates = [
            start + timedelta(days=i)
            for i in range((end-start).days + 1)
        ]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        daily_arrays = {}
        transforms = {}
        missing_dates = []
        cached_days = 0
        downloaded_days = 0

        for index, day in enumerate(dates, start=1):
            logger.info(
                "CHIRPS v3: %s (%d/%d)",
                day.date(), index, len(dates),
            )

            result = self._load_day(region_id, bbox, day)

            if result is None:
                missing_dates.append(day.strftime("%Y-%m-%d"))
                continue

            rainfall, transform, row_off, col_off, cached = result

            if cached:
                cached_days += 1
            else:
                downloaded_days += 1

            daily_arrays[day] = rainfall
            transforms[day] = transform

        if not daily_arrays:
            return AcquisitionResult(
                dataset_name="CHIRPS_V3_Regional_Rainfall",
                status=AcquisitionStatus.UNAVAILABLE,
                local_path=None,
                records_count=0,
                missing_count=len(dates),
                spatial_resolution=self.RESOLUTION,
                temporal_resolution="Daily",
                provenance=None,
                error_message="No daily CHIRPS rasters acquired.",
                execution_time_seconds=round(time.time()-start_time, 2),
            )

        template_day = next(iter(daily_arrays))
        template = daily_arrays[template_day]
        template_transform = transforms[template_day]

        for missing_day in missing_dates:
            day = datetime.strptime(missing_day, "%Y-%m-%d")
            daily_arrays[day] = np.full(
                template.shape, np.nan, dtype=np.float32
            )

        for day in dates:
            if day in transforms:
                if daily_arrays[day].shape != template.shape:
                    raise ValueError(f"Raster shape mismatch: {day}")

                if not np.allclose(
                    tuple(transforms[day])[:6],
                    tuple(template_transform)[:6],
                ):
                    raise ValueError(f"Raster grid mismatch: {day}")

        daily_records = self._generate_daily_records(
            dates, daily_arrays, template_transform
        )

        daily_path = output_dir / "daily_rainfall.csv"

        self._write_csv(
            daily_path,
            [
                "date", "row", "col", "latitude",
                "longitude", "rainfall_mm",
            ],
            daily_records,
        )

        rolling_files = {}

        for window_days, filename in self.ROLLING_WINDOWS.items():
            records = self._generate_rolling_records(
                dates, daily_arrays, template_transform, window_days
            )

            rolling_path = output_dir / filename

            self._write_csv(
                rolling_path,
                [
                    "date", "period_start", "row", "col",
                    "latitude", "longitude", "rainfall_sum_mm",
                    "valid_days", "window_days", "coverage_fraction",
                ],
                records,
            )

            rolling_files[str(window_days)] = str(rolling_path)

        missing_pixels = sum(
            int(np.count_nonzero(~np.isfinite(daily_arrays[d])))
            for d in dates
        )

        manifest = {
            "dataset": "CHIRPS v3 Daily RNL",
            "version": self.DATASET_VERSION,
            "source": self.BASE_URL,
            "region_id": region_id,
            "bbox": bbox.as_dict(),
            "crs": "EPSG:4326",
            "spatial_resolution": self.RESOLUTION,
            "temporal_resolution": "Daily",
            "units": "mm",
            "start_date": start_date,
            "end_date": end_date,
            "requested_days": len(dates),
            "downloaded_days": downloaded_days,
            "cached_days": cached_days,
            "missing_dates": missing_dates,
            "missing_day_count": len(missing_dates),
            "missing_pixel_count": missing_pixels,
            "total_pixel_records": len(daily_records),
            "minimum_rolling_coverage": self.minimum_coverage,
            "rolling_outputs": rolling_files,
            "daily_output": str(daily_path),
        }

        manifest_path = output_dir / "acquisition_manifest.json"

        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        provenance = self.provenance_tracker.record(
            dataset_name="CHIRPS_V3_Regional_Rainfall",
            source_name="CHC / WFP CHIRPS v3 Daily RNL",
            source_url_or_api=self.BASE_URL,
            region_id=region_id,
            geographic_extent=bbox.as_dict(),
            coordinate_reference_system="EPSG:4326",
            spatial_resolution=self.RESOLUTION,
            temporal_resolution="Daily",
            temporal_coverage={
                "start": start_date,
                "end": end_date,
            },
            total_records_or_pixels=len(daily_records),
            missing_value_count=missing_pixels,
            dataset_version=self.DATASET_VERSION,
            data_quality_flag=(
                "DEGRADED" if missing_pixels > 0
                else "VERIFIED_AUTHENTIC"
            ),
            processing_history=[
                "Remote regional COG acquisition",
                "NoData preservation",
                "Daily rainfall extraction",
                "3-day rainfall aggregation",
                "7-day rainfall aggregation",
                "30-day rainfall aggregation",
            ],
            artifact_path=manifest_path,
            extra_metadata={
                "missing_dates": missing_dates,
                "cached_days": cached_days,
                "downloaded_days": downloaded_days,
            },
        )

        all_cached = (
            cached_days == len(dates) and not missing_dates
        )

        return AcquisitionResult(
            dataset_name="CHIRPS_V3_Regional_Rainfall",
            status=(
                AcquisitionStatus.CACHED
                if all_cached else AcquisitionStatus.SUCCESS
            ),
            local_path=manifest_path,
            records_count=downloaded_days + cached_days,
            missing_count=len(missing_dates),
            spatial_resolution=self.RESOLUTION,
            temporal_resolution="Daily",
            provenance=provenance,
            execution_time_seconds=round(time.time()-start_time, 2),
        )
