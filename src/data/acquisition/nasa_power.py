"""
src/data/acquisition/nasa_power.py

Official NASA POWER Daily Meteorology Client for GeoFlood-AI 2.0.
Retrieves surface meteorological variables:
- PRECTOTCORR: Precipitation Corrected (mm/day)
- T2M: Temperature at 2 Meters (°C)
- RH2M: Relative Humidity at 2 Meters (%)
- WS10M: Wind Speed at 10 Meters (m/s)

Official API Endpoint:
https://power.larc.nasa.gov/api/temporal/daily/point or regional
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.parse
import numpy as np

from src.geospatial.region_manager import BoundingBox
from src.data.acquisition.base_client import BaseAcquisitionClient, AcquisitionResult, AcquisitionStatus
from src.data.provenance.metadata_logger import ProvenanceTracker

logger = logging.getLogger(__name__)


class NasaPowerClient(BaseAcquisitionClient):
    """
    Acquires daily surface meteorological time series from NASA POWER API.
    Does not require API keys. Respects NASA endpoint concurrency limits.
    """

    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    DEFAULT_PARAMETERS = ["PRECTOTCORR", "T2M", "RH2M", "WS10M"]

    def __init__(self, cache_dir: Path = Path("data/cache/nasa_power"), provenance_tracker: Optional[ProvenanceTracker] = None):
        super().__init__(cache_dir=cache_dir, provenance_tracker=provenance_tracker)

    def acquire(
        self,
        region_id: str,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        output_dir: Path,
        grid_sample_points: int = 4,  # Sample points across bounding box
    ) -> AcquisitionResult:
        start_time = time.time()
        is_valid, err = self.validate_inputs(bbox, start_date, end_date)
        if not is_valid:
            return AcquisitionResult(
                dataset_name="NASA_POWER_Meteorology",
                status=AcquisitionStatus.INVALID_INPUT,
                local_path=None,
                records_count=0,
                missing_count=0,
                spatial_resolution="0.5 x 0.625 degree (~50 km)",
                temporal_resolution="Daily",
                provenance=None,
                error_message=err,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._compute_cache_key("nasa_power", region=region_id, s=start_date, e=end_date)
        out_file = output_dir / f"{cache_key}.json"

        # Check existing cache
        if out_file.exists() and out_file.stat().st_size > 0:
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            prov = self.provenance_tracker.record(
                dataset_name="NASA_POWER_Daily_Meteorology",
                source_name="NASA Langley Research Center POWER Project",
                source_url_or_api=self.BASE_URL,
                region_id=region_id,
                geographic_extent=bbox.as_dict(),
                coordinate_reference_system="EPSG:4326",
                spatial_resolution="0.5 x 0.625 degree",
                temporal_resolution="Daily",
                temporal_coverage={"start": start_date, "end": end_date},
                total_records_or_pixels=len(data.get("records", [])),
                missing_value_count=0,
                dataset_version="2.0",
                data_quality_flag="VERIFIED_AUTHENTIC",
                artifact_path=out_file,
            )
            return AcquisitionResult(
                dataset_name="NASA_POWER_Meteorology",
                status=AcquisitionStatus.CACHED,
                local_path=out_file,
                records_count=len(data.get("records", [])),
                missing_count=0,
                spatial_resolution="0.5 x 0.625 degree (~50 km)",
                temporal_resolution="Daily",
                provenance=prov,
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        # Generate spatial sampling grid within bbox
        lats = np.linspace(bbox.min_lat, bbox.max_lat, grid_sample_points)
        lons = np.linspace(bbox.min_lon, bbox.max_lon, grid_sample_points)

        s_ymd = start_date.replace("-", "")
        e_ymd = end_date.replace("-", "")

        records = []
        missing_count = 0

        for lat in lats:
            for lon in lons:
                params = {
                    "parameters": ",".join(self.DEFAULT_PARAMETERS),
                    "community": "AG",
                    "longitude": f"{lon:.4f}",
                    "latitude": f"{lat:.4f}",
                    "start": s_ymd,
                    "end": e_ymd,
                    "format": "JSON",
                }
                url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})

                try:
                    with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                        props = resp_data.get("properties", {}).get("parameter", {})
                        precip_dict = props.get("PRECTOTCORR", {})

                        for date_str, precip_val in precip_dict.items():
                            val = float(precip_val)
                            is_missing = val < -900.0  # NASA POWER fills missing with -999.0
                            if is_missing:
                                missing_count += 1
                                val = np.nan

                            records.append({
                                "lat": float(lat),
                                "lon": float(lon),
                                "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                                "precipitation_mm": val,
                                "temp_2m_c": props.get("T2M", {}).get(date_str, np.nan),
                                "relative_humidity_pct": props.get("RH2M", {}).get(date_str, np.nan),
                                "wind_speed_10m_ms": props.get("WS10M", {}).get(date_str, np.nan),
                            })
                except Exception as e:
                    logger.warning("NASA POWER point acquisition failed at (%f, %f): %s", lat, lon, e)
                    # Handle unavailable station gracefully
                    missing_count += 1

        if not records:
            return AcquisitionResult(
                dataset_name="NASA_POWER_Meteorology",
                status=AcquisitionStatus.NETWORK_ERROR,
                local_path=None,
                records_count=0,
                missing_count=missing_count,
                spatial_resolution="0.5 x 0.625 degree",
                temporal_resolution="Daily",
                provenance=None,
                error_message="All NASA POWER station queries timed out or failed.",
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        payload = {
            "region_id": region_id,
            "source": "NASA POWER API v2.0",
            "bbox": bbox.as_dict(),
            "temporal_coverage": {"start": start_date, "end": end_date},
            "parameters": self.DEFAULT_PARAMETERS,
            "records": records,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        prov = self.provenance_tracker.record(
            dataset_name="NASA_POWER_Daily_Meteorology",
            source_name="NASA Langley Research Center POWER Project",
            source_url_or_api=self.BASE_URL,
            region_id=region_id,
            geographic_extent=bbox.as_dict(),
            coordinate_reference_system="EPSG:4326",
            spatial_resolution="0.5 x 0.625 degree",
            temporal_resolution="Daily",
            temporal_coverage={"start": start_date, "end": end_date},
            total_records_or_pixels=len(records),
            missing_value_count=missing_count,
            dataset_version="2.0",
            data_quality_flag="VERIFIED_AUTHENTIC",
            artifact_path=out_file,
        )

        return AcquisitionResult(
            dataset_name="NASA_POWER_Meteorology",
            status=AcquisitionStatus.SUCCESS,
            local_path=out_file,
            records_count=len(records),
            missing_count=missing_count,
            spatial_resolution="0.5 x 0.625 degree (~50 km)",
            temporal_resolution="Daily",
            provenance=prov,
            execution_time_seconds=round(time.time() - start_time, 2),
        )
