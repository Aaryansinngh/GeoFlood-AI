"""
src/data/acquisition/chirps.py

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) Client.
Source: Climate Hazards Center, University of California, Santa Barbara.
Spatial Resolution: 0.05° (~5.3 km)
Temporal Resolution: Daily
Archive Base URL: https://data.chc.ucsb.edu/products/CHIRPS-2.0/
"""

from datetime import datetime, timedelta
import gzip
import json
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import urllib.request
import numpy as np

from src.geospatial.region_manager import BoundingBox
from src.data.acquisition.base_client import BaseAcquisitionClient, AcquisitionResult, AcquisitionStatus
from src.data.provenance.metadata_logger import ProvenanceTracker

logger = logging.getLogger(__name__)


class ChirpsClient(BaseAcquisitionClient):
    """
    Acquires high-resolution quasi-global daily precipitation from CHIRPS v2.0.
    """

    BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"

    def __init__(self, cache_dir: Path = Path("data/cache/chirps"), provenance_tracker: Optional[ProvenanceTracker] = None):
        super().__init__(cache_dir=cache_dir, provenance_tracker=provenance_tracker)

    def acquire(
        self,
        region_id: str,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        output_dir: Path,
    ) -> AcquisitionResult:
        start_time = time.time()
        is_valid, err = self.validate_inputs(bbox, start_date, end_date)
        if not is_valid:
            return AcquisitionResult(
                dataset_name="CHIRPS_Precipitation",
                status=AcquisitionStatus.INVALID_INPUT,
                local_path=None,
                records_count=0,
                missing_count=0,
                spatial_resolution="0.05 degree (~5 km)",
                temporal_resolution="Daily",
                provenance=None,
                error_message=err,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._compute_cache_key("chirps", region=region_id, s=start_date, e=end_date)
        out_manifest = output_dir / f"{cache_key}.json"

        # Check existing cached run
        if out_manifest.exists() and out_manifest.stat().st_size > 0:
            with open(out_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            prov = self.provenance_tracker.record(
                dataset_name="CHIRPS_Daily_Precipitation",
                source_name="UCSB Climate Hazards Center CHIRPS v2.0",
                source_url_or_api=self.BASE_URL,
                region_id=region_id,
                geographic_extent=bbox.as_dict(),
                coordinate_reference_system="EPSG:4326",
                spatial_resolution="0.05 degree (~5 km)",
                temporal_resolution="Daily",
                temporal_coverage={"start": start_date, "end": end_date},
                total_records_or_pixels=data.get("downloaded_days_count", 0),
                missing_value_count=0,
                dataset_version="2.0",
                data_quality_flag="VERIFIED_AUTHENTIC",
                artifact_path=out_manifest,
            )
            return AcquisitionResult(
                dataset_name="CHIRPS_Precipitation",
                status=AcquisitionStatus.CACHED,
                local_path=out_manifest,
                records_count=data.get("downloaded_days_count", 0),
                missing_count=0,
                spatial_resolution="0.05 degree (~5 km)",
                temporal_resolution="Daily",
                provenance=prov,
                execution_time_seconds=round(time.time() - start_time, 2),
            )

        # Parse date range
        s_dt = datetime.strptime(start_date, "%Y-%m-%d")
        e_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_count = (e_dt - s_dt).days + 1

        downloaded_files = []
        missing_count = 0

        # Construct daily file URLs for CHIRPS
        # Pattern: chirps-v2.0.YYYY.MM.DD.tif.gz
        curr = s_dt
        while curr <= e_dt:
            year_str = curr.strftime("%Y")
            file_name = f"chirps-v2.0.{curr.strftime('%Y.%m.%d')}.tif.gz"
            day_url = f"{self.BASE_URL}/{year_str}/{file_name}"
            target_file = self.cache_dir / file_name

            # Probe or download
            ok, err_msg = self.download_with_retry(day_url, target_file)
            if ok:
                downloaded_files.append(str(target_file))
            else:
                logger.debug("CHIRPS daily granule %s not acquired: %s", file_name, err_msg)
                missing_count += 1
            curr += timedelta(days=1)

        manifest_data = {
            "region_id": region_id,
            "source": "CHIRPS v2.0 Global Daily",
            "bbox": bbox.as_dict(),
            "temporal_coverage": {"start": start_date, "end": end_date},
            "total_requested_days": days_count,
            "downloaded_days_count": len(downloaded_files),
            "missing_days_count": missing_count,
            "files": downloaded_files,
        }

        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        status = AcquisitionStatus.SUCCESS if len(downloaded_files) > 0 else AcquisitionStatus.UNAVAILABLE
        prov = None
        if len(downloaded_files) > 0:
            prov = self.provenance_tracker.record(
                dataset_name="CHIRPS_Daily_Precipitation",
                source_name="UCSB Climate Hazards Center CHIRPS v2.0",
                source_url_or_api=self.BASE_URL,
                region_id=region_id,
                geographic_extent=bbox.as_dict(),
                coordinate_reference_system="EPSG:4326",
                spatial_resolution="0.05 degree (~5 km)",
                temporal_resolution="Daily",
                temporal_coverage={"start": start_date, "end": end_date},
                total_records_or_pixels=len(downloaded_files),
                missing_value_count=missing_count,
                dataset_version="2.0",
                data_quality_flag="VERIFIED_AUTHENTIC",
                artifact_path=out_manifest,
            )

        return AcquisitionResult(
            dataset_name="CHIRPS_Precipitation",
            status=status,
            local_path=out_manifest if len(downloaded_files) > 0 else None,
            records_count=len(downloaded_files),
            missing_count=missing_count,
            spatial_resolution="0.05 degree (~5 km)",
            temporal_resolution="Daily",
            provenance=prov,
            execution_time_seconds=round(time.time() - start_time, 2),
        )
