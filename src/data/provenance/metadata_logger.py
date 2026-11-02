"""
src/data/provenance/metadata_logger.py

Scientific Data Provenance and Quality Tracking for GeoFlood-AI 2.0.
Records full metadata for every acquired dataset:
- Data source & version
- Acquisition timestamp
- Geographic bounding extent & CRS
- Spatial & temporal resolutions
- Missing-value percentage & imputation logs
- Processing history / pipeline transformations
- Verification hash (SHA-256) of raw/processed artifacts
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DatasetProvenanceRecord:
    dataset_name: str
    source_name: str
    source_url_or_api: str
    dataset_version: str
    acquisition_timestamp: str
    region_id: str
    geographic_extent: Dict[str, float]
    coordinate_reference_system: str
    spatial_resolution: str
    temporal_resolution: str
    temporal_coverage: Dict[str, str]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    total_records_or_pixels: int
    missing_value_count: int
    missing_value_percentage: float
    data_quality_flag: str  # "VERIFIED_AUTHENTIC", "DEGRADED", "SYNTHETIC_BENCHMARK", "UNAVAILABLE"
    processing_history: List[str] = field(default_factory=list)
    sha256_checksum: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProvenanceTracker:
    """
    Manages and persists provenance records to disk (JSON/CSV) for auditability.
    """

    def __init__(self, output_dir: Path = Path("reports/provenance")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[DatasetProvenanceRecord] = []

    def record(
        self,
        dataset_name: str,
        source_name: str,
        source_url_or_api: str,
        region_id: str,
        geographic_extent: Dict[str, float],
        coordinate_reference_system: str,
        spatial_resolution: str,
        temporal_resolution: str,
        temporal_coverage: Dict[str, str],
        total_records_or_pixels: int,
        missing_value_count: int,
        dataset_version: str = "1.0",
        data_quality_flag: str = "VERIFIED_AUTHENTIC",
        processing_history: Optional[List[str]] = None,
        artifact_path: Optional[Path] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetProvenanceRecord:
        """Create and store a provenance record."""
        pct_missing = (
            round((missing_value_count / max(1, total_records_or_pixels)) * 100.0, 3)
            if total_records_or_pixels > 0
            else 0.0
        )

        checksum = None
        if artifact_path and Path(artifact_path).exists() and Path(artifact_path).is_file():
            hasher = hashlib.sha256()
            with open(artifact_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            checksum = hasher.hexdigest()

        rec = DatasetProvenanceRecord(
            dataset_name=dataset_name,
            source_name=source_name,
            source_url_or_api=source_url_or_api,
            dataset_version=dataset_version,
            acquisition_timestamp=datetime.now(timezone.utc).isoformat(),
            region_id=region_id,
            geographic_extent=geographic_extent,
            coordinate_reference_system=coordinate_reference_system,
            spatial_resolution=spatial_resolution,
            temporal_resolution=temporal_resolution,
            temporal_coverage=temporal_coverage,
            total_records_or_pixels=total_records_or_pixels,
            missing_value_count=missing_value_count,
            missing_value_percentage=pct_missing,
            data_quality_flag=data_quality_flag,
            processing_history=processing_history or ["Raw data ingested"],
            sha256_checksum=checksum,
            extra_metadata=extra_metadata or {},
        )
        self._records.append(rec)
        self.save_record(rec)
        return rec

    def save_record(self, record: DatasetProvenanceRecord) -> Path:
        """Save a single record as JSON in output_dir."""
        safe_name = f"{record.region_id}_{record.dataset_name}".replace(" ", "_").lower()
        out_json = self.output_dir / f"provenance_{safe_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
        logger.info("Saved dataset provenance record to %s", out_json)
        return out_json

    def export_summary(self) -> Path:
        """Export all recorded datasets into an aggregated summary JSON."""
        summary_path = self.output_dir / "provenance_catalog.json"
        data = [r.to_dict() for r in self._records]
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return summary_path
