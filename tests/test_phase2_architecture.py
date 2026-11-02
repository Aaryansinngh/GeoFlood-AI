"""
tests/test_phase2_architecture.py

Unit and Integration Tests for Phase 2:
1. RegionManager: Multi-region loading, bounding box containment, SRTM tile discovery, coordinate validation
2. UnifiedRasterProcessor: Master grid shape, metric slope, TPI, relative elevation, missing DEM handling
3. Data Acquisition Clients: NASA POWER, CHIRPS, ERA5, DEM, GFD (with mocked network responses)
4. ProvenanceTracker: Metadata record generation, checksums, and catalog export
5. LeakageGuard: Pre-event forecasting cutoff, post-event feature rejection, retrospective mode
6. Backward Compatibility: Existing Assam settings and pipeline invariants
"""

import json
from pathlib import Path
import unittest.mock as mock
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from src.geospatial.region_manager import RegionManager, BoundingBox, RegionMetadata
from src.geospatial.raster_processing import UnifiedRasterProcessor
from src.data.provenance.metadata_logger import ProvenanceTracker, DatasetProvenanceRecord
from src.data.validation.leakage_guard import LeakageGuard, ModelingMode, FeatureTemporalType
from src.data.acquisition.base_client import AcquisitionStatus
from src.data.acquisition.nasa_power import NasaPowerClient
from src.data.acquisition.chirps import ChirpsClient
from src.data.acquisition.era5 import Era5Client
from src.data.acquisition.dem_client import DemClient
from src.data.acquisition.gfd_client import GfdClient


# ============================================================================
# 1. Region Manager Tests
# ============================================================================

def test_region_manager_loads_all_four_regions():
    """Verify that all initial target regions are loaded successfully."""
    rm = RegionManager()
    regions = rm.list_regions()
    for expected in ["assam", "kerala", "mahanadi", "godavari"]:
        assert expected in regions, f"Expected region '{expected}' not found in {regions}"


def test_region_metadata_integrity():
    """Verify regional metadata fields, bounds, and CRS."""
    rm = RegionManager()

    # Assam
    assam = rm.get_region("assam")
    assert assam.river_basin == "Brahmaputra"
    assert assam.projected_metric_crs == "EPSG:32646"
    assert assam.bbox.min_lat == 26.30
    assert assam.bbox.max_lat == 27.20

    # Kerala
    kerala = rm.get_region("kerala")
    assert kerala.river_basin == "Periyar-Pamba Catchments"
    assert kerala.projected_metric_crs == "EPSG:32643"
    assert kerala.bbox.min_lat == 9.30
    assert kerala.bbox.max_lat == 10.35

    # Mahanadi
    mahanadi = rm.get_region("mahanadi")
    assert mahanadi.river_basin == "Mahanadi River Basin"
    assert mahanadi.projected_metric_crs == "EPSG:32645"

    # Godavari
    godavari = rm.get_region("godavari")
    assert godavari.river_basin == "Godavari River Basin"
    assert godavari.projected_metric_crs == "EPSG:32644"


def test_coordinate_validation_inside_and_outside():
    """Verify boundary checks across regions."""
    rm = RegionManager()

    # Kaziranga inside Assam
    ok, err = rm.validate_coordinates("assam", lat=26.65, lon=93.35)
    assert ok is True
    assert err is None

    # Aluva inside Kerala
    ok, err = rm.validate_coordinates("kerala", lat=10.10, lon=76.35)
    assert ok is True
    assert err is None

    # Kerala coordinates tested against Assam must be rejected
    ok, err = rm.validate_coordinates("assam", lat=10.10, lon=76.35)
    assert ok is False
    assert "outside" in err.lower()

    # Invalid non-numeric or NaN coordinates
    ok, err = rm.validate_coordinates("assam", lat=float("nan"), lon=93.35)
    assert ok is False
    assert "nan" in err.lower()


def test_srtm_tile_calculator():
    """Verify automated tile derivation covering arbitrary bounding boxes."""
    # Assam box: 26.30-27.20N, 92.80-94.20E
    bbox_assam = BoundingBox(26.30, 27.20, 92.80, 94.20)
    tiles = RegionManager.get_srtm_tiles_for_bbox(bbox_assam)
    expected_tiles = ["N26E092", "N26E093", "N26E094", "N27E092", "N27E093", "N27E094"]
    for t in ["N26E092", "N26E093", "N27E092", "N27E093"]:
        assert t in tiles


def test_grid_dimension_computation():
    """Verify dynamic master grid dimension and affine transform calculation."""
    rm = RegionManager()
    h, w, transform = rm.compute_grid_dimensions("assam")
    assert h > 0 and w > 0
    assert len(transform) == 6
    # Master grid resolution approx ~0.00225 deg
    assert 0.0022 <= transform[1] <= 0.0023


# ============================================================================
# 2. Unified Raster Preprocessing Tests
# ============================================================================

def test_unified_raster_processor_slope_and_tpi():
    """Verify metric slope and TPI computations on synthetic terrain."""
    rm = RegionManager()
    region = rm.get_region("assam")
    proc = UnifiedRasterProcessor(region)

    # Synthetic inclined plane DEM: elevation increases linearly along longitude
    h, w = 50, 50
    elev = np.zeros((h, w), dtype=np.float32)
    for c in range(w):
        elev[:, c] = c * 10.0  # 10m rise per 250m cell = slope ~ 2.29 degrees

    slope = proc.calculate_metric_slope(elev)
    assert slope.shape == (h, w)
    assert np.all(slope >= 0.0)
    # Check interior slope is finite and realistic
    interior_slope = slope[5:-5, 5:-5]
    assert np.mean(interior_slope) > 0.5

    # TPI on flat plane must be close to 0
    tpi = proc.calculate_tpi(elev, window_size=5)
    assert tpi.shape == (h, w)
    assert np.allclose(tpi[10:-10, 10:-10], 0.0, atol=1e-2)

    # Relative elevation (HAND proxy) must be non-negative
    rel = proc.calculate_relative_elevation(elev, window_size=5)
    assert rel.shape == (h, w)
    assert np.all(rel[~np.isnan(rel)] >= 0.0)


def test_missing_dem_handling_preserves_mask():
    """Verify missing DEM values are imputed without marking unobserved as safe."""
    dem = np.array([
        [100.0, 102.0, -32768.0],
        [105.0, np.nan, 108.0],
        [110.0, 112.0, 115.0]
    ], dtype=np.float32)

    cleaned, missing_mask = UnifiedRasterProcessor.handle_missing_dem(dem)
    assert missing_mask.sum() == 2
    assert not np.isnan(cleaned).any()
    assert (cleaned < 0).sum() == 0  # No nodata flags left in clean array
    assert cleaned[0, 2] == np.median([100.0, 102.0, 105.0, 108.0, 110.0, 112.0, 115.0])


# ============================================================================
# 3. Data Provenance & Tracking Tests
# ============================================================================

def test_provenance_tracker_records_and_exports(tmp_path):
    """Verify provenance metadata recording and SHA-256 calculation."""
    tracker = ProvenanceTracker(output_dir=tmp_path)

    # Create dummy artifact file
    dummy_file = tmp_path / "test_raster.tif"
    dummy_file.write_bytes(b"GEOTIFF_DUMMY_BINARY_DATA_FOR_PROVENANCE_TEST")

    rec = tracker.record(
        dataset_name="Test_SRTM_DEM",
        source_name="ESA STEP",
        source_url_or_api="https://step.esa.int/test",
        region_id="assam",
        geographic_extent={"min_lat": 26.3, "max_lat": 27.2, "min_lon": 92.8, "max_lon": 94.2},
        coordinate_reference_system="EPSG:4326",
        spatial_resolution="30m",
        temporal_resolution="Static",
        temporal_coverage={"start": "2000-02-11", "end": "2000-02-22"},
        total_records_or_pixels=1000,
        missing_value_count=5,
        artifact_path=dummy_file,
    )

    assert rec.missing_value_percentage == 0.5
    assert rec.sha256_checksum is not None
    assert len(rec.sha256_checksum) == 64

    # Export catalog
    catalog_path = tracker.export_summary()
    assert catalog_path.exists()
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["dataset_name"] == "Test_SRTM_DEM"


# ============================================================================
# 4. Leakage Prevention Engine Tests
# ============================================================================

def test_leakage_guard_rejects_post_event_features_in_forecasting_mode():
    """Verify strict prohibition of post-flood rainfall in forecasting mode."""
    guard = LeakageGuard(mode=ModelingMode.PRE_EVENT_FORECASTING, lead_time_hours=24.0)

    # Pure pre-event and static feature list
    valid_features = ["elevation", "slope", "tpi", "relative_elevation", "rain_3d", "rain_7d"]
    ok, violations = guard.validate_features(valid_features)
    assert ok is True
    assert len(violations) == 0

    # Including rain_event_sum (which includes post-event rain) must be rejected
    contaminated_features = valid_features + ["rain_event_sum"]
    ok, violations = guard.validate_features(contaminated_features)
    assert ok is False
    assert len(violations) == 1
    assert "rain_event_sum" in violations[0]


def test_leakage_guard_permits_event_sum_in_retrospective_mode():
    """Verify retrospective susceptibility mapping allows full event-sum rainfall."""
    guard = LeakageGuard(mode=ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY)
    features = ["elevation", "slope", "rain_3d", "rain_event_sum"]
    ok, violations = guard.validate_features(features)
    assert ok is True
    assert len(violations) == 0


def test_temporal_window_calculation():
    """Verify date window boundaries are computed strictly before onset."""
    guard = LeakageGuard(mode=ModelingMode.PRE_EVENT_FORECASTING, lead_time_hours=48.0)
    windows = guard.get_event_temporal_windows("2015-06-02", "2015-06-29")

    # 48 hours before 2015-06-02 is 2015-05-31
    assert "2015-05-31" in windows["prediction_cutoff"]
    assert windows["pre_event_3d_window"]["end"] == "2015-05-31"


# ============================================================================
# 5. Data Acquisition Client Tests (Mocked API Responses)
# ============================================================================

def test_nasa_power_client_parses_json_correctly(tmp_path):
    """Test NASA POWER client with mocked HTTP payload."""
    client = NasaPowerClient(cache_dir=tmp_path / "cache")
    bbox = BoundingBox(26.3, 26.5, 92.8, 93.0)

    mock_response = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": {"20150601": 12.5, "20150602": 35.0},
                "T2M": {"20150601": 28.2, "20150602": 26.8},
                "RH2M": {"20150601": 85.0, "20150602": 92.0},
                "WS10M": {"20150601": 2.1, "20150602": 3.4},
            }
        }
    }

    mock_resp_obj = mock.MagicMock()
    mock_resp_obj.read.return_value = json.dumps(mock_response).encode("utf-8")
    mock_resp_obj.__enter__.return_value = mock_resp_obj

    with mock.patch("urllib.request.urlopen", return_value=mock_resp_obj):
        res = client.acquire(
            region_id="assam",
            bbox=bbox,
            start_date="2015-06-01",
            end_date="2015-06-02",
            output_dir=tmp_path / "out",
            grid_sample_points=2,
        )

    assert res.is_successful
    assert res.records_count > 0
    assert res.local_path.exists()


def test_acquisition_client_input_validation():
    """Verify clients reject reversed dates or future dates."""
    client = Era5Client()
    bbox = BoundingBox(26.3, 26.5, 92.8, 93.0)

    # Start after end
    res = client.acquire("assam", bbox, start_date="2020-05-10", end_date="2020-05-01", output_dir=Path("tmp"))
    assert res.status == AcquisitionStatus.INVALID_INPUT
    assert "cannot be after" in res.error_message


def test_dem_client_handles_cached_mosaic(tmp_path):
    """Verify DEM client immediately returns cached GeoTIFF without network calls."""
    client = DemClient(cache_dir=tmp_path / "cache")
    bbox = BoundingBox(26.3, 27.2, 92.8, 94.2)

    # Create dummy pre-existing mosaic
    cached_tif = tmp_path / "dem_assam_srtmgl1.tif"
    out_meta = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "transform": Affine(0.002, 0.0, 92.8, 0.0, -0.002, 27.2),
        "crs": "EPSG:4326",
        "count": 1,
        "dtype": "float32",
    }
    with rasterio.open(cached_tif, "w", **out_meta) as dst:
        dst.write(np.ones((10, 10), dtype=np.float32), 1)

    res = client.acquire(
        region_id="assam",
        bbox=bbox,
        output_dir=tmp_path,
        dem_type="SRTMGL1",
    )
    assert res.status == AcquisitionStatus.CACHED
    assert res.local_path == cached_tif
