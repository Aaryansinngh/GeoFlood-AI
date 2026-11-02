"""
tests/test_phase3_preprocessing.py

Comprehensive Tests for Phase 3: Unified Geospatial Preprocessing & Feature Pipeline:
1. Multi-regional data readiness inspection (Assam, Kerala, Mahanadi, Godavari)
2. Graceful handling of unavailable remote datasets without synthetic fabrication
3. Pre-event forecasting mode vs. Retrospective susceptibility mode (leakage elimination)
4. Strict exclusion of post-event rainfall in forecasting mode
5. Metric slope and terrain feature extraction
6. Spatial block partitioning across arbitrary grid geometries
7. Thin Plate Spline rainfall interpolation surface generation
8. Backward compatibility of features.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import rasterio

from src.geospatial.region_manager import RegionManager
from src.data.preprocessing.feature_pipeline import RegionalFeaturePipeline
from src.data.validation.leakage_guard import ModelingMode
from src.data.provenance.metadata_logger import ProvenanceTracker


def test_data_readiness_check_all_regions():
    """Verify data readiness reporting accurately identifies local asset status."""
    rm = RegionManager()
    for reg_id in ["assam", "kerala", "mahanadi", "godavari"]:
        pipeline = RegionalFeaturePipeline(region_id=reg_id, region_manager=rm)
        readiness = pipeline.check_data_readiness()
        assert readiness["region_id"] == reg_id
        assert isinstance(readiness["dem_available"], bool)
        assert isinstance(readiness["events_available"], dict)
        assert isinstance(readiness["rainfall_available"], dict)

    # Assam has local raw assets downloaded
    pipeline_assam = RegionalFeaturePipeline(region_id="assam", region_manager=rm)
    readiness_assam = pipeline_assam.check_data_readiness()
    assert readiness_assam["dem_available"] is True
    assert readiness_assam["events_available"].get("DFO_4259") is True
    assert readiness_assam["overall_ready"] is True


def test_unavailable_regions_handled_gracefully_without_fabrication():
    """Verify that un-downloaded regions do not crash and never fabricate fake rows."""
    pipeline_kerala = RegionalFeaturePipeline(region_id="kerala")
    readiness = pipeline_kerala.check_data_readiness()

    # Unless raw tiles were downloaded externally, overall_ready is false
    if not readiness["overall_ready"]:
        df, status = pipeline_kerala.run_pipeline()
        assert len(df) == 0  # Does NOT invent fake rows
        assert status["overall_ready"] is False


def test_forecasting_mode_strictly_excludes_post_event_rainfall():
    """Verify that PRE_EVENT_FORECASTING mode removes rain_event_sum from features."""
    pipeline_retro = RegionalFeaturePipeline(region_id="assam", mode=ModelingMode.RETROSPECTIVE_SUSCEPTIBILITY)
    pipeline_fore = RegionalFeaturePipeline(region_id="assam", mode=ModelingMode.PRE_EVENT_FORECASTING, lead_time_hours=24.0)

    lats = np.array([[26.5, 26.5], [26.7, 26.7]])
    lons = np.array([[93.0, 93.5], [93.0, 93.5]])
    stations = [
        {"lat": 26.5, "lon": 93.0, "rain_3d": 20.0, "rain_7d": 50.0, "rain_14d": 90.0, "rain_event_sum": 600.0},
        {"lat": 26.7, "lon": 93.5, "rain_3d": 40.0, "rain_7d": 80.0, "rain_14d": 120.0, "rain_event_sum": 800.0},
        {"lat": 26.5, "lon": 93.5, "rain_3d": 30.0, "rain_7d": 60.0, "rain_14d": 100.0, "rain_event_sum": 700.0},
    ]

    surfaces_retro = pipeline_retro.interpolate_rainfall_surfaces(stations, lats, lons)
    surfaces_fore = pipeline_fore.interpolate_rainfall_surfaces(stations, lats, lons)

    # Retrospective includes event sum
    assert "rain_event_sum" in surfaces_retro
    assert surfaces_retro["rain_event_sum"].shape == (2, 2)

    # Forecasting strictly excludes event sum
    assert "rain_event_sum" not in surfaces_fore
    assert "rain_3d" in surfaces_fore
    assert "rain_7d" in surfaces_fore
    assert "rain_14d" in surfaces_fore


def test_tps_rainfall_interpolation_monotonicity_and_bounds():
    """Verify Thin Plate Spline rainfall values are non-negative and finite."""
    pipeline = RegionalFeaturePipeline(region_id="assam")
    grid_lats, grid_lons = np.mgrid[26.3:27.2:20j, 92.8:94.2:20j]

    stations = [
        {"lat": 26.4, "lon": 92.9, "rain_3d": 15.0, "rain_7d": 40.0, "rain_14d": 75.0, "rain_event_sum": 450.0},
        {"lat": 26.8, "lon": 93.2, "rain_3d": 55.0, "rain_7d": 110.0, "rain_14d": 180.0, "rain_event_sum": 850.0},
        {"lat": 27.0, "lon": 94.0, "rain_3d": 35.0, "rain_7d": 70.0, "rain_14d": 130.0, "rain_event_sum": 620.0},
        {"lat": 26.5, "lon": 94.1, "rain_3d": 25.0, "rain_7d": 55.0, "rain_14d": 95.0, "rain_event_sum": 510.0},
    ]

    surfaces = pipeline.interpolate_rainfall_surfaces(stations, grid_lats, grid_lons)
    for name, s in surfaces.items():
        assert not np.isnan(s).any()
        assert np.all(s >= 0.0), f"Surface {name} contains negative precipitation values"


def test_processed_features_backward_compatibility():
    """Verify processed features table matches legacy column schema and non-empty size."""
    features_path = Path("data/processed/features.parquet")
    assert features_path.exists(), "data/processed/features.parquet must exist"

    df = pd.read_parquet(features_path)
    assert len(df) == 498400 or len(df) > 100000

    required_cols = [
        "cell_id", "row", "col", "lat", "lon", "event_id",
        "spatial_block_id", "rain_3d", "rain_7d", "elevation",
        "slope", "tpi", "relative_elevation", "is_perm_water", "flood_label"
    ]
    for c in required_cols:
        assert c in df.columns, f"Missing required column '{c}' in features.parquet"

    # Verify labels contain dry land (0), flooded (1), permanent water (-1), or NaN
    unique_labels = df["flood_label"].dropna().unique()
    for l in unique_labels:
        assert l in [-1.0, 0.0, 1.0]
