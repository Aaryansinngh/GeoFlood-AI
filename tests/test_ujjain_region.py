from src.geospatial.region_manager import RegionManager


def test_ujjain_is_registered():
    manager = RegionManager()
    regions = manager.list_regions()

    assert "ujjain" in regions


def test_ujjain_metadata():
    manager = RegionManager()
    region = manager.get_region("ujjain")

    assert region.name == "Ujjain_Shipra"
    assert region.river_basin == "Shipra"
    assert region.projected_metric_crs == "EPSG:32643"


def test_ujjain_bounding_box():
    manager = RegionManager()
    region = manager.get_region("ujjain")
    bbox = region.bbox.as_dict()

    assert bbox["min_lat"] == 23.12
    assert bbox["max_lat"] == 23.25
    assert bbox["min_lon"] == 75.68
    assert bbox["max_lon"] == 75.85


def test_ujjain_coordinate_validation():
    manager = RegionManager()

    valid, error = manager.validate_coordinates(
        "ujjain", 23.1828, 75.7683
    )

    assert valid is True
    assert error is None


def test_ujjain_rejects_outside_coordinates():
    manager = RegionManager()

    valid, error = manager.validate_coordinates(
        "ujjain", 26.0, 75.0
    )

    assert valid is False


def test_ujjain_srtm_tile():
    manager = RegionManager()
    region = manager.get_region("ujjain")

    tiles = RegionManager.get_srtm_tiles_for_bbox(region.bbox)

    assert "N23E075" in tiles
