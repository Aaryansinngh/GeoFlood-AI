"""
tests/test_map_basemap.py

Tests confirming:
1. Default basemap in build_base_map is 'OpenStreetMap'
2. Generated Folium map uses OpenStreetMap and contains no API-key CartoDB references
3. Study boundary, raster overlay, landmarks, and selected location markers integrate without issue
"""

import folium
import numpy as np
import pytest

from src.ui.map_builder import (
    build_base_map,
    add_study_boundary,
    add_landmarks,
    add_selected_location,
)
from folium.raster_layers import ImageOverlay


def test_default_basemap_is_openstreetmap():
    """Verify default tiles parameter is OpenStreetMap and not CartoDB."""
    bounds = [[26.0, 92.0], [27.0, 94.0]]
    center_lat = 26.5
    center_lon = 93.0

    m = build_base_map(bounds=bounds, center_lat=center_lat, center_lon=center_lon)

    # Check tile layers in map children
    tile_layers = [child for child in m._children.values() if isinstance(child, folium.TileLayer)]
    assert len(tile_layers) >= 1
    default_layer = tile_layers[0]

    # Verify tile name is OpenStreetMap and does not reference Carto
    assert "openstreetmap" in default_layer.tile_name.lower() or "openstreetmap" in str(default_layer._template).lower()
    map_dump = str(m.to_dict())
    assert "cartodb" not in map_dump.lower()
    assert "carto.com" not in map_dump.lower()


def test_map_preserves_overlays_and_markers():
    """Verify raster overlays and markers stack properly on the OpenStreetMap basemap."""
    bounds = [[26.4, 92.8], [26.9, 94.2]]
    m = build_base_map(bounds=bounds, center_lat=26.65, center_lon=93.5)
    m = add_study_boundary(m, bounds, 92.8, 26.4, 94.2, 26.9)

    # Mock RGBA raster (10x10x4)
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[:, :, 0] = 239  # Red high risk
    rgba[:, :, 3] = 200

    ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=0.85,
        name="Test Susceptibility",
    ).add_to(m)

    m = add_landmarks(m)
    m = add_selected_location(
        m, lat=26.65, lon=93.5,
        info={"risk_category": "High Risk", "susceptibility_prob": 0.82}
    )

    children = list(m._children.values())
    types = [type(c).__name__ for c in children]

    assert "TileLayer" in types
    assert "Rectangle" in types
    assert "ImageOverlay" in types
    assert "Marker" in types
    assert "CircleMarker" in types
