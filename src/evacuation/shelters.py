"""Candidate evacuation shelter discovery using OpenStreetMap."""

import osmnx as ox
import geopandas as gpd
from pathlib import Path

CACHE_DIR = Path("data/cache/evacuation")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_shelters(
    latitude=26.1445,
    longitude=91.7362,
    radius_m=5000,
):
    """Retrieve candidate shelter facilities from OSM.

    These are candidates only, not officially designated shelters.
    """

    cache_path = CACHE_DIR / "guwahati_candidate_shelters.geojson"

    if cache_path.exists():
        return gpd.read_file(cache_path)

    tags = {
        "amenity": [
            "school",
            "hospital",
            "community_centre",
            "place_of_worship",
        ],
        "emergency": "assembly_point",
    }

    try:
        places = ox.features_from_point(
            (latitude, longitude),
            tags=tags,
            dist=radius_m,
        )
    except Exception:
        return gpd.GeoDataFrame(
            columns=["name", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    if places.empty:
        return places

    places = places.reset_index()

    if "name" not in places.columns:
        places["name"] = "Unnamed candidate facility"

    places = places[
        places.geometry.notna()
    ].copy()

    places = places[
        places.geometry.geom_type.isin(
            ["Point", "Polygon", "MultiPolygon"]
        )
    ].copy()

    places["geometry"] = places.geometry.representative_point()

    places = places.to_crs("EPSG:32646")
    places.to_file(cache_path, driver="GeoJSON")

    return places
