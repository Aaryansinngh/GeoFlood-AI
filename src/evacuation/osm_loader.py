"""
GeoFlood AI - OpenStreetMap Road Network Loader

Downloads real road networks using OSMnx and converts
them into the custom GeoFlood AI RoadGraph.

OpenStreetMap supplies road geometry and attributes.
Flood-risk information is supplied separately.
"""

import math

from typing import Tuple

import osmnx as ox

from src.evacuation.road_graph import RoadEdge, RoadGraph


def load_osm_road_network(
    latitude: float,
    longitude: float,
    radius_m: int = 2000,
    network_type: str = "drive",
    default_speed_kmph: float = 30.0,
    default_flood_risk: float = 0.5,
) -> Tuple[RoadGraph, object]:
    """
    Download and convert an OSM road network.

    Parallel road segments are retained independently.
    The default flood risk is a placeholder, not an observation.
    """

    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude")

    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude")

    if radius_m <= 0:
        raise ValueError("Radius must be positive")

    if (
        not math.isfinite(default_speed_kmph)
        or default_speed_kmph <= 0
    ):
        raise ValueError("Default speed must be positive")

    if (
        not math.isfinite(default_flood_risk)
        or not 0 <= default_flood_risk <= 1
    ):
        raise ValueError("Flood risk must be between 0 and 1")

    osm_graph = ox.graph.graph_from_point(
        (latitude, longitude),
        dist=radius_m,
        network_type=network_type,
        simplify=True,
    )

    if len(osm_graph.nodes) == 0:
        raise ValueError("OpenStreetMap returned an empty road network")

    osm_graph = ox.routing.add_edge_speeds(
        osm_graph,
        fallback=default_speed_kmph,
    )

    osm_graph = ox.routing.add_edge_travel_times(osm_graph)

    road_graph = RoadGraph()

    for node in osm_graph.nodes:
        road_graph.add_node(node)

    for source, target, key, data in osm_graph.edges(
        keys=True,
        data=True,
    ):
        distance_m = float(data.get("length", 0))

        if not math.isfinite(distance_m) or distance_m <= 0:
            continue

        distance_km = distance_m / 1000.0

        speed_kmph = float(
            data.get("speed_kph", default_speed_kmph)
        )

        if not math.isfinite(speed_kmph) or speed_kmph <= 0:
            speed_kmph = default_speed_kmph

        road_name = data.get("name")

        if isinstance(road_name, list):
            road_name = ", ".join(str(name) for name in road_name)

        edge = RoadEdge(
            source=source,
            target=target,
            distance_km=distance_km,
            speed_kmph=speed_kmph,
            flood_risk=default_flood_risk,
            accessibility_penalty=0.0,
            is_closed=False,
            is_flooded=False,
            road_name=str(road_name) if road_name else None,
            segment_id=key,
        )

        road_graph.add_edge(edge)

    if road_graph.get_edge_count() == 0:
        raise ValueError("No usable road segments were found")

    return road_graph, osm_graph
