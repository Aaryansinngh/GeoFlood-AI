from pathlib import Path

import geopandas as gpd
import numpy as np
import networkx as nx
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
OSM_DIR = ROOT / "data" / "processed" / "ujjain" / "osm"
ROAD_FILE = OSM_DIR / "ujjain_road_edges.gpkg"
RIVER_FILE = OSM_DIR / "ujjain_shipra_river.geojson"
METRIC_CRS = "EPSG:32643"


def _node_id(value, graph):
    """Resolve GeoPackage IDs against the graph without assuming string/int IDs."""
    if value in graph:
        return value
    text = str(value)
    if text in graph:
        return text
    try:
        number = int(float(text))
        if number in graph:
            return number
        if str(number) in graph:
            return str(number)
    except (TypeError, ValueError):
        pass
    raise KeyError(f"Road endpoint {value!r} is not present in the graph.")


def load_shipra_geometry():
    river = gpd.read_file(RIVER_FILE)
    if river.empty:
        raise ValueError("Shipra River geometry is empty.")
    if river.crs is None:
        raise ValueError("Shipra River geometry has no CRS.")
    return unary_union(river.to_crs(METRIC_CRS).geometry)


def calculate_edge_river_distances(graph):
    """
    Measure actual OSM LineString-to-river distance in metres.
    Return one entry per directed graph edge, including valid reverse edges.
    """
    roads = gpd.read_file(ROAD_FILE)
    if roads.empty:
        raise ValueError("Road GeoPackage is empty.")
    if roads.crs is None:
        raise ValueError("Road GeoPackage has no CRS.")

    roads = roads.to_crs(METRIC_CRS)
    river = load_shipra_geometry()
    roads["river_distance_m"] = roads.geometry.distance(river)

    distances = {}
    physical_roads = {}

    for row in roads.itertuples():
        u = _node_id(row.u, graph)
        v = _node_id(row.v, graph)
        distance = float(row.river_distance_m)

        # Keep a stable physical-road identifier for summary counts.
        physical_id = str(getattr(row, "id", f"{u}-{v}"))
        physical_roads[physical_id] = {
            "u": u,
            "v": v,
            "distance_m": distance,
            "oneway": str(getattr(row, "oneway", "")),
            "road_name": str(getattr(row, "name", "")),
            "highway": str(getattr(row, "highway", "")),
        }

        if graph.has_edge(u, v):
            distances[(u, v)] = distance
        if graph.has_edge(v, u):
            distances[(v, u)] = distance

    missing = set(graph.edges()) - set(distances)
    if missing:
        sample = list(missing)[:5]
        raise ValueError(
            f"Could not map {len(missing)} directed graph edges to physical "
            f"roads. Sample: {sample}"
        )

    return distances, physical_roads


def classify_flood_scenario(
    river_distances,
    terrain_risks,
    flood_spread_m=500,
    vulnerability_threshold=0.65,
    terrain_weight=0.55,
    proximity_weight=0.45,
):
    """Create an illustrative scenario classification, not a flood forecast."""
    if flood_spread_m <= 0:
        raise ValueError("Flood spread must be positive.")
    if not 0 <= vulnerability_threshold <= 1:
        raise ValueError("Vulnerability threshold must be between 0 and 1.")
    if terrain_weight < 0 or proximity_weight < 0:
        raise ValueError("Weights cannot be negative.")
    weight_sum = terrain_weight + proximity_weight
    if weight_sum <= 0:
        raise ValueError("At least one scoring weight must be positive.")

    tw = terrain_weight / weight_sum
    pw = proximity_weight / weight_sum
    results = {}

    for edge, distance in river_distances.items():
        terrain = float(np.clip(terrain_risks.get(edge, 0.0), 0.0, 1.0))
        proximity = float(np.clip(1.0 - distance / flood_spread_m, 0.0, 1.0))
        vulnerability = float(np.clip(tw * terrain + pw * proximity, 0.0, 1.0))
        within_spread = distance <= flood_spread_m
        blocked = bool(within_spread and vulnerability >= vulnerability_threshold)

        if blocked:
            category = "BLOCKED"
        elif vulnerability >= 0.60:
            category = "HIGH"
        elif vulnerability >= 0.35:
            category = "MODERATE"
        else:
            category = "LOW"

        results[edge] = {
            "river_distance_m": round(float(distance), 2),
            "terrain_risk": round(terrain, 4),
            "proximity_risk": round(proximity, 4),
            "vulnerability": round(vulnerability, 4),
            "within_spread": bool(within_spread),
            "blocked": blocked,
            "category": category,
        }

    return results


def apply_flood_closures(graph, scenario, activate=True):
    """Return a graph copy with scenario-blocked directed edges removed."""
    result = graph.copy()
    if not activate:
        return result

    blocked_edges = [
        edge for edge, values in scenario.items() if values["blocked"]
    ]
    for u, v in blocked_edges:
        if result.has_edge(u, v):
            result.remove_edge(u, v)
    return result


def summarize_scenario(results):
    counts = {"BLOCKED": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    for values in results.values():
        counts[values["category"]] += 1
    return {
        "directed_edges": len(results),
        "blocked_directed_edges": counts["BLOCKED"],
        "categories": counts,
    }
