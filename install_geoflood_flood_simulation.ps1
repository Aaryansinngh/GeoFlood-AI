$ErrorActionPreference = "Stop"

$Project = Join-Path $HOME ".gemini\antigravity\scratch\GeoFlood-AI"
Set-Location $Project

$AppPath = Join-Path $Project "evacuation_app.py"
$ModulePath = Join-Path $Project "src\evacuation\flood_simulator.py"

if (-not (Test-Path $AppPath)) {
    throw "evacuation_app.py was not found. No files were changed."
}
if (-not (Test-Path (Join-Path $Project "data\processed\ujjain\osm\ujjain_shipra_river.geojson"))) {
    throw "Shipra GeoJSON was not found. Run prepare_shipra.py first."
}
if (-not (Test-Path (Join-Path $Project "data\processed\ujjain\osm\ujjain_road_edges.gpkg"))) {
    throw "Road GeoPackage was not found. No files were changed."
}

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Copy-Item $AppPath "$AppPath.backup-$Stamp" -Force
if (Test-Path $ModulePath) {
    Copy-Item $ModulePath "$ModulePath.backup-$Stamp" -Force
}

$Module = @'
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
'@

$App = @'
from __future__ import annotations

import json
import math
from pathlib import Path

import folium
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import streamlit as st
from scipy.spatial import cKDTree
from streamlit_folium import st_folium

from src.evacuation.flood_simulator import (
    calculate_edge_river_distances,
    classify_flood_scenario,
    apply_flood_closures,
    summarize_scenario,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "ujjain"
OSM_DIR = DATA / "osm"
GRAPH_FILE = OSM_DIR / "ujjain_directed_roads.graphml"
RIVER_FILE = OSM_DIR / "ujjain_shipra_river.geojson"

CENTER = (23.1828, 75.7683)
LANDMARKS = {
    "Ram Ghat (study reference)": (75.7683, 23.1828),
    "Ujjain Junction (study reference)": (75.7850, 23.1785),
    "Mahakaleshwar area (study reference)": (75.7685, 23.1828),
}
DEMO_DESTINATIONS = {
    "Demo evacuation point A (simulated)": (75.7920, 23.2020),
    "Demo evacuation point B (simulated)": (75.7900, 23.1630),
    "Demo evacuation point C (simulated)": (75.7350, 23.1950),
}

st.set_page_config(
    page_title="GeoFlood AI | Adaptive Evacuation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top:1.1rem;padding-bottom:2rem}
.hero {padding:22px 25px;border-radius:14px;background:linear-gradient(115deg,#10243a,#174c62);color:#fff;margin-bottom:15px}
.hero h1 {margin:0 0 5px;font-size:2rem}
.hero p {margin:0;color:#d8e8ef}
[data-testid="stMetric"] {background:rgba(50,70,90,.12);padding:12px;border-radius:10px}
</style>
<div class="hero">
<h1>GEOFLOOD AI</h1>
<p>Adaptive flood-aware evacuation intelligence · Ujjain / Shipra River</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "ACADEMIC PROTOTYPE — This is a hypothetical scenario simulator, not a "
    "live flood forecast or operational emergency service. Flood spread and "
    "road closures are illustrative and must not be used for real evacuation."
)


@st.cache_resource(show_spinner="Loading directed Ujjain road graph…")
def load_graph():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Directed road graph not found: {GRAPH_FILE}")
    graph = nx.read_graphml(GRAPH_FILE)
    graph = nx.relabel_nodes(graph, {n: int(n) for n in graph.nodes})
    for _, _, data in graph.edges(data=True):
        for key in ("length_m", "speed_kph", "travel_s"):
            data[key] = float(data[key])
    for _, data in graph.nodes(data=True):
        data["x"] = float(data["x"])
        data["y"] = float(data["y"])
    return graph


@st.cache_resource(show_spinner=False)
def load_rasters():
    files = {
        "relative": DATA / "ujjain_relative_elevation.tif",
        "tpi": DATA / "ujjain_tpi.tif",
        "slope": DATA / "ujjain_slope.tif",
        "elevation": DATA / "ujjain_elevation.tif",
    }
    rasters = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Terrain raster missing: {path}")
        rasters[key] = rasterio.open(path)
    return rasters


@st.cache_data(show_spinner="Sampling terrain along road network…")
def prepare_edge_terrain(edge_records):
    rasters = load_rasters()
    rel_ds = rasters["relative"]
    tpi_ds = rasters["tpi"]
    slope_ds = rasters["slope"]
    coords = [(record[2], record[3]) for record in edge_records]

    rel_values = np.array([value[0] for value in rel_ds.sample(coords)], dtype=float)
    tpi_values = np.array([value[0] for value in tpi_ds.sample(coords)], dtype=float)
    slope_values = np.array([value[0] for value in slope_ds.sample(coords)], dtype=float)

    def normalize(values, invert=False):
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            output = np.full(len(values), 0.5)
        else:
            low, high = np.nanpercentile(valid, [5, 95])
            if high <= low:
                output = np.full(len(values), 0.5)
            else:
                output = np.clip((values - low) / (high - low), 0, 1)
        return 1 - output if invert else output

    relative_score = normalize(rel_values, invert=True)
    tpi_score = normalize(tpi_values, invert=True)
    slope_score = normalize(slope_values)
    index = np.clip(
        0.65 * relative_score + 0.25 * tpi_score + 0.10 * slope_score,
        0, 1,
    )
    return index.astype(float).tolist()


@st.cache_data(show_spinner="Measuring road proximity to Shipra River…")
def cached_river_distances(graph_edge_keys):
    # The graph keys are supplied as primitive tuples for Streamlit caching.
    graph = load_graph()
    distances, _physical_roads = calculate_edge_river_distances(graph)
    return [(int(u), int(v), float(distance)) for (u, v), distance in distances.items()]


def edge_records_from_graph(graph):
    records = []
    for u, v, data in graph.edges(data=True):
        lon = (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2
        lat = (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2
        records.append((u, v, lon, lat))
    return records


def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))


def nearest_node(graph, tree, node_ids, point):
    lon, lat = point
    _, index = tree.query([lon, lat], k=1)
    return node_ids[int(index)]


def route_stats(graph, path):
    if len(path) < 2:
        return {"distance_km": 0.0, "minutes": 0.0, "risk": 0.0}
    edges = [graph[u][v] for u, v in zip(path[:-1], path[1:])]
    distance = sum(float(edge["length_m"]) for edge in edges)
    seconds = sum(float(edge["travel_s"]) for edge in edges)
    weighted = sum(
        float(edge["length_m"]) * float(edge.get("risk", 0.0))
        for edge in edges
    ) / max(distance, 1e-9)
    return {
        "distance_km": distance / 1000,
        "minutes": seconds / 60,
        "risk": weighted,
    }


def path_coordinates(graph, path):
    coordinates = []
    for u, v in zip(path[:-1], path[1:]):
        x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        if not coordinates:
            coordinates.append((y1, x1))
        coordinates.append((y2, x2))
    return coordinates


def build_map(graph, baseline, safer, start, destination, scenario,
              flood_active, flood_spread_m):
    map_object = folium.Map(
        location=CENTER,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Draw the river as a separate selectable map layer.
    if RIVER_FILE.exists():
        river = json.loads(RIVER_FILE.read_text(encoding="utf-8"))
        folium.GeoJson(
            river,
            name="Shipra River",
            style_function=lambda feature: {
                "color": "#0877c9",
                "weight": 5,
                "opacity": 0.95,
            },
            tooltip="Shipra River — OSM geometry",
        ).add_to(map_object)

    major = {
        "primary", "secondary", "tertiary", "trunk",
        "primary_link", "trunk_link",
    }
    shown = 0
    blocked_drawn = 0
    seen_physical = set()

    for u, v, data in graph.edges(data=True):
        if data.get("highway", "") not in major:
            continue

        # Avoid drawing the same two-way physical segment twice.
        pair = (min(u, v), max(u, v))
        if pair in seen_physical:
            continue
        seen_physical.add(pair)

        values = scenario.get((u, v), {})
        vulnerability = float(values.get("vulnerability", data.get("risk", 0.0)))
        blocked = bool(flood_active and values.get("blocked", False))

        if blocked:
            color = "#d7191c"
            blocked_drawn += 1
        elif vulnerability >= 0.60:
            color = "#f28e2b"
        elif vulnerability >= 0.35:
            color = "#f2c744"
        else:
            color = "#2ca25f"

        folium.PolyLine(
            [
                (graph.nodes[u]["y"], graph.nodes[u]["x"]),
                (graph.nodes[v]["y"], graph.nodes[v]["x"]),
            ],
            color=color,
            weight=3 if blocked else 2,
            opacity=0.85 if blocked else 0.55,
            tooltip=(
                f'{data.get("name", "Unnamed road")} | '
                f'{vulnerability:.2f} vulnerability'
                + (" | SIMULATED CLOSED" if blocked else "")
            ),
        ).add_to(map_object)
        shown += 1
        if shown >= 9000:
            break

    if baseline:
        folium.PolyLine(
            path_coordinates(graph, baseline),
            color="#2471a3",
            weight=5,
            opacity=0.9,
            tooltip="Normal shortest-time route",
        ).add_to(map_object)

    if safer:
        folium.PolyLine(
            path_coordinates(graph, safer),
            color="#8e44ad",
            weight=6,
            opacity=0.98,
            tooltip="Flood-aware A* alternative route",
        ).add_to(map_object)

    folium.Marker(
        (start[1], start[0]),
        tooltip="Starting location",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(map_object)

    folium.Marker(
        (destination[1], destination[0]),
        tooltip="Destination (study/demo point)",
        icon=folium.Icon(color="purple", icon="flag"),
    ).add_to(map_object)

    folium.LayerControl().add_to(map_object)
    return map_object, shown, blocked_drawn


with st.sidebar:
    st.header("Evacuation scenario")
    start_label = st.selectbox("Starting location", list(LANDMARKS))
    dest_label = st.selectbox("Destination", list(DEMO_DESTINATIONS))

    st.subheader("Flood scenario simulator")
    flood_active = st.toggle(
        "Activate hypothetical flood",
        value=False,
        help="Applies scenario-based road closures. This is not a live flood observation.",
    )
    flood_spread_m = st.slider(
        "Simulated flood spread from river (metres)",
        min_value=100,
        max_value=1500,
        value=500,
        step=50,
    )
    vulnerability_threshold = st.slider(
        "Road closure vulnerability threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="A road is closed in the scenario only if it is within the selected spread and meets this score.",
    )
    rainfall_mm = st.slider(
        "Hypothetical 24-hour rainfall (mm)",
        0,
        150,
        60,
        5,
        help="Scenario input only; not live rainfall or a forecast.",
    )
    risk_weight = st.slider(
        "Risk avoidance weight",
        0.0,
        5.0,
        2.5,
        0.25,
        help="Higher values penalize roads with greater illustrative susceptibility.",
    )
    st.caption(
        "Roads: OpenStreetMap. Terrain: processed SRTM. "
        "River: mapped Shipra OSM geometry."
    )


try:
    G = load_graph()
    load_rasters()
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

node_ids = list(G.nodes)
node_xy = np.array([[G.nodes[node]["x"], G.nodes[node]["y"]] for node in node_ids])
tree = cKDTree(node_xy)

with st.spinner("Preparing terrain susceptibility…"):
    records = edge_records_from_graph(G)
    edge_risks = prepare_edge_terrain(records)

terrain_by_edge = {
    (u, v): float(risk)
    for (u, v, _lon, _lat), risk in zip(records, edge_risks)
}

try:
    with st.spinner("Calculating road-to-river proximity…"):
        river_rows = cached_river_distances(tuple((u, v) for u, v in G.edges()))
    river_distances = {(u, v): distance for u, v, distance in river_rows}
except Exception as exc:
    st.error(f"River proximity calculation failed: {exc}")
    st.stop()

# Preserve the existing rainfall scenario scaling for terrain susceptibility.
rainfall_factor = 0.30 + 0.70 * (rainfall_mm / 150.0)
terrain_scenario = {
    edge: float(np.clip(value * rainfall_factor, 0, 1))
    for edge, value in terrain_by_edge.items()
}

scenario = classify_flood_scenario(
    river_distances=river_distances,
    terrain_risks=terrain_scenario,
    flood_spread_m=flood_spread_m,
    vulnerability_threshold=vulnerability_threshold,
    terrain_weight=0.55,
    proximity_weight=0.45,
)

# Build the weighted graph and attach the scenario vulnerability to each edge.
R = nx.DiGraph()
R.add_nodes_from(G.nodes(data=True))
for u, v, data in G.edges(data=True):
    risk = float(scenario[(u, v)]["vulnerability"])
    R.add_edge(
        u,
        v,
        length_m=float(data["length_m"]),
        travel_s=float(data["travel_s"]),
        risk=risk,
        highway=data.get("highway", "unclassified"),
        name=data.get("name", "Unnamed road"),
    )

start = LANDMARKS[start_label]
destination = DEMO_DESTINATIONS[dest_label]
source = nearest_node(R, tree, node_ids, start)
target = nearest_node(R, tree, node_ids, destination)

try:
    baseline = nx.shortest_path(R, source, target, weight="travel_s")
except (nx.NetworkXNoPath, nx.NodeNotFound):
    st.error("No baseline driving route found for the selected points.")
    st.stop()


def risk_cost(u, v, data):
    return data["travel_s"] * (1.0 + risk_weight * data["risk"])


def heuristic(node, goal):
    # 120 km/h is an upper-bound speed used for an admissible time heuristic.
    return haversine_m(
        R.nodes[node]["x"],
        R.nodes[node]["y"],
        R.nodes[goal]["x"],
        R.nodes[goal]["y"],
    ) / (120_000 / 3600)


# Flood closures are applied to a copy; the source graph is never mutated.
A = apply_flood_closures(R, scenario, activate=flood_active)

try:
    safer = nx.astar_path(
        A,
        source,
        target,
        heuristic=heuristic,
        weight=risk_cost,
    )
    safe_stats = route_stats(A, safer)
    route_error = None
except (nx.NetworkXNoPath, nx.NodeNotFound):
    safer = []
    safe_stats = None
    route_error = (
        "No feasible route remains under this simulated flood scenario. "
        "Reduce the spread or adjust the scenario threshold."
    )

base_stats = route_stats(R, baseline)
scenario_summary = summarize_scenario(scenario)
blocked_directed = scenario_summary["blocked_directed_edges"] if flood_active else 0
blocked_physical = len({
    (min(u, v), max(u, v))
    for (u, v), values in scenario.items()
    if flood_active and values["blocked"]
})

map_obj, roads_shown, blocked_drawn = build_map(
    R,
    baseline,
    safer,
    start,
    destination,
    scenario,
    flood_active,
    flood_spread_m,
)

st.subheader("Shipra River and evacuation network")
st.caption(
    "Green = lower illustrative susceptibility; yellow = moderate; orange = high; "
    "red = simulated blocked road. Blue = normal route; purple = flood-aware A* route. "
    "Map road display is capped for browser performance."
)
st_folium(map_obj, height=590, use_container_width=True, returned_objects=[])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Directed road edges", f"{R.number_of_edges():,}")
m2.metric("Road edges within spread", f"{sum(d <= flood_spread_m for d in river_distances.values()):,}")
m3.metric("Simulated blocked directions", f"{blocked_directed:,}")
m4.metric("Flood spread", f"{flood_spread_m:,} m", "Hypothetical")

st.subheader("Flood scenario summary")
s1, s2, s3 = st.columns(3)
s1.metric("Physical road pairs blocked", f"{blocked_physical:,}")
s2.metric("High susceptibility edges", f"{scenario_summary['categories']['HIGH']:,}")
s3.metric("Moderate susceptibility edges", f"{scenario_summary['categories']['MODERATE']:,}")

st.subheader("Route comparison")
if route_error:
    st.error(route_error)
    st.metric("Normal route distance", f"{base_stats['distance_km']:.2f} km")
    st.metric("Normal route estimated time", f"{base_stats['minutes']:.1f} min")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Normal route distance", f"{base_stats['distance_km']:.2f} km")
    c2.metric("Normal route time estimate", f"{base_stats['minutes']:.1f} min")
    c3.metric("Normal route susceptibility", f"{base_stats['risk']:.2f} / 1.00")

    d1, d2, d3 = st.columns(3)
    d1.metric("A* route distance", f"{safe_stats['distance_km']:.2f} km")
    d2.metric("A* route time estimate", f"{safe_stats['minutes']:.1f} min")
    d3.metric("A* route susceptibility", f"{safe_stats['risk']:.2f} / 1.00")

    distance_change = safe_stats["distance_km"] - base_stats["distance_km"]
    time_change = safe_stats["minutes"] - base_stats["minutes"]
    comparison = pd.DataFrame([
        {
            "Route": "Normal shortest-time route",
            "Distance (km)": round(base_stats["distance_km"], 3),
            "Estimated time (min)": round(base_stats["minutes"], 2),
            "Weighted susceptibility": round(base_stats["risk"], 3),
        },
        {
            "Route": "Flood-aware A* route",
            "Distance (km)": round(safe_stats["distance_km"], 3),
            "Estimated time (min)": round(safe_stats["minutes"], 2),
            "Weighted susceptibility": round(safe_stats["risk"], 3),
        },
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    if flood_active and blocked_directed:
        st.success(
            f"Flood scenario active: {blocked_directed:,} directed road edges "
            f"({blocked_physical:,} physical road pairs) were excluded before A* routing."
        )
        if safer != baseline:
            st.info(
                f"A* selected a different route. Distance change: {distance_change:+.2f} km; "
                f"estimated time change: {time_change:+.1f} min."
            )
        else:
            st.info(
                "The selected route did not change. The simulated closures did not intersect "
                "the normal route, or an equally suitable route remained."
            )
    elif flood_active:
        st.info(
            "The flood scenario is active, but no road met both the selected spread "
            "and vulnerability threshold. Increase the spread or adjust the threshold "
            "to explore a stronger hypothetical scenario."
        )
    else:
        st.info("Flood simulation is off. Activate it in the sidebar to test adaptive rerouting.")

st.subheader("Historical rainfall context")
rain_file = DATA / "rainfall" / "seasons" / "2020" / "daily_rainfall.csv"
if rain_file.exists():
    rain = pd.read_csv(rain_file, parse_dates=["date"])
    daily = rain.groupby("date", as_index=False)["rainfall_mm"].mean()
    st.caption(
        f"CHIRPS v3 historical pilot: {daily['date'].min().date()} to "
        f"{daily['date'].max().date()}. Regional grid-cell mean, not street-level rainfall."
    )
    fig = px.line(
        daily,
        x="date",
        y="rainfall_mm",
        labels={"date": "Date", "rainfall_mm": "Rainfall (mm/day)"},
        title="Historical rainfall pilot — 2020",
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Historical rainfall pilot CSV not found.")

st.subheader("Prototype methodology")
st.markdown("""
1. The road network and Shipra River geometry are derived from OpenStreetMap.
2. Road-to-river distance is calculated from actual road LineString geometries in a metric CRS.
3. Terrain susceptibility uses the existing relative-elevation, TPI and slope rasters.
4. The terrain index combines 65% relative elevation, 25% TPI and 10% slope components.
5. A scenario vulnerability score combines 55% terrain susceptibility and 45% river proximity.
6. Roads are marked blocked only when they are within the selected hypothetical spread and meet the selected vulnerability threshold.
7. The normal route minimizes estimated travel time. Flood-aware A* runs on a graph copy with simulated blocked directed edges removed.
""")
st.caption(
    "Limitations: this is a scenario-based academic prototype, not a hydraulic model or flood forecast. "
    "The terrain grid is approximately 250 m; relative elevation is a local terrain proxy, not calibrated HAND "
    "or flood depth. The flood-spread slider is a hypothetical distance, not a measured river stage. "
    "Travel speeds are estimated where OSM maxspeed is unavailable. Demo destinations are synthetic, "
    "not verified shelters. Do not use this prototype for operational emergency decisions."
)
'@

Set-Content -Path $ModulePath -Value $Module -Encoding UTF8
Set-Content -Path $AppPath -Value $App -Encoding UTF8

Write-Host ""
Write-Host "GeoFlood AI flood simulation integration installed." -ForegroundColor Green
Write-Host "Backup created: $AppPath.backup-$Stamp"
Write-Host "Backup created for flood_simulator.py if it existed."
Write-Host ""
Write-Host "Next run:"
Write-Host ".\venv\Scripts\python.exe -m py_compile .\evacuation_app.py .\src\evacuation\flood_simulator.py"
Write-Host "If compilation succeeds:"
Write-Host ".\venv\Scripts\python.exe -m streamlit run .\evacuation_app.py"
