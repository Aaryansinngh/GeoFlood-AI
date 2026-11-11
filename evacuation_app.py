from __future__ import annotations

import json
import math
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import streamlit as st
from folium.plugins import Draw
from scipy.spatial import cKDTree
from shapely.geometry import shape
from streamlit_folium import st_folium

from src.evacuation.flood_simulator import (
    calculate_edge_river_distances,
    classify_flood_scenario,
    summarize_scenario,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "ujjain"
OSM_DIR = DATA / "osm"
GRAPH_FILE = OSM_DIR / "ujjain_directed_roads.graphml"
ROAD_FILE = OSM_DIR / "ujjain_road_edges.gpkg"
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
<div class="hero"><h1>GEOFLOOD AI</h1>
<p>Adaptive flood-aware evacuation intelligence · Ujjain / Shipra River</p></div>
""", unsafe_allow_html=True)
st.warning(
    "ACADEMIC PROTOTYPE — Flood zones and closures are user-defined simulations, "
    "not measured flood extents or operational emergency guidance."
)


@st.cache_resource(show_spinner="Loading directed Ujjain road graph…")
def load_graph():
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
    }
    return {key: rasterio.open(path) for key, path in files.items()}


@st.cache_data(show_spinner="Sampling terrain along road network…")
def prepare_edge_terrain(edge_records):
    rasters = load_rasters()
    coords = [(record[2], record[3]) for record in edge_records]
    arrays = {
        key: np.array([value[0] for value in rasters[key].sample(coords)], dtype=float)
        for key in ("relative", "tpi", "slope")
    }

    def normalize(values, invert=False):
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            result = np.full(len(values), 0.5)
        else:
            low, high = np.nanpercentile(valid, [5, 95])
            result = np.full(len(values), 0.5) if high <= low else np.clip(
                (values - low) / (high - low), 0, 1
            )
        return 1 - result if invert else result

    rel = normalize(arrays["relative"], invert=True)
    tpi = normalize(arrays["tpi"], invert=True)
    slope = normalize(arrays["slope"])
    return np.clip(0.65 * rel + 0.25 * tpi + 0.10 * slope, 0, 1).tolist()


@st.cache_resource(show_spinner="Loading physical road geometries…")
def load_roads_metric():
    roads = gpd.read_file(ROAD_FILE)
    if roads.crs is None:
        raise ValueError("Road GeoPackage has no CRS.")
    return roads.to_crs("EPSG:32643")


@st.cache_data(show_spinner="Measuring river proximity…")
def get_river_distances(_graph_signature):
    graph = load_graph()
    distances, _ = calculate_edge_river_distances(graph)
    return [(int(u), int(v), float(d)) for (u, v), d in distances.items()]


@st.cache_data(show_spinner="Finding roads intersecting drawn flood zones…")
def find_polygon_blocked_edges(drawings_json, _graph_signature):
    if not drawings_json:
        return []

    parsed = json.loads(drawings_json)

    # Support list, Feature and FeatureCollection responses.
    if isinstance(parsed, dict):
        if parsed.get("type") == "FeatureCollection":
            drawings = parsed.get("features", [])
        elif parsed.get("type") == "Feature":
            drawings = [parsed]
        else:
            drawings = []
    elif isinstance(parsed, list):
        drawings = parsed
    else:
        drawings = []

    polygons = []
    for feature in drawings:
        if not isinstance(feature, dict):
            continue

        geometry = feature.get("geometry", feature)
        if not isinstance(geometry, dict):
            continue

        try:
            geom = shape(geometry)
            if geom.geom_type in ("Polygon", "MultiPolygon") and not geom.is_empty:
                polygons.append(geom)
        except Exception:
            continue

    st.session_state["flood_debug"] = {
        "response_type": type(parsed).__name__,
        "feature_count": len(drawings),
        "polygon_count": len(polygons),
    }

    if not polygons:
        return []

    zones = gpd.GeoSeries(polygons, crs="EPSG:4326").to_crs("EPSG:32643")
    zone_union = zones.unary_union.buffer(1.0)

    roads = load_roads_metric()
    hits = roads[roads.geometry.intersects(zone_union)]

    graph = load_graph()
    node_lookup = {str(node): node for node in graph.nodes}

    blocked = set()
    mapped_rows = 0

    for row in hits.itertuples():
        u = node_lookup.get(str(row.u))
        v = node_lookup.get(str(row.v))

        if u is None or v is None:
            try:
                u = int(float(row.u))
                v = int(float(row.v))
            except (TypeError, ValueError):
                continue

        if graph.has_edge(u, v):
            blocked.add((u, v))
            mapped_rows += 1

        if graph.has_edge(v, u):
            blocked.add((v, u))
            mapped_rows += 1

    st.session_state["flood_debug"].update({
        "road_geometry_count": len(roads),
        "intersecting_physical_roads": len(hits),
        "mapped_directed_edges": mapped_rows,
        "blocked_directed_edges": len(blocked),
        "zone_bounds_metric": list(zone_union.bounds),
    })

    return sorted(blocked)
def edge_records_from_graph(graph):
    return [
        (u, v,
         (graph.nodes[u]["x"] + graph.nodes[v]["x"]) / 2,
         (graph.nodes[u]["y"] + graph.nodes[v]["y"]) / 2)
        for u, v in graph.edges()
    ]


def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))


def nearest_node(graph, tree, node_ids, point):
    _, index = tree.query([point[0], point[1]], k=1)
    return node_ids[int(index)]


def route_stats(graph, path):
    edges = [graph[u][v] for u, v in zip(path[:-1], path[1:])]
    distance = sum(float(e["length_m"]) for e in edges)
    seconds = sum(float(e["travel_s"]) for e in edges)
    risk = sum(float(e["length_m"]) * float(e.get("risk", 0)) for e in edges)
    return {
        "distance_km": distance / 1000,
        "minutes": seconds / 60,
        "risk": risk / max(distance, 1e-9),
    }


def path_coordinates(graph, path):
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        if not coords:
            coords.append((graph.nodes[u]["y"], graph.nodes[u]["x"]))
        coords.append((graph.nodes[v]["y"], graph.nodes[v]["x"]))
    return coords


def build_map(graph, baseline, safer, start, destination, scenario,
              blocked_edges, flood_features, draw_enabled):
    m = folium.Map(location=CENTER, zoom_start=14, tiles="OpenStreetMap",
                   control_scale=True)

    if RIVER_FILE.exists():
        river = json.loads(RIVER_FILE.read_text(encoding="utf-8"))
        folium.GeoJson(
            river, name="Shipra River",
            style_function=lambda _: {"color": "#0877c9", "weight": 5, "opacity": 0.95},
            tooltip="Shipra River — OpenStreetMap geometry",
        ).add_to(m)

    if flood_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": flood_features},
            name="User-drawn simulated flood zones",
            style_function=lambda _: {
                "color": "#d7191c", "weight": 2,
                "fillColor": "#ef4444", "fillOpacity": 0.22,
            },
            tooltip="User-defined hypothetical flood zone",
        ).add_to(m)

    blocked_set = set(blocked_edges)
    seen = set()
    shown = 0
    major = {"primary", "secondary", "tertiary", "trunk",
             "primary_link", "secondary_link", "tertiary_link", "trunk_link"}

    for u, v, data in graph.edges(data=True):
        if data.get("highway", "") not in major:
            continue
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        seen.add(pair)
        info = scenario.get((u, v), {})
        blocked = (u, v) in blocked_set or (v, u) in blocked_set
        score = float(info.get("vulnerability", data.get("risk", 0)))
        color = "#d7191c" if blocked else (
            "#f28e2b" if score >= 0.60 else
            "#f2c744" if score >= 0.35 else "#2ca25f"
        )
        folium.PolyLine(
            [(graph.nodes[u]["y"], graph.nodes[u]["x"]),
             (graph.nodes[v]["y"], graph.nodes[v]["x"])],
            color=color, weight=4 if blocked else 2,
            opacity=0.9 if blocked else 0.62,
            tooltip=f'{data.get("name", "Unnamed road")} | score {score:.2f}'
                    + (" | SIMULATED BLOCKED" if blocked else ""),
        ).add_to(m)
        shown += 1
        if shown >= 9000:
            break

    if baseline:
        folium.PolyLine(path_coordinates(graph, baseline), color="#2471a3",
                        weight=5, opacity=0.9,
                        tooltip="Normal shortest-time route").add_to(m)
    if safer:
        folium.PolyLine(path_coordinates(graph, safer), color="#8e44ad",
                        weight=6, opacity=0.98,
                        tooltip="Flood-aware A* alternative").add_to(m)

    folium.Marker((start[1], start[0]), tooltip="Start",
                  icon=folium.Icon(color="blue", icon="play")).add_to(m)
    folium.Marker((destination[1], destination[0]), tooltip="Simulated destination",
                  icon=folium.Icon(color="purple", icon="flag")).add_to(m)

    if draw_enabled:
        Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": False, "circle": False, "circlemarker": False,
                "marker": False,
                "polygon": {"allowIntersection": False, "showArea": True},
                "rectangle": True,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m, shown


with st.sidebar:
    st.header("Evacuation scenario")
    start_label = st.selectbox("Starting location", list(LANDMARKS))
    dest_label = st.selectbox("Destination", list(DEMO_DESTINATIONS))

    st.subheader("Flood simulation")
    flood_active = st.toggle("Activate flood simulation", value=False)
    flood_mode = st.radio(
        "Flood zone input",
        ["Draw on map", "Shipra river spread"],
        help="Draw polygons/rectangles anywhere on the map, or use the river-proximity scenario.",
    )
    flood_spread_m = st.slider("Shipra spread distance (m)", 100, 1500, 500, 50,
                               disabled=(flood_mode != "Shipra river spread"))
    threshold = st.slider("Vulnerability threshold", 0.30, 0.90, 0.65, 0.05)
    rainfall_mm = st.slider("Hypothetical rainfall (mm)", 0, 150, 60, 5)
    risk_weight = st.slider("Risk avoidance weight", 0.0, 5.0, 2.5, 0.25)
    st.caption("Flood zones and closures are simulated. They are not observed flood extents.")

try:
    G = load_graph()
    load_rasters()
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

node_ids = list(G.nodes)
node_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_ids])
tree = cKDTree(node_xy)
records = edge_records_from_graph(G)
edge_risks = prepare_edge_terrain(records)
terrain_by_edge = {
    (u, v): float(risk)
    for (u, v, _, _), risk in zip(records, edge_risks)
}

try:
    river_rows = get_river_distances(G.number_of_edges())
    river_distances = {(u, v): d for u, v, d in river_rows}
except Exception as exc:
    st.error(f"Could not calculate river proximity: {exc}")
    st.stop()

rainfall_factor = 0.30 + 0.70 * (rainfall_mm / 150.0)
terrain_scenario = {
    edge: float(np.clip(value * rainfall_factor, 0, 1))
    for edge, value in terrain_by_edge.items()
}
scenario = classify_flood_scenario(
    river_distances, terrain_scenario,
    flood_spread_m=flood_spread_m,
    vulnerability_threshold=threshold,
    terrain_weight=0.55,
    proximity_weight=0.45,
)

R = nx.DiGraph()
R.add_nodes_from(G.nodes(data=True))
for u, v, data in G.edges(data=True):
    R.add_edge(
        u, v,
        length_m=float(data["length_m"]),
        travel_s=float(data["travel_s"]),
        risk=float(scenario[(u, v)]["vulnerability"]),
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
    st.error("No normal route found for these points.")
    st.stop()


def risk_cost(u, v, data):
    return data["travel_s"] * (1 + risk_weight * data["risk"])


def heuristic(node, goal):
    return haversine_m(
        R.nodes[node]["x"], R.nodes[node]["y"],
        R.nodes[goal]["x"], R.nodes[goal]["y"],
    ) / (120_000 / 3600)


# Build a first map in a placeholder to collect Leaflet Draw geometry.
if "geoflood_drawings" not in st.session_state:
    st.session_state["geoflood_drawings"] = []

preview, _ = build_map(
    R, baseline, [], start, destination, scenario,
    [], st.session_state["geoflood_drawings"], draw_enabled=True,
)
map_slot = st.empty()
draw_result = map_slot_folium = st_folium(
    preview, height=590, use_container_width=True,
    returned_objects=["all_drawings"], key="geoflood_draw_map",
)
new_drawings = draw_result.get("all_drawings") or []
if new_drawings != st.session_state["geoflood_drawings"]:
    st.session_state["geoflood_drawings"] = new_drawings

flood_features = st.session_state["geoflood_drawings"]
drawing_json = json.dumps(flood_features, sort_keys=True)

polygon_blocked = []
if flood_mode == "Draw on map" and drawing_json != "[]":
    try:
        polygon_blocked = find_polygon_blocked_edges(drawing_json, G.number_of_edges())
    except Exception as exc:
        st.error(f"Could not intersect drawn zone with roads: {exc}")

# Manual closure selection is limited to the currently calculated normal route.
route_options = {}
for index, (u, v) in enumerate(zip(baseline[:-1], baseline[1:]), start=1):
    edge_data = R[u][v]
    label = (
        f"{index}. {edge_data.get('name', 'Unnamed road')} "
        f"({edge_data['length_m']:.0f} m; node pair {u} → {v})"
    )
    route_options[label] = (u, v)

with st.sidebar:
    st.subheader("Manual road closures")
    manual_labels = st.multiselect(
        "Select roads on the normal route to block",
        options=list(route_options),
        help="For individual road closures. Use the map drawing tool for larger areas.",
    )

manual_blocked = [route_options[label] for label in manual_labels]
if flood_mode == "Draw on map":
    selected_blocked = sorted(set(polygon_blocked) | set(manual_blocked))
else:
    selected_blocked = [
        edge for edge, values in scenario.items() if values["blocked"]
    ]
    selected_blocked = sorted(set(selected_blocked) | set(manual_blocked))

active_blocked = selected_blocked if flood_active else []
A = R.copy()
for u, v in active_blocked:
    if A.has_edge(u, v):
        A.remove_edge(u, v)

try:
    safer = nx.astar_path(A, source, target, heuristic=heuristic, weight=risk_cost)
    safe_stats = route_stats(A, safer)
    route_error = None
except (nx.NetworkXNoPath, nx.NodeNotFound):
    safer, safe_stats = [], None
    route_error = "No route remains after applying the selected simulated closures."

base_stats = route_stats(R, baseline)
summary = summarize_scenario(scenario)
blocked_physical = len({(min(u, v), max(u, v)) for u, v in active_blocked})

final_map, roads_shown = build_map(
    R, baseline, safer, start, destination, scenario,
    active_blocked, flood_features if flood_mode == "Draw on map" else [],
    draw_enabled=True,
)
map_slot.empty()
st.subheader("Interactive flood and evacuation map")
st.caption(
    "Use the polygon or rectangle drawing buttons in the map's upper-left corner. "
    "Draw around roads to define a simulated flood zone. Edit/remove drawings using the map toolbar. "
    "Red = blocked; blue = normal route; purple = A* route; blue line = Shipra River."
)
st_folium(
    final_map, height=590, use_container_width=True,
    returned_objects=["all_drawings"], key="geoflood_final_map",
)

if flood_mode == "Draw on map":
    st.info(
        f"Drawn zones currently intersect {len(polygon_blocked):,} directed road edges."
    )

    st.subheader("Flood Detection Diagnostics")
    st.write("Drawing object type:", type(flood_features).__name__)
    st.write("Number of drawing features:", len(flood_features))

    if flood_features:
        st.write("First drawing structure:")
        st.json(flood_features[0])

    st.write("Detected blocked directed edges:", len(polygon_blocked))

    debug_data = st.session_state.get("flood_debug", {})
    if debug_data:
        st.write("Geometry processing diagnostics:")
        st.json(debug_data)
    st.write("Flood simulation active:", flood_active)

    if polygon_blocked:
        st.success("Road intersection detection is working.")
    else:
        st.warning("No road intersections detected. Further geometry debugging required.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Directed road edges", f"{R.number_of_edges():,}")
m2.metric("Roads intersecting selected zones", f"{len(polygon_blocked):,}" if flood_mode == "Draw on map" else "—")
m3.metric("Simulated blocked directions", f"{len(active_blocked):,}")
m4.metric("Physical road pairs blocked", f"{blocked_physical:,}")

st.subheader("Route comparison")
if route_error:
    st.error(route_error)
    st.metric("Normal route distance", f"{base_stats['distance_km']:.2f} km")
    st.metric("Normal route estimated time", f"{base_stats['minutes']:.1f} min")
else:
    left, right = st.columns(2)
    with left:
        st.markdown("**Normal shortest-time route**")
        st.metric("Distance", f"{base_stats['distance_km']:.2f} km")
        st.metric("Estimated time", f"{base_stats['minutes']:.1f} min")
        st.metric("Susceptibility", f"{base_stats['risk']:.2f} / 1.00")
    with right:
        st.markdown("**Flood-aware A* route**")
        st.metric("Distance", f"{safe_stats['distance_km']:.2f} km")
        st.metric("Estimated time", f"{safe_stats['minutes']:.1f} min")
        st.metric("Susceptibility", f"{safe_stats['risk']:.2f} / 1.00")
    st.dataframe(pd.DataFrame([
        {"Route": "Normal shortest-time", "Distance (km)": base_stats["distance_km"],
         "Estimated time (min)": base_stats["minutes"], "Susceptibility": base_stats["risk"]},
        {"Route": "Flood-aware A*", "Distance (km)": safe_stats["distance_km"],
         "Estimated time (min)": safe_stats["minutes"], "Susceptibility": safe_stats["risk"]},
    ]).round(3), use_container_width=True, hide_index=True)

    if flood_active and active_blocked:
        st.success(
            f"Scenario applied: {len(active_blocked):,} directed edges "
            f"across {blocked_physical:,} physical road pairs excluded from routing."
        )
        if safer != baseline:
            st.info(
                f"A* changed the route by {safe_stats['distance_km'] - base_stats['distance_km']:+.2f} km "
                f"and {safe_stats['minutes'] - base_stats['minutes']:+.1f} estimated minutes."
            )
        else:
            st.info("The selected closures did not change the chosen route.")
    elif flood_active:
        st.info("Flood simulation is active, but no roads currently meet the selected closure criteria.")
    else:
        st.info("Flood simulation is off. Draw a zone or select road closures, then activate the simulation.")

st.subheader("Historical rainfall context")
rain_file = DATA / "rainfall" / "seasons" / "2020" / "daily_rainfall.csv"
if rain_file.exists():
    rain = pd.read_csv(rain_file, parse_dates=["date"])
    daily = rain.groupby("date", as_index=False)["rainfall_mm"].mean()
    st.caption(
        f"CHIRPS v3 historical pilot: {daily['date'].min().date()} to "
        f"{daily['date'].max().date()}. Regional grid-cell mean, not street-level rainfall."
    )
    fig = px.line(daily, x="date", y="rainfall_mm",
                  labels={"date": "Date", "rainfall_mm": "Rainfall (mm/day)"},
                  title="Historical rainfall pilot — 2020")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Prototype methodology and limitations")
st.markdown("""
- Road and river geometry originate from OpenStreetMap.
- User-drawn polygons/rectangles are transformed to a metric CRS and intersected with actual OSM road LineStrings.
- Intersecting physical roads are mapped back to directed graph edges; manual closures can also be selected from the normal route.
- Flood-aware A* operates on a copy of the directed graph after simulated closures are removed.
- Shipra proximity mode combines river distance and terrain susceptibility as an illustrative scenario score.
""")
st.caption(
    "This is not a hydraulic flood model or real-time forecast. The terrain grid is approximately 250 m; "
    "relative elevation is a local proxy, not calibrated HAND or flood depth. User-drawn areas are hypothetical. "
    "Estimated road speeds are not validated travel times. Demo destinations are synthetic and are not verified shelters. "
    "Do not use this prototype for operational emergency response."
)




