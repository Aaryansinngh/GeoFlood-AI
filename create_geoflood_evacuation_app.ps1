$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
if (-not (Test-Path (Join-Path $root "requirements.txt"))) {
    throw "Run this from the GeoFlood-AI project root."
}
$content = @'
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "ujjain"
OSM_DIR = DATA / "osm"
GRAPH_FILE = OSM_DIR / "ujjain_directed_roads.graphml"

CENTER = (23.1828, 75.7683)
LANDMARKS = {
    "Ram Ghat (study reference)": (75.7683, 23.1828),
    "Ujjain Junction (study reference)": (75.7850, 23.1785),
    "Mahakaleshwar area (study reference)": (75.7685, 23.1828),
}
# Explicitly synthetic points used only to demonstrate route planning.
DEMO_DESTINATIONS = {
    "Demo evacuation point A (simulated)": (75.7920, 23.2020),
    "Demo evacuation point B (simulated)": (75.7900, 23.1630),
    "Demo evacuation point C (simulated)": (75.7350, 23.1950),
}

st.set_page_config(
    page_title="GeoFlood AI | Evacuation Intelligence",
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
<p>Flood-aware evacuation intelligence · Ujjain / Shipra River study area</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "ACADEMIC PROTOTYPE — Not an operational emergency service. "
    "Rainfall scenarios and destination points are simulated. "
    "Do not use these routes for real evacuation decisions."
)

@st.cache_resource(show_spinner="Loading directed Ujjain road graph…")
def load_graph():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(f"Directed road graph not found: {GRAPH_FILE}")
    graph = nx.read_graphml(GRAPH_FILE)
    graph = nx.relabel_nodes(graph, {n: int(n) for n in graph.nodes})
    # GraphML stores attributes as strings in some NetworkX/OSM combinations.
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

@st.cache_data(show_spinner="Sampling terrain along the road network…")
def prepare_edge_terrain(edge_records):
    # Input records are plain tuples so the result can be safely cached.
    rasters = load_rasters()
    rel_ds, tpi_ds, slope_ds = rasters["relative"], rasters["tpi"], rasters["slope"]
    coords = [(r[2], r[3]) for r in edge_records]
    rel_values = np.array([v[0] for v in rel_ds.sample(coords)], dtype=float)
    tpi_values = np.array([v[0] for v in tpi_ds.sample(coords)], dtype=float)
    slope_values = np.array([v[0] for v in slope_ds.sample(coords)], dtype=float)

    def normalize(values, invert=False):
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.full(len(values), 0.5)
        lo, hi = np.nanpercentile(valid, [5, 95])
        if hi <= lo:
            out = np.full(len(values), 0.5)
        else:
            out = np.clip((values - lo) / (hi - lo), 0, 1)
        return 1 - out if invert else out

    rel_score = normalize(rel_values, invert=True)
    tpi_score = normalize(tpi_values, invert=True)
    slope_score = normalize(slope_values)
    index = np.clip(0.65 * rel_score + 0.25 * tpi_score + 0.10 * slope_score, 0, 1)
    return index.astype(float).tolist()

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
    _, idx = tree.query([lon, lat], k=1)
    return node_ids[int(idx)]

def route_stats(graph, path):
    if len(path) < 2:
        return {"distance_km": 0.0, "minutes": 0.0, "risk": 0.0}
    edges = [graph[u][v] for u, v in zip(path[:-1], path[1:])]
    distance = sum(float(e["length_m"]) for e in edges)
    seconds = sum(float(e["travel_s"]) for e in edges)
    weighted = sum(float(e["length_m"]) * float(e["risk"]) for e in edges) / max(distance, 1e-9)
    return {"distance_km": distance / 1000, "minutes": seconds / 60, "risk": weighted}

def path_coordinates(graph, path):
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        # Segment WKT is stored during graph construction. Use node endpoints
        # for the map if WKT parsing is unavailable.
        x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
        x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
        coords.extend([(y1, x1), (y2, x2)])
    return coords

def build_map(graph, node_ids, terrain_risk, baseline, safer, start, destination, closed_edge):
    m = folium.Map(location=CENTER, zoom_start=14, tiles="CartoDB positron", control_scale=True)
    # Draw major roads only to keep the interactive map responsive.
    major = {"primary", "secondary", "tertiary", "trunk", "primary_link", "trunk_link"}
    shown = 0
    for u, v, data in graph.edges(data=True):
        if data.get("highway", "") not in major:
            continue
        if shown >= 9000:
            break
        risk = float(data.get("risk", 0.0))
        color = "#e74c3c" if risk >= 0.70 else "#e6b422" if risk >= 0.40 else "#31a66a"
        folium.PolyLine(
            [(graph.nodes[u]["y"], graph.nodes[u]["x"]),
             (graph.nodes[v]["y"], graph.nodes[v]["x"])],
            color=color, weight=2, opacity=0.55,
            tooltip=f'{data.get("name", "Unnamed road")} · illustrative index {risk:.2f}'
        ).add_to(m)
        shown += 1

    if baseline:
        folium.PolyLine(path_coordinates(graph, baseline), color="#2471a3",
                        weight=5, opacity=0.85, tooltip="Shortest-time baseline").add_to(m)
    if safer:
        folium.PolyLine(path_coordinates(graph, safer), color="#8e44ad",
                        weight=6, opacity=0.95, tooltip="Risk-aware A* route").add_to(m)
    if closed_edge:
        u, v = closed_edge
        folium.PolyLine(
            [(graph.nodes[u]["y"], graph.nodes[u]["x"]),
             (graph.nodes[v]["y"], graph.nodes[v]["x"])],
            color="#c0392b", weight=9, opacity=1,
            tooltip="Simulated closed road segment"
        ).add_to(m)

    folium.Marker((start[1], start[0]), tooltip="Start",
                  icon=folium.Icon(color="blue", icon="play")).add_to(m)
    folium.Marker((destination[1], destination[0]), tooltip="Simulated destination",
                  icon=folium.Icon(color="purple", icon="flag")).add_to(m)
    folium.LayerControl().add_to(m)
    return m, shown

with st.sidebar:
    st.header("Evacuation scenario")
    start_label = st.selectbox("Starting location", list(LANDMARKS))
    dest_label = st.selectbox("Destination", list(DEMO_DESTINATIONS))
    rainfall_mm = st.slider(
        "Hypothetical 24-hour rainfall (mm)", 0, 150, 60, 5,
        help="Scenario input only; this is not live rainfall or a forecast."
    )
    risk_weight = st.slider(
        "Risk avoidance weight", 0.0, 5.0, 2.5, 0.25,
        help="Higher values penalize roads with higher terrain susceptibility."
    )
    close_road = st.checkbox("Close highest-index road on baseline route")
    st.caption("Roads: OpenStreetMap regional extract. Terrain: processed SRTM rasters.")

try:
    G = load_graph()
    rasters = load_rasters()
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

node_ids = list(G.nodes)
node_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_ids])
tree = cKDTree(node_xy)

with st.spinner("Calculating terrain susceptibility for road segments…"):
    records = edge_records_from_graph(G)
    edge_risks = prepare_edge_terrain(records)

# Build a directed weighted graph with terrain/rainfall scenario costs.
scenario_factor = 0.30 + 0.70 * (rainfall_mm / 150.0)
R = nx.DiGraph()
R.add_nodes_from(G.nodes(data=True))
for (u, v, _, _), terrain_index in zip(records, edge_risks):
    risk = float(np.clip(terrain_index * scenario_factor, 0, 1))
    data = G[u][v]
    R.add_edge(
        u, v,
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

closed_edge = None
A = R.copy()
if close_road:
    candidates = [(u, v, A[u][v]["risk"]) for u, v in zip(baseline[:-1], baseline[1:])]
    if candidates:
        cu, cv, _ = max(candidates, key=lambda x: x[2])
        closed_edge = (cu, cv)
        A.remove_edge(cu, cv)

def risk_cost(u, v, data):
    return data["travel_s"] * (1.0 + risk_weight * data["risk"])

def heuristic(node, goal):
    return haversine_m(
        R.nodes[node]["x"], R.nodes[node]["y"],
        R.nodes[goal]["x"], R.nodes[goal]["y"]
    ) / (120_000 / 3600)

try:
    safer = nx.astar_path(A, source, target, heuristic=heuristic, weight=risk_cost)
    safe_stats = route_stats(A, safer)
    route_error = None
except (nx.NetworkXNoPath, nx.NodeNotFound):
    safer = []
    safe_stats = None
    route_error = "No risk-aware route remains after applying the selected closure."

base_stats = route_stats(R, baseline)
map_obj, roads_shown = build_map(R, node_ids, edge_risks, baseline, safer,
                                 start, destination, closed_edge)

st.subheader("Live evacuation network")
st.caption("Green/yellow/red roads show a relative terrain-and-scenario susceptibility index, not calibrated flood probability. Blue = shortest-time route; purple = risk-aware A* route.")
st_folium(map_obj, height=570, use_container_width=True, returned_objects=[])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Directed road nodes", f"{R.number_of_nodes():,}")
m2.metric("Directed road edges", f"{R.number_of_edges():,}")
m3.metric("Major road segments displayed", f"{roads_shown:,}")
m4.metric("Scenario rainfall", f"{rainfall_mm} mm", "Hypothetical")

st.subheader("Route comparison")
if route_error:
    st.error(route_error)
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline distance", f"{base_stats['distance_km']:.2f} km")
    c2.metric("Baseline travel estimate", f"{base_stats['minutes']:.1f} min")
    c3.metric("Baseline susceptibility", f"{base_stats['risk']:.2f} / 1.00")
    d1, d2, d3 = st.columns(3)
    d1.metric("A* route distance", f"{safe_stats['distance_km']:.2f} km")
    d2.metric("A* travel estimate", f"{safe_stats['minutes']:.1f} min")
    d3.metric("A* susceptibility", f"{safe_stats['risk']:.2f} / 1.00")

    table = pd.DataFrame([
        {"Route": "Shortest-time baseline", "Distance (km)": base_stats["distance_km"],
         "Estimated time (min)": base_stats["minutes"], "Weighted susceptibility": base_stats["risk"]},
        {"Route": "Risk-aware A*", "Distance (km)": safe_stats["distance_km"],
         "Estimated time (min)": safe_stats["minutes"], "Weighted susceptibility": safe_stats["risk"]},
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    if close_road:
        st.success("Road closure simulation applied. The A* route was recalculated after removing the selected directed road segment.")
    else:
        st.info("Enable the road closure control in the sidebar to demonstrate adaptive rerouting.")

st.subheader("Historical rainfall context")
rain_file = DATA / "rainfall" / "seasons" / "2020" / "daily_rainfall.csv"
if rain_file.exists():
    rain = pd.read_csv(rain_file, parse_dates=["date"])
    daily = rain.groupby("date", as_index=False)["rainfall_mm"].mean()
    st.caption(
        f"CHIRPS v3 historical pilot: {daily['date'].min().date()} to "
        f"{daily['date'].max().date()}. Regional grid-cell mean; not street-level rainfall."
    )
    fig = px.line(daily, x="date", y="rainfall_mm",
                  labels={"date": "Date", "rainfall_mm": "Rainfall (mm/day)"},
                  title="Historical rainfall pilot — 2020")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Historical rainfall pilot CSV not found.")

st.subheader("Method used in this prototype")
st.markdown("""
1. Road graph is derived from the regional OpenStreetMap extract and retains directed driving edges.
2. Each directed road segment is sampled against the processed SRTM relative-elevation, TPI and slope rasters.
3. A relative terrain index is formed using 65% low-relative-elevation, 25% low-TPI and 10% slope components.
4. The rainfall control scales this index as a hypothetical scenario; it is not a live forecast.
5. Baseline routing minimizes estimated travel time. Risk-aware A* minimizes  
   `travel time × (1 + risk avoidance weight × susceptibility index)`.
6. A road closure removes one directed edge for the rerouting demonstration.
""")
st.caption(
    "Limitations: the terrain grid is approximately 250 m; the relative-elevation raster is a local terrain proxy, "
    "not a calibrated HAND or flood-depth model. Travel times use estimated road-class speeds where OSM maxspeed "
    "is unavailable. Demo destinations are synthetic and are not verified public shelters. "
    "This prototype is not suitable for operational emergency response."
)

'@
Set-Content -Path (Join-Path $root "evacuation_app.py") -Value $content -Encoding UTF8
Write-Host "Created evacuation_app.py. Existing app.py was not changed." -ForegroundColor Green
Write-Host "Launch with: .\venv\Scripts\python.exe -m streamlit run evacuation_app.py"
