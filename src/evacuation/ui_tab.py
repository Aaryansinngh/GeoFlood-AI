"""Guwahati evacuation intelligence: 2D selection + 3D digital-twin view.

The 3D view uses OSM road/building geometry and extruded historical flood-screening
cells. Extrusion is a visual index only; it is NOT water depth or a forecast.
"""

import base64
import io
import json
import os

import folium
import numpy as np
import osmnx as ox
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image
from pyproj import Transformer
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import shape, Point, LineString
from shapely.ops import transform as shapely_transform

from src.evacuation.graph_builder import build_road_graph, attach_historical_exposure
from src.evacuation.routing import astar_route
from src.evacuation.costs import edge_cost
from src.evacuation.shelters import get_candidate_shelters
from src.evacuation.risk_mapper import build_guwahati_risk_map

RISK_GRAPH_CACHE = "data/cache/evacuation/guwahati_drive_5000m_risk.graphml"
BUILDINGS_CACHE = "data/cache/evacuation/guwahati_buildings.geojson"
CENTRE = [26.1445, 91.7362]
MAX_3D_ROADS = 5000
MAX_FLOOD_CELLS = 1800


@st.cache_resource(show_spinner="Loading Guwahati road network...")
def load_evacuation_network():
    if os.path.exists(RISK_GRAPH_CACHE):
        return ox.load_graphml(RISK_GRAPH_CACHE)
    graph = attach_historical_exposure(build_road_graph(radius_m=5000))
    os.makedirs(os.path.dirname(RISK_GRAPH_CACHE), exist_ok=True)
    ox.save_graphml(graph, RISK_GRAPH_CACHE)
    return graph


@st.cache_resource(show_spinner="Loading historical flood observations...")
def load_historical_risk():
    # Pass the verified project-relative paths explicitly. The risk mapper
    # does not infer these paths from guwahati.yaml when called with no args.
    event_paths = [
        "data/raw/DFO_4259_From_20150602_to_20150629.tif",
        "data/raw/DFO_4288_From_20150813_to_20150911.tif",
    ]
    missing = [path for path in event_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing historical DFO raster(s): " + ", ".join(missing)
        )
    return build_guwahati_risk_map(historical_event_paths=event_paths)


@st.cache_resource(show_spinner="Loading candidate facilities...")
def load_shelters():
    return get_candidate_shelters()


@st.cache_data(show_spinner="Loading 3D building footprints from OpenStreetMap...")
def load_building_records():
    os.makedirs(os.path.dirname(BUILDINGS_CACHE), exist_ok=True)
    if os.path.exists(BUILDINGS_CACHE):
        try:
            gdf = __import__("geopandas").read_file(BUILDINGS_CACHE)
            return _buildings_to_records(gdf)
        except Exception:
            pass
    try:
        gdf = ox.features_from_point(
            (CENTRE[0], CENTRE[1]), tags={"building": True}, dist=5000
        )
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not gdf.empty:
            gdf = gdf.to_crs("EPSG:4326")
            gdf.to_file(BUILDINGS_CACHE, driver="GeoJSON")
        return _buildings_to_records(gdf)
    except Exception:
        return []


def _buildings_to_records(gdf):
    records = []
    if gdf is None or gdf.empty:
        return records
    for _, row in gdf.head(3500).iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polygons:
            ring = [[float(x), float(y)] for x, y in list(poly.exterior.coords)]
            if len(ring) >= 4:
                height = row.get("height", None)
                levels = row.get("building:levels", None)
                try:
                    height_m = float(str(height).replace(" m", ""))
                except Exception:
                    try:
                        height_m = float(levels) * 3.2
                    except Exception:
                        height_m = 9.0
                records.append({"polygon": ring, "height": max(3.0, min(height_m, 90.0))})
    return records


def _to_projected(graph, lat, lon):
    t = Transformer.from_crs("EPSG:4326", graph.graph.get("crs", "EPSG:32646"), always_xy=True)
    return t.transform(lon, lat)


def _largest_road_component(graph):
    """Return the largest weakly connected component for reliable road snapping."""
    import networkx as nx
    if graph.number_of_nodes() == 0:
        raise RuntimeError("The downloaded road graph is empty.")
    components = list(nx.weakly_connected_components(graph))
    if not components:
        raise RuntimeError("No connected road component is available.")
    largest = max(components, key=len)
    # Preserve directed edge attributes while excluding isolated fragments.
    return graph.subgraph(largest).copy()


def _nearest_node(graph, lat, lon):
    # Snap only to the largest connected road component, not a nearby isolated
    # driveway/service-road fragment that may have no path to the other point.
    import networkx as nx
    component = _largest_road_component(graph)
    x, y = _to_projected(graph, lat, lon)
    node = ox.distance.nearest_nodes(component, X=x, Y=y)
    return node


def _calculate_route_pair(
    graph, start, end, penalty, edge_penalties=None, blocked_edges=None
):
    """Calculate fastest baseline and polygon-aware risk/closure route."""
    import networkx as nx

    edge_penalties = edge_penalties or {}
    blocked_edges = blocked_edges or set()

    def solve(network):
        fastest = astar_route(network, start, end, risk_lambda=0)
        risk_aware = astar_route(
            network, start, end, risk_lambda=penalty,
            edge_penalties=edge_penalties, blocked_edges=blocked_edges
        )
        return fastest, risk_aware

    try:
        if not nx.has_path(graph, start, end):
            raise nx.NetworkXNoPath(f"No directed path from {start} to {end}")
        fastest, risk_aware = solve(graph)
        return {
            "fastest": fastest,
            "risk_aware": risk_aware,
            "routing_note": "Directed road routing; OSM one-way restrictions respected.",
        }
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Clearly labelled fallback only; it does not claim legal one-way routing.
        undirected = graph.to_undirected(as_view=False)
        if not nx.has_path(undirected, start, end):
            raise RuntimeError(
                "No route exists for these points, even when one-way restrictions "
                "are ignored. Select points closer to the connected road network."
            )
        fastest, risk_aware = solve(undirected)
        return {
            "fastest": fastest,
            "risk_aware": risk_aware,
            "routing_note": (
                "Fallback route: one-way restrictions were ignored because the "
                "directed OSM graph had no path. Verify before real-world use."
            ),
        }


def _selected_region_edges(graph, geojson):
    """Return directed road-edge pairs intersecting a user-drawn WGS84 polygon."""
    if not geojson:
        return set()

    geometry = geojson.get("geometry", geojson)
    polygon_wgs84 = shape(geometry)
    if polygon_wgs84.is_empty or polygon_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
        return set()

    graph_crs = graph.graph.get("crs", "EPSG:32646")
    project = Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
    polygon_projected = shapely_transform(
        lambda x, y, z=None: project.transform(x, y), polygon_wgs84
    )

    impacted = set()
    is_multi = graph.is_multigraph()
    edge_iter = graph.edges(keys=True, data=True) if is_multi else (
        (u, v, None, data) for u, v, data in graph.edges(data=True)
    )
    for u, v, key, data in edge_iter:
        geom = data.get("geometry")
        if geom is None:
            nu, nv = graph.nodes[u], graph.nodes[v]
            geom = LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])])
        if geom.intersects(polygon_projected):
            impacted.add((u, v))
    return impacted


def _route_coordinates(graph, path, risk_lambda=0.0):
    """Draw the same parallel edge selected by the route cost, not always the fastest edge."""
    t = Transformer.from_crs(graph.graph.get("crs", "EPSG:32646"), "EPSG:4326", always_xy=True)
    coords = []
    for u, v in zip(path[:-1], path[1:]):
        edges = graph.get_edge_data(u, v)
        if not edges:
            continue
        data = min(
            edges.values(),
            key=lambda d: edge_cost(d, risk_lambda=risk_lambda)
        )
        geom = data.get("geometry")
        points = list(geom.coords) if geom is not None else [
            (graph.nodes[u]["x"], graph.nodes[u]["y"]),
            (graph.nodes[v]["x"], graph.nodes[v]["y"]),
        ]
        for x, y in points:
            lon, lat = t.transform(x, y)
            point = [float(lon), float(lat)]
            if not coords or coords[-1] != point:
                coords.append(point)
    return coords


def _route_frame(graph, route, color, name, risk_lambda=0.0):
    if not route or not route.get("path"):
        return []
    coords = _route_coordinates(graph, route["path"], risk_lambda=risk_lambda)
    return [{"path": coords, "name": name, "color": color}] if len(coords) > 1 else []


def _flood_records(risk_map, stage):
    arr = np.asarray(risk_map.historical_flood, dtype=float)
    valid = np.isfinite(arr) & (arr > 0)
    permanent = getattr(risk_map, "permanent_water", None)
    if permanent is not None:
        valid &= ~np.asarray(permanent, dtype=bool)
    # The slider reveals historically flooded land cells, excluding permanent water.
    threshold = 1.0 - float(stage) / 100.0
    selected = valid & (arr >= max(0.001, threshold))
    rows, cols = np.where(selected)
    if len(rows) > MAX_FLOOD_CELLS:
        stride = int(np.ceil(len(rows) / MAX_FLOOD_CELLS))
        rows, cols = rows[::stride], cols[::stride]
    to_geo = Transformer.from_crs(risk_map.crs, "EPSG:4326", always_xy=True)
    records = []
    for r, c in zip(rows, cols):
        x, y = risk_map.transform * (float(c) + 0.5, float(r) + 0.5)
        lon, lat = to_geo.transform(x, y)
        index = float(arr[r, c])
        records.append({
            "lon": float(lon), "lat": float(lat),
            # Deliberately exaggerated display height for visibility; not water depth.
            "elevation": 45 + 110 * index,
            "index": index,
            "radius": 85,
        })
    return records, int(selected.sum())


def _road_records(graph):
    t = Transformer.from_crs(graph.graph.get("crs", "EPSG:32646"), "EPSG:4326", always_xy=True)
    records = []
    # Graph is a MultiDiGraph; sample unique directed edges and cap rendering load.
    edges = list(graph.edges(keys=True, data=True))
    if len(edges) > MAX_3D_ROADS:
        stride = int(np.ceil(len(edges) / MAX_3D_ROADS))
        edges = edges[::stride]
    for u, v, key, data in edges:
        geom = data.get("geometry")
        points = list(geom.coords) if geom is not None else [
            (graph.nodes[u]["x"], graph.nodes[u]["y"]),
            (graph.nodes[v]["x"], graph.nodes[v]["y"]),
        ]
        path = []
        for x, y in points:
            lon, lat = t.transform(x, y)
            path.append([float(lon), float(lat)])
        if len(path) >= 2:
            records.append({"path": path})
    return records


def _make_2d_map(graph, risk_map, shelters, stage, origin, destination, routes=None):
    arr = np.asarray(risk_map.historical_flood, dtype=float)
    valid = np.isfinite(arr)
    permanent = getattr(risk_map, "permanent_water", None)
    if permanent is not None:
        valid &= ~np.asarray(permanent, dtype=bool)
    threshold = 1.0 - float(stage) / 100.0
    flooded = valid & (arr > 0) & (arr >= max(0.001, threshold))
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[flooded] = [235, 55, 35, 175]
    from rasterio.warp import transform_bounds
    west, south, east, north = transform_bounds(
        risk_map.crs, "EPSG:4326", *risk_map.bounds, densify_pts=21
    )
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    m = folium.Map(location=CENTRE, zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    folium.Circle(
        location=CENTRE, radius=5000, color="#00d4ff", weight=3,
        fill=False, tooltip="Guwahati study boundary — 5 km radius"
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{encoded}",
        bounds=[[south, west], [north, east]], opacity=0.78,
        name="Historical flood observations", interactive=False, zindex=3
    ).add_to(m)
    # Re-draw the user's selected flood polygon after every Streamlit rerun.
    # st_folium's Draw layer is browser-side state; rebuilding the Folium map
    # after route calculation otherwise makes the selected region disappear.
    selected_region = st.session_state.get("evac_selected_region_geojson")
    if selected_region:
        region_geometry = selected_region.get("geometry", selected_region)
        if region_geometry and region_geometry.get("type") in ("Polygon", "MultiPolygon"):
            folium.GeoJson(
                data=region_geometry,
                name="Selected flood-affected region",
                style_function=lambda _feature: {
                    "color": "#ffb000",
                    "weight": 4,
                    "opacity": 1.0,
                    "fillColor": "#ff3b30",
                    "fillOpacity": 0.28,
                },
                tooltip="Selected flood-affected region — applied to route calculation",
            ).add_to(m)
    folium.Marker(origin, tooltip="Origin", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(destination, tooltip="Destination", icon=folium.Icon(color="green")).add_to(m)
    if routes:
        for key, color, label in [
            ("fastest", "#e74c3c", "Fastest"),
            ("risk_aware", "#16a34a", "Historical-exposure-aware"),
        ]:
            route = routes.get(key)
            if route:
                route_lambda = 0.0 if key == "fastest" else float(
                    st.session_state.get("evac_route_penalty_compact", 40)
                )
                coords = [[lat, lon] for lon, lat in _route_coordinates(
                    graph, route["path"], risk_lambda=route_lambda
                )]
                folium.PolyLine(coords, color=color, weight=5, tooltip=label).add_to(m)
    if shelters is not None and not shelters.empty:
        t = Transformer.from_crs("EPSG:32646", "EPSG:4326", always_xy=True)
        for _, row in shelters.iterrows():
            geom = row.geometry
            if geom is not None and not geom.is_empty:
                lon, lat = t.transform(geom.x, geom.y)
                folium.CircleMarker([lat, lon], radius=4, color="#7b2cbf",
                                    fill=True, tooltip="OSM candidate facility").add_to(m)
    folium.LayerControl().add_to(m)
    # Draw a polygon or rectangle around the area the user wants to inspect.
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False, "circle": False, "marker": False,
            "circlemarker": False,
            "polygon": {"allowIntersection": False, "showArea": True},
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    return m, int(flooded.sum())


def _render_3d(graph, risk_map, buildings, stage, origin, destination, routes):
    flood, count = _flood_records(risk_map, stage)
    roads = _road_records(graph)
    layers = []

    if buildings:
        layers.append(pdk.Layer(
            "PolygonLayer", data=buildings, get_polygon="polygon",
            get_elevation="height", elevation_scale=1, extruded=True,
            get_fill_color=[145, 157, 174, 190],
            get_line_color=[70, 80, 95, 100], line_width_min_pixels=0.5,
            pickable=True, auto_highlight=True
        ))

    if roads:
        layers.append(pdk.Layer(
            "PathLayer", data=roads, get_path="path",
            get_color=[235, 165, 35, 230], get_width=4,
            width_min_pixels=2, pickable=True, auto_highlight=True
        ))

    if flood:
        # Ground-level markers ensure flood cells remain visible even when the
        # browser/device has limited support for 3D column rendering.
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=flood, get_position="[lon, lat]",
            get_radius="radius", radius_min_pixels=5, radius_max_pixels=18,
            get_fill_color=[0, 145, 255, 205], get_line_color=[5, 65, 160, 255],
            line_width_min_pixels=1, stroked=True, pickable=True,
            auto_highlight=True
        ))
        layers.append(pdk.Layer(
            "ColumnLayer", data=flood, get_position="[lon, lat]",
            get_elevation="elevation", elevation_scale=1,
            radius=85, get_fill_color=[0, 115, 255, 220],
            disk_resolution=8, extruded=True, pickable=True,
            auto_highlight=True
        ))

    markers = [
        {"lon": origin[1], "lat": origin[0], "kind": "Origin"},
        {"lon": destination[1], "lat": destination[0], "kind": "Destination"},
    ]
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=markers, get_position="[lon, lat]",
        get_radius=65, radius_min_pixels=6, radius_max_pixels=14,
        get_fill_color=[20, 210, 120, 255], pickable=True
    ))

    route_data = []
    if routes:
        route_data += _route_frame(
            graph, routes.get("fastest"), [235, 65, 65, 255], "Fastest", risk_lambda=0.0
        )
        route_data += _route_frame(
            graph, routes.get("risk_aware"), [20, 210, 110, 255],
            "Historical exposure aware",
            risk_lambda=float(st.session_state.get("evac_route_penalty_compact", 40))
        )
    if route_data:
        layers.append(pdk.Layer(
            "PathLayer", data=route_data, get_path="path", get_color="color",
            get_width=7, width_min_pixels=3, pickable=True
        ))

    view = pdk.ViewState(
        latitude=CENTRE[0], longitude=CENTRE[1], zoom=13.2,
        pitch=58, bearing=-12
    )
    deck = pdk.Deck(
        layers=layers, initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{name} {kind} {index}"}
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(
        f"3D view: {len(buildings):,} OSM building footprints, {len(roads):,} sampled road segments, "
        f"and {count:,} historically flooded raster cells. Blue discs mark flood cells; raised blue "
        "columns emphasize them. Column height is an illustrative visual encoding, NOT measured water depth."
    )
    if count == 0:
        st.warning(
            "No flooded cells passed the current stage filter. Move the flood slider to 100 "
            "to display every cell with a positive historical flood index."
        )


def _analyze_drawn_flood_region(risk_map, drawing):
    """Summarize historical flood observations inside a user-drawn polygon."""
    if not drawing or not isinstance(drawing, dict):
        return None
    geometry = drawing.get("geometry", drawing)
    try:
        polygon = shape(geometry)
    except Exception:
        return None
    if polygon.is_empty or polygon.geom_type not in ("Polygon", "MultiPolygon"):
        return None

    arr = np.asarray(risk_map.historical_flood, dtype=float)
    valid = np.isfinite(arr)
    permanent = getattr(risk_map, "permanent_water", None)
    if permanent is not None:
        valid &= ~np.asarray(permanent, dtype=bool)

    to_geo = Transformer.from_crs(risk_map.crs, "EPSG:4326", always_xy=True)
    inside_rows, inside_cols = [], []
    for row, col in zip(*np.where(valid)):
        x, y = risk_map.transform * (float(col) + 0.5, float(row) + 0.5)
        lon, lat = to_geo.transform(x, y)
        if polygon.covers(Point(lon, lat)):
            inside_rows.append(row)
            inside_cols.append(col)

    if not inside_rows:
        return {"valid_cells": 0, "flooded_cells": 0, "flood_fraction": 0.0,
                "mean_index": None, "max_index": None, "area_km2": 0.0}

    values = arr[np.asarray(inside_rows), np.asarray(inside_cols)]
    flooded = values > 0
    # The risk grid is projected in metres; use its pixel dimensions for an
    # approximate raster-cell area, not a surveyed boundary area.
    transform = risk_map.transform
    cell_area_km2 = abs(float(transform.a) * float(transform.e) -
                        float(transform.b) * float(transform.d)) / 1_000_000.0
    return {
        "valid_cells": int(len(values)),
        "flooded_cells": int(flooded.sum()),
        "flood_fraction": float(flooded.mean()),
        "mean_index": float(values.mean()),
        "max_index": float(values.max()),
        "area_km2": float(len(values) * cell_area_km2),
    }


def render_evacuation_tab():
    """Compact presentation-first evacuation demo interface."""
    st.title("Guwahati Evacuation Intelligence")
    st.caption(
        "Interactive flood screening, 3D city view and evacuation routing. "
        "Flood observations are from two 2015 DFO events—not live forecasts."
    )

    try:
        graph = load_evacuation_network()
        risk_map = load_historical_risk()
        # The cached GraphML may contain stale or default-zero exposure values.
        # Re-sample its edges against the loaded historical DFO grid before routing.
        if not graph.graph.get("_dfo_exposure_refreshed", False):
            graph = attach_historical_exposure(graph, risk_map=risk_map)
            graph.graph["_dfo_exposure_refreshed"] = True
        shelters = load_shelters()
    except Exception as exc:
        st.error(f"Unable to load evacuation layers: {exc}")
        return

    if "evac_origin" not in st.session_state:
        st.session_state["evac_origin"] = [26.1445, 91.7362]
    if "evac_destination" not in st.session_state:
        st.session_state["evac_destination"] = [26.1550, 91.7500]

    origin = st.session_state["evac_origin"]
    destination = st.session_state["evac_destination"]
    routes = st.session_state.get("evac_selected_routes")

    # Keep the main screen focused: one stage control and two view tabs.
    control_col, play_col = st.columns([5, 1])
    with control_col:
        stage = st.slider(
            "Historical flood reveal", 0, 100, 100, 5,
            key="evac_flood_stage_compact",
            help="Reveals more cells marked flooded in the 2015 DFO observations."
        )
    with play_col:
        st.write("")
        play_animation = st.button(
            "▶ Play", key="evac_flood_play_compact",
            use_container_width=True
        )

    map2d, flooded_count = _make_2d_map(
        graph, risk_map, shelters, stage, origin, destination, routes
    )
    tab_map, tab_3d = st.tabs(["2D MAP & SELECTION", "3D DIGITAL TWIN"])

    with tab_map:
        st.caption("Study area: 5 km radius (10 km diameter) around central Guwahati. Select an action, then click the map.")
        click_mode = st.radio(
            "Map action",
            ["Set origin", "Set destination", "Inspect flood location"],
            horizontal=True, key="evac_map_selection_mode_compact",
            label_visibility="collapsed"
        )
        state = st_folium(
            map2d, height=520, use_container_width=True,
            key="guwahati_evacuation_map_compact",
            returned_objects=["last_clicked", "all_drawings"]
        )

        clicked = state.get("last_clicked") if state else None
        if clicked and clicked.get("lat") is not None:
            point = [float(clicked["lat"]), float(clicked["lng"])]
            signature = (round(point[0], 6), round(point[1], 6), click_mode)
            if st.session_state.get("_last_evac_click_compact") != signature:
                st.session_state["_last_evac_click_compact"] = signature
                if click_mode == "Set origin":
                    st.session_state["evac_origin"] = point
                    st.session_state.pop("evac_selected_routes", None)
                    st.rerun()
                elif click_mode == "Set destination":
                    st.session_state["evac_destination"] = point
                    st.session_state.pop("evac_selected_routes", None)
                    st.rerun()
                else:
                    x, y = _to_projected(graph, *point)
                    import rasterio
                    row, col = rasterio.transform.rowcol(risk_map.transform, x, y)
                    arr = risk_map.historical_flood
                    if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1] and np.isfinite(arr[row, col]):
                        st.info(
                            f"Historical screening index: {float(arr[row, col]):.3f} "
                            "(not a current flood probability)."
                        )
                    else:
                        st.info("No valid historical observation at this location.")

        # Region selection is available but tucked away until needed.
        with st.expander("Analyze a selected flood region", expanded=False):
            st.caption("Draw a polygon/rectangle on the map, or enter rectangular coordinates.")
            drawings = state.get("all_drawings", []) if state else []
            if drawings:
                latest_drawing = drawings[-1]
                latest_geometry = latest_drawing.get("geometry", latest_drawing)
                if latest_geometry and latest_geometry.get("type") in ("Polygon", "MultiPolygon"):
                    region_signature = json.dumps(latest_geometry, sort_keys=True)
                    if st.session_state.get("_evac_region_signature") != region_signature:
                        st.session_state["evac_selected_region_geojson"] = latest_drawing
                        st.session_state["_evac_region_signature"] = region_signature
                        st.session_state.pop("evac_selected_routes", None)
            if drawings:
                region_stats = _analyze_drawn_flood_region(risk_map, drawings[-1])
                if region_stats and region_stats["valid_cells"]:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Flooded cells", f'{region_stats["flooded_cells"]:,}')
                    m2.metric("Flooded share", f'{region_stats["flood_fraction"] * 100:.1f}%')
                    m3.metric("Approx. area", f'{region_stats["area_km2"]:.2f} km²')
                else:
                    st.info("The drawn region does not overlap valid historical raster cells.")

            with st.form("evac_region_bounds_form_compact"):
                st.caption("Or define a rectangle using latitude/longitude bounds.")
                a, b = st.columns(2)
                south = a.number_input("South", 26.10, 26.30, 26.13, format="%.5f")
                north = b.number_input("North", 26.10, 26.30, 26.16, format="%.5f")
                c, d = st.columns(2)
                west = c.number_input("West", 91.65, 91.85, 91.72, format="%.5f")
                east = d.number_input("East", 91.65, 91.85, 91.75, format="%.5f")
                analyze = st.form_submit_button("Analyze region")
            if analyze:
                if south >= north or west >= east:
                    st.error("Check the coordinate bounds.")
                else:
                    from shapely.geometry import box
                    selected_rectangle = {
                        "geometry": box(west, south, east, north).__geo_interface__
                    }
                    st.session_state["evac_selected_region_geojson"] = selected_rectangle
                    st.session_state["_evac_region_signature"] = json.dumps(
                        selected_rectangle["geometry"], sort_keys=True
                    )
                    st.session_state.pop("evac_selected_routes", None)
                    stats = _analyze_drawn_flood_region(risk_map, selected_rectangle)
                    if stats and stats["valid_cells"]:
                        x1, x2, x3 = st.columns(3)
                        x1.metric("Flooded cells", f'{stats["flooded_cells"]:,}')
                        x2.metric("Flooded share", f'{stats["flood_fraction"] * 100:.1f}%')
                        x3.metric("Approx. area", f'{stats["area_km2"]:.2f} km²')
                    else:
                        st.warning("No valid historical raster cells in that rectangle.")

        st.caption(
            f"Origin: {origin[0]:.5f}, {origin[1]:.5f}  ·  "
            f"Destination: {destination[0]:.5f}, {destination[1]:.5f}"
        )

    with tab_3d:
        st.caption("Rotate, tilt and zoom. Blue columns show historical flood-screening cells.")
        buildings = load_building_records()
        if play_animation:
            import time
            frame = st.empty()
            for animation_stage in range(0, 101, 10):
                with frame.container():
                    _render_3d(
                        graph, risk_map, buildings, animation_stage,
                        origin, destination, routes
                    )
                    st.caption(f"Historical flood reveal: {animation_stage}%")
                time.sleep(0.16)
        else:
            _render_3d(graph, risk_map, buildings, stage, origin, destination, routes)

    # Keep route controls in a compact, predictable footer.
    with st.expander("Evacuation route comparison", expanded=True):
        penalty = st.slider(
            "Historical exposure penalty", 0, 100, 40,
            key="evac_route_penalty_compact",
            help="Higher values more strongly penalize roads with sampled historical DFO flood exposure."
        )
        region_mode = st.radio(
            "How should selected flood-affected roads be treated?",
            ["High risk penalty", "Simulate road closure"],
            horizontal=True, key="evac_region_route_mode"
        )
        selected_region = st.session_state.get("evac_selected_region_geojson")
        if selected_region:
            st.success("A selected flood region is active and will influence the next route calculation.")
        else:
            st.info("Draw a polygon/rectangle on the 2D map to make the selected region influence routing.")
        if st.button(
            "Calculate evacuation routes", type="primary",
            use_container_width=True, key="evac_calculate_routes_compact"
        ):
            try:
                start_node = _nearest_node(graph, *origin)
                end_node = _nearest_node(graph, *destination)
                impacted_edges = _selected_region_edges(graph, selected_region)
                region_edge_penalties = {}
                region_blocked_edges = set()
                if region_mode == "High risk penalty":
                    # Add a substantial generalized-cost penalty to every road
                    # intersecting the selected polygon (in addition to DFO exposure).
                    region_edge_penalties = {edge: 25.0 for edge in impacted_edges}
                else:
                    region_blocked_edges = impacted_edges

                calculated = _calculate_route_pair(
                    graph, start_node, end_node, penalty,
                    edge_penalties=region_edge_penalties,
                    blocked_edges=region_blocked_edges,
                )
                calculated["selected_region_edges"] = len(impacted_edges)
                calculated["region_mode"] = region_mode
                calculated["selected_region_edge_pairs"] = list(impacted_edges)
                st.session_state["evac_selected_routes"] = calculated
                st.rerun()
            except Exception as exc:
                st.error(f"Route calculation failed: {exc}")

        routes = st.session_state.get("evac_selected_routes")
        if routes:
            c1, c2 = st.columns(2)
            c1.metric("Fastest route", f'{routes["fastest"]["travel_time_min"]:.1f} min')
            c2.metric(
                "Exposure-aware route",
                f'{routes["risk_aware"]["travel_time_min"]:.1f} min'
            )
            e1, e2 = st.columns(2)
            e1.metric(
                "Fastest route historical exposure",
                f'{routes["fastest"].get("mean_edge_exposure", 0.0):.3f}'
            )
            e2.metric(
                "Exposure-aware route historical exposure",
                f'{routes["risk_aware"].get("mean_edge_exposure", 0.0):.3f}'
            )
            if routes["risk_aware"].get("mean_edge_exposure", 0.0) >= routes["fastest"].get("mean_edge_exposure", 0.0):
                st.warning(
                    "The selected exposure penalty did not produce a lower-exposure route. "
                    "This can happen when the road network has limited alternatives or sampled coverage is sparse."
                )
            st.metric(
                "Road segments intersecting selected region",
                f'{routes.get("selected_region_edges", 0):,}'
            )
            st.caption(
                f"Selected-region treatment: {routes.get('region_mode', 'Not applied')}. "
                "The selected polygon is converted to intersecting road-edge constraints."
            )
            if routes.get("routing_note"):
                st.caption(routes["routing_note"])
        else:
            st.caption("Choose origin and destination on the map, then calculate routes.")

    sampled_edges = sum(
        1 for _, _, data in graph.edges(data=True)
        if data.get("risk_source") == "DFO_HISTORICAL_SCREENING"
    )
    total_edges = graph.number_of_edges()
    st.caption(
        f"Historical exposure sampled on {sampled_edges:,}/{total_edges:,} road edges. "
        "Unsampled roads are unknown—not confirmed flood-free."
    )
    st.caption(
        f"{flooded_count:,} historical flood cells shown · "
        f"{len(shelters) if shelters is not None else 0} OSM candidate facilities. "
        "Candidate facilities are not verified official shelters; routes are not certified safe."
    )
