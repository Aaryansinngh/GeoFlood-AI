"""OSM road network builder for the Guwahati evacuation prototype."""

from pathlib import Path
import time

import osmnx as ox
import numpy as np
from shapely.geometry import LineString

from src.evacuation.risk_mapper import build_guwahati_risk_map

CACHE_DIR = Path("data/cache/evacuation")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.nchc.org.tw/api",
]


def configure_osmnx():
    """Configure OSMnx for slower or unreliable Overpass connections."""
    ox.settings.timeout = 300
    ox.settings.overpass_rate_limit = True
    ox.settings.requests_timeout = 300
    ox.settings.log_console = True


def build_road_graph(
    latitude=26.1445,
    longitude=91.7362,
    radius_m=5000,
    network_type="drive",
    force_download=False,
):
    """Download or load the projected Guwahati road network."""
    configure_osmnx()

    cache_path = CACHE_DIR / (
        f"guwahati_{network_type}_{int(radius_m)}m.graphml"
    )

    if cache_path.exists() and not force_download:
        print(f"Loading cached road graph: {cache_path}")
        graph = ox.load_graphml(cache_path)
    else:
        graph = None
        errors = []

        for endpoint in OVERPASS_ENDPOINTS:
            try:
                print(f"Trying Overpass endpoint: {endpoint}")
                ox.settings.overpass_url = endpoint

                graph = ox.graph.graph_from_point(
                    (latitude, longitude),
                    dist=radius_m,
                    network_type=network_type,
                    simplify=True,
                    retain_all=True,
                )
                break

            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")
                print(f"Endpoint failed: {exc}")
                time.sleep(3)

        if graph is None:
            raise RuntimeError(
                "Unable to download Guwahati road network.\n"
                + "\n".join(errors)
            )

        graph = ox.project_graph(graph, to_crs="EPSG:32646")

        graph = ox.routing.add_edge_speeds(
            graph,
            fallback=30,
        )
        graph = ox.routing.add_edge_travel_times(graph)

        ox.save_graphml(graph, cache_path)
        print(f"Road graph cached at: {cache_path}")

    if not graph.number_of_edges():
        raise ValueError("Downloaded road network contains no edges.")

    for u, v, key, data in graph.edges(keys=True, data=True):
        length = float(data.get("length", 1.0))
        speed = float(data.get("speed_kph", 30.0))

        data["length_m"] = length
        data["travel_time_min"] = (
            length / 1000.0 / max(speed, 1.0) * 60.0
        )
        data.setdefault("historical_exposure", 0.0)
        data.setdefault("risk_source", "NOT_SAMPLED")

    return graph


def attach_historical_exposure(
    graph,
    risk_map=None,
    sample_spacing_m=20,
):
    """Attach historical DFO screening exposure to road edges."""

    if risk_map is None:
        risk_map = build_guwahati_risk_map(
            historical_event_paths=[
                "data/raw/DFO_4259_From_20150602_to_20150629.tif",
                "data/raw/DFO_4288_From_20150813_to_20150911.tif",
            ]
        )

    for u, v, key, data in graph.edges(keys=True, data=True):
        geometry = data.get("geometry")

        if geometry is None:
            geometry = LineString([
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ])

        count = max(
            2,
            int(geometry.length / sample_spacing_m) + 1,
        )

        values = []

        for i in range(count):
            point = geometry.interpolate(
                i / (count - 1),
                normalized=True,
            )

            col_float, row_float = (
                ~risk_map.transform * (point.x, point.y)
            )
            col, row = int(col_float), int(row_float)

            if (
                0 <= row < risk_map.risk.shape[0]
                and 0 <= col < risk_map.risk.shape[1]
            ):
                value = risk_map.risk[row, col]

                if np.isfinite(value):
                    values.append(float(value))

        if values:
            data["historical_exposure"] = float(
                np.percentile(values, 90)
            )
            data["risk_source"] = "DFO_HISTORICAL_SCREENING"
        else:
            data["historical_exposure"] = 0.0
            data["risk_source"] = "NO_VALID_HISTORICAL_SAMPLE"

    return graph


def graph_summary(graph):
    edges = list(graph.edges(data=True))

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "total_road_km": round(
            sum(
                float(data.get("length_m", 0))
                for _, _, data in edges
            ) / 1000,
            2,
        ),
        "average_historical_exposure": round(
            float(np.mean([
                data.get("historical_exposure", 0)
                for _, _, data in edges
            ])),
            4,
        ) if edges else 0.0,
    }
