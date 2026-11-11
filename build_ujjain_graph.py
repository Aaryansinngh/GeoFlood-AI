from pathlib import Path
import re
import networkx as nx
import geopandas as gpd

BASE = Path("data/processed/ujjain/osm")
NODES_FILE = BASE / "ujjain_road_nodes.gpkg"
EDGES_FILE = BASE / "ujjain_road_edges.gpkg"
OUTPUT = BASE / "ujjain_directed_roads.graphml"

print("Loading extracted road data...")
nodes = gpd.read_file(NODES_FILE)
edges = gpd.read_file(EDGES_FILE)

G = nx.DiGraph()

for row in nodes.itertuples():
    node_id = int(row.id)
    G.add_node(
        node_id,
        x=float(row.lon),
        y=float(row.lat)
    )

DEFAULT_SPEEDS = {
    "motorway": 70,
    "trunk": 60,
    "primary": 45,
    "secondary": 35,
    "tertiary": 30,
    "residential": 25,
    "unclassified": 25,
    "living_street": 15,
    "service": 15
}

def parse_speed(value, highway):
    if value is not None:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if match:
            speed = float(match.group())
            if speed > 0:
                if "mph" in str(value).lower():
                    speed *= 1.60934
                return min(speed, 120)

    if isinstance(highway, list):
        highway = highway[0] if highway else "unclassified"

    return DEFAULT_SPEEDS.get(str(highway), 25)

def is_oneway(value):
    if isinstance(value, bool):
        return value, False

    text = str(value).strip().lower()

    if text in ("yes", "true", "1"):
        return True, False

    if text == "-1":
        return True, True

    return False, False

def add_segment(u, v, row, reverse=False):
    length = float(row["length"])
    highway = row.get("highway", "unclassified")
    speed = parse_speed(row.get("maxspeed"), highway)

    travel_time = length / (speed * 1000 / 3600)

    geometry = row.geometry
    if geometry is not None and not geometry.is_empty:
        coords = list(geometry.coords)
        if reverse:
            coords.reverse()
        wkt = geometry.wkt if not reverse else geometry.__class__(coords).wkt
    else:
        wkt = ""

    name = row.get("name")
    if isinstance(name, list):
        name = ", ".join(str(x) for x in name)

    attributes = {
        "length_m": length,
        "speed_kph": speed,
        "travel_s": travel_time,
        "highway": str(highway),
        "name": str(name) if name is not None else "Unnamed road",
        "geometry_wkt": wkt,
        "flood_risk": 0.0
    }

    # Keep the fastest parallel segment for this directed node pair.
    if not G.has_edge(u, v) or travel_time < G[u][v]["travel_s"]:
        G.add_edge(u, v, **attributes)

print("Building directed road network...")

for _, row in edges.iterrows():
    u = int(row["u"])
    v = int(row["v"])

    if u not in G or v not in G:
        continue

    oneway, reverse_only = is_oneway(row.get("oneway"))

    if reverse_only:
        add_segment(v, u, row, reverse=True)
    else:
        add_segment(u, v, row)

        if not oneway:
            add_segment(v, u, row, reverse=True)

print("Calculating network connectivity...")
components = list(nx.weakly_connected_components(G))
largest = max(components, key=len) if components else set()

BASE.mkdir(parents=True, exist_ok=True)
nx.write_graphml(G, OUTPUT)

print("\n========== DIRECTED GRAPH COMPLETE ==========")
print("Directed nodes:", G.number_of_nodes())
print("Directed edges:", G.number_of_edges())
print("Weakly connected components:", len(components))
print("Largest component:", len(largest))
print("Largest component coverage:",
      round(100 * len(largest) / G.number_of_nodes(), 2), "%")
print("Saved:", OUTPUT.resolve())
print("\nGraph ready for flood-aware A* routing.")
