import geopandas as gpd
import networkx as nx

edges_file = "data/processed/ujjain/osm/ujjain_road_edges.gpkg"
graph_file = "data/processed/ujjain/osm/ujjain_directed_roads.graphml"

roads = gpd.read_file(edges_file)
G = nx.read_graphml(graph_file)

print("===== ROAD MAPPING DIAGNOSTICS =====")

road_pairs = {
    (str(row.u), str(row.v))
    for row in roads.itertuples()
}

reverse_pairs = {
    (v, u) for u, v in road_pairs
}

graph_pairs = {
    (str(u), str(v)) for u, v in G.edges()
}

print("Physical road segments:", len(roads))
print("Directed graph edges:", G.number_of_edges())

print("Matching forward edges:",
      len(graph_pairs & road_pairs))

print("Matching reverse edges:",
      len(graph_pairs & reverse_pairs))

print("One-way distribution:")
print(roads["oneway"].fillna("not tagged").value_counts())

print("\nRoad geometry validity:")
print("Valid:", roads.geometry.is_valid.sum())
print("Invalid:", (~roads.geometry.is_valid).sum())

print("\nSample graph edge:")
u, v, data = next(iter(G.edges(data=True)))
print("Source:", u)
print("Target:", v)
print("Attributes:", data)

