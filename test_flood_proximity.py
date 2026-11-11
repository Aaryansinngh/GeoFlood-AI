import networkx as nx

from src.evacuation.flood_simulator import (
    calculate_edge_river_distances
)

G = nx.read_graphml(
    "data/processed/ujjain/osm/ujjain_directed_roads.graphml"
)

print("Calculating actual road geometry distances...")

distances, physical_mapping = calculate_edge_river_distances(G)

print("\n===== UPDATED RESULTS =====")

print("Physical roads:", len(physical_mapping))
print("Directed edges mapped:", len(distances))

for radius in [100, 250, 500, 750, 1000, 1500]:
    count = sum(d <= radius for d in distances.values())
    print(f"Within {radius} m: {count:,} directed edges")

print("\nMinimum distance:", round(min(distances.values()), 2))
print("Maximum distance:", round(max(distances.values()), 2))

print("\nMapping validation:",
      "PASSED" if len(distances) == G.number_of_edges() else "FAILED")

