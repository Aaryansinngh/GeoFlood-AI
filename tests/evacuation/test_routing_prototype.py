import networkx as nx

from src.evacuation.routing import astar_route, dijkstra_route


G = nx.MultiDiGraph()

G.add_node(1, x=0, y=0)
G.add_node(2, x=1000, y=0)
G.add_node(3, x=2000, y=0)
G.add_node(4, x=1000, y=1000)

G.add_edge(
    1, 2, length_m=1000,
    travel_time_min=2,
    historical_exposure=0.8,
)
G.add_edge(
    2, 3, length_m=1000,
    travel_time_min=2,
    historical_exposure=0.8,
)
G.add_edge(
    1, 4, length_m=1500,
    travel_time_min=3,
    historical_exposure=0.0,
)
G.add_edge(
    4, 3, length_m=1500,
    travel_time_min=3,
    historical_exposure=0.0,
)

fastest = astar_route(G, 1, 3, risk_lambda=0)
safer = astar_route(G, 1, 3, risk_lambda=10)
baseline = dijkstra_route(G, 1, 3, risk_lambda=10)

print("Fastest route:", fastest)
print("Risk-aware route:", safer)
print("Dijkstra route:", baseline)

assert fastest["path"] == [1, 2, 3]
assert safer["path"] == [1, 4, 3]
assert safer["path"] == baseline
print("\nAll routing checks passed.")
