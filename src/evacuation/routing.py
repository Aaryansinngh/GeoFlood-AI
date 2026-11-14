"""A* and Dijkstra evacuation routing."""

import heapq
import math
import networkx as nx

from src.evacuation.costs import edge_cost, edge_travel_time


def heuristic(graph, node, destination):
    """Admissible straight-line travel-time heuristic in minutes."""
    a = graph.nodes[node]
    b = graph.nodes[destination]

    distance = math.hypot(
        a["x"] - b["x"],
        a["y"] - b["y"],
    )

    # Maximum assumed speed: 130 km/h.
    return distance / 1000.0 / 130.0 * 60.0


def _best_edge(graph, u, v, weight):
    edges = graph.get_edge_data(u, v)

    if not edges:
        return None, float("inf")

    best_key = None
    best_cost = float("inf")

    for key, data in edges.items():
        cost = weight(data)

        if cost < best_cost:
            best_cost = cost
            best_key = key

    return best_key, best_cost


def astar_route(
    graph,
    origin,
    destination,
    risk_lambda=0.0,
    blocked_edges=None,
    edge_penalties=None,
):
    """Find a route using A*."""

    blocked_edges = blocked_edges or set()
    edge_penalties = edge_penalties or {}

    weight = lambda data: edge_cost(data, risk_lambda)

    queue = [
        (heuristic(graph, origin, destination), 0.0, origin)
    ]

    distance = {origin: 0.0}
    parent = {}
    expanded = 0

    while queue:
        _, current_cost, node = heapq.heappop(queue)

        if current_cost > distance.get(node, float("inf")):
            continue

        expanded += 1

        if node == destination:
            break

        for neighbor in graph.successors(node):
            if (node, neighbor) in blocked_edges:
                continue

            key, cost = _best_edge(
                graph,
                node,
                neighbor,
                weight,
            )

            if key is None:
                continue

            # Additional generalized cost for user-selected flood-region roads.
            # This is separate from the historical DFO exposure value.
            new_cost = current_cost + cost + float(
                edge_penalties.get((node, neighbor), 0.0)
            )

            if new_cost < distance.get(
                neighbor,
                float("inf"),
            ):
                distance[neighbor] = new_cost
                parent[neighbor] = (node, key)

                priority = new_cost + heuristic(
                    graph,
                    neighbor,
                    destination,
                )

                heapq.heappush(
                    queue,
                    (priority, new_cost, neighbor),
                )

    if destination not in distance:
        raise nx.NetworkXNoPath(
            "No evacuation route exists between these nodes."
        )

    path = [destination]
    current = destination

    while current != origin:
        current = parent[current][0]
        path.append(current)

    path.reverse()

    total_time = 0.0
    total_distance = 0.0
    total_exposure = 0.0

    for u, v in zip(path[:-1], path[1:]):
        key = parent[v][1]
        data = graph[u][v][key]

        total_time += edge_travel_time(data)
        total_distance += float(data.get("length_m", 0.0))
        total_exposure += float(
            data.get("historical_exposure", 0.0)
        )

    return {
        "path": path,
        "cost": distance[destination],
        "travel_time_min": total_time,
        "distance_km": total_distance / 1000.0,
        "mean_edge_exposure": (
            total_exposure / max(len(path) - 1, 1)
        ),
        "nodes_expanded": expanded,
        "risk_lambda": risk_lambda,
        "extra_region_penalty": sum(
            float(edge_penalties.get((u, v), 0.0))
            for u, v in zip(path[:-1], path[1:])
        ),
    }


def dijkstra_route(
    graph,
    origin,
    destination,
    risk_lambda=0.0,
):
    """NetworkX Dijkstra baseline."""
    def weight(u, v, data):
        if "travel_time_min" in data:
            return edge_cost(data, risk_lambda)

        # MultiGraph edge dictionaries.
        return min(
            edge_cost(edge, risk_lambda)
            for edge in data.values()
        )

    path = nx.shortest_path(
        graph,
        origin,
        destination,
        weight=weight,
        method="dijkstra",
    )

    return path
