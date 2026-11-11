"""
GeoFlood AI - Custom A* Pathfinding Engine

Core algorithm for flood-aware evacuation routing.

f(n) = g(n) + h(n)

g(n): Accumulated route cost.
h(n): Estimated remaining cost.
f(n): Total estimated route cost.
"""

import heapq
import math

from dataclasses import dataclass, field
from itertools import count
from typing import (
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Mapping,
    Optional,
    TypeVar,
)


Node = TypeVar("Node", bound=Hashable)


@dataclass
class AStarResult(Generic[Node]):

    path: List[Node]

    total_cost: float

    expanded_nodes: List[Node] = field(default_factory=list)

    g_scores: Dict[Node, float] = field(default_factory=dict)

    parents: Dict[Node, Node] = field(default_factory=dict)

    steps: List[dict] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.path)

    @property
    def nodes_explored(self) -> int:
        return len(self.expanded_nodes)


def astar_search(
    graph: Mapping[Node, Mapping[Node, float]],
    start: Node,
    goal: Node,
    heuristic: Optional[Callable[[Node, Node], float]] = None,
) -> AStarResult[Node]:

    """
    Execute custom A* search on a weighted graph.

    Parameters
    ----------
    graph:
        Adjacency mapping:
        {node: {neighbor: edge_cost}}

    start:
        Starting node.

    goal:
        Destination node.

    heuristic:
        Admissible estimate of remaining cost.
        Defaults to zero (Dijkstra-equivalent search).

    Returns
    -------
    AStarResult:
        Optimal path under the supplied edge costs
        when the heuristic is admissible.
    """

    if heuristic is None:
        heuristic = lambda node, target: 0.0

    def get_h(node):

        value = float(heuristic(node, goal))

        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Invalid heuristic value for {node}: {value}"
            )

        return value

    if start not in graph:
        raise ValueError(f"Start node {start} not in graph")

    if goal not in graph:
        raise ValueError(f"Goal node {goal} not in graph")

    if start == goal:

        return AStarResult(
            path=[start],
            total_cost=0.0,
            expanded_nodes=[start],
            g_scores={start: 0.0},
        )

    # Priority queue:
    # (f_score, h_score, insertion_counter, node, g_at_insertion)

    open_heap = []

    counter = count()

    g_scores = {start: 0.0}

    parents = {}

    expanded_nodes = []

    steps = []

    closed_scores = {}

    start_h = get_h(start)

    heapq.heappush(
        open_heap,
        (
            start_h,
            start_h,
            next(counter),
            start,
            0.0,
        ),
    )

    EPSILON = 1e-12

    while open_heap:

        (
            current_f,
            current_h,
            _,
            current,
            queued_g,
        ) = heapq.heappop(open_heap)

        # Ignore outdated queue entries.

        if queued_g > g_scores.get(current, math.inf) + EPSILON:
            continue

        # Ignore nodes already expanded at an equal or lower cost.

        if queued_g >= closed_scores.get(current, math.inf) - EPSILON:
            continue

        current_g = g_scores[current]

        closed_scores[current] = current_g

        expanded_nodes.append(current)

        steps.append({
            "node": current,
            "g": current_g,
            "h": current_h,
            "f": current_f,
            "expanded": len(expanded_nodes),
        })

        # Destination reached.

        if current == goal:

            path = [goal]

            while path[-1] != start:

                path.append(parents[path[-1]])

            path.reverse()

            return AStarResult(
                path=path,
                total_cost=current_g,
                expanded_nodes=expanded_nodes,
                g_scores=dict(g_scores),
                parents=dict(parents),
                steps=steps,
            )

        # Explore neighboring road intersections.

        for neighbor, raw_cost in graph.get(current, {}).items():

            edge_cost = float(raw_cost)

            if not math.isfinite(edge_cost) or edge_cost < 0:

                raise ValueError(
                    f"Invalid edge cost: {current} -> {neighbor}"
                )

            tentative_g = current_g + edge_cost

            previous_g = g_scores.get(neighbor, math.inf)

            if tentative_g + EPSILON < previous_g:

                g_scores[neighbor] = tentative_g

                parents[neighbor] = current

                neighbor_h = get_h(neighbor)

                neighbor_f = tentative_g + neighbor_h

                heapq.heappush(
                    open_heap,
                    (
                        neighbor_f,
                        neighbor_h,
                        next(counter),
                        neighbor,
                        tentative_g,
                    ),
                )

    # No feasible path exists.

    return AStarResult(
        path=[],
        total_cost=math.inf,
        expanded_nodes=expanded_nodes,
        g_scores=dict(g_scores),
        parents=dict(parents),
        steps=steps,
    )