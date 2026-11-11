"""
GeoFlood AI - Intelligent Evacuation Planner

Integrates:
    1. Flood-aware road graph
    2. Custom A* pathfinding engine
    3. Route cost optimization
    4. Distance-weighted flood exposure
    5. Evacuation route statistics
"""

import math

from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, List, Optional, TypeVar

from src.evacuation.astar_engine import (
    AStarResult,
    astar_search,
)

from src.evacuation.road_graph import RoadGraph


Node = TypeVar("Node", bound=Hashable)


@dataclass
class EvacuationPlan(Generic[Node]):
    """Contains the complete result of an evacuation search."""

    start: Node
    destination: Node

    path: List[Node] = field(default_factory=list)

    found: bool = False

    total_cost: float = math.inf

    total_distance_km: float = 0.0

    estimated_travel_time_minutes: float = 0.0

    average_flood_risk: float = 0.0

    distance_weighted_flood_risk: float = 0.0

    maximum_flood_risk: float = 0.0

    high_risk_segments: int = 0

    blocked_roads_count: int = 0

    expanded_nodes: int = 0

    search_steps: List[dict] = field(default_factory=list)

    route_segments: List[dict] = field(default_factory=list)

    @property
    def route_length(self) -> int:
        return len(self.path)

    @property
    def nodes_explored(self) -> int:
        return self.expanded_nodes


def calculate_route_statistics(
    road_graph: RoadGraph,
    path: List[Node],
    risk_weight: float = 2.0,
    accessibility_weight: float = 1.0,
) -> dict:
    """
    Calculate distance, travel time and flood-risk statistics.

    The distance-weighted risk represents the average flood-risk
    score experienced per kilometre of the selected route.
    """

    total_distance = 0.0
    total_travel_time = 0.0
    risk_values = []
    weighted_risk_sum = 0.0

    high_risk_segments = 0
    route_segments = []

    for index in range(len(path) - 1):

        source = path[index]
        target = path[index + 1]

        edge = road_graph.get_edge(
            source,
            target,
            risk_weight=risk_weight,
            accessibility_weight=accessibility_weight,
        )

        if edge is None:
            raise ValueError(
                f"Road metadata missing for route segment "
                f"{source} -> {target}"
            )

        total_distance += edge.distance_km
        total_travel_time += edge.travel_time_minutes

        risk_values.append(edge.flood_risk)

        weighted_risk_sum += edge.distance_km * edge.flood_risk

        if edge.flood_risk >= 0.70:
            high_risk_segments += 1

        route_segments.append({
            "source": source,
            "target": target,
            "segment_id": edge.segment_id,
            "road_name": edge.road_name,
            "distance_km": edge.distance_km,
            "travel_time_minutes": edge.travel_time_minutes,
            "flood_risk": edge.flood_risk,
            "risk_category": edge.risk_category,
        })

    average_risk = (
        sum(risk_values) / len(risk_values)
        if risk_values else 0.0
    )

    distance_weighted_risk = (
        weighted_risk_sum / total_distance
        if total_distance > 0 else 0.0
    )

    maximum_risk = (
        max(risk_values)
        if risk_values else 0.0
    )

    return {
        "total_distance_km": total_distance,
        "estimated_travel_time_minutes": total_travel_time,
        "average_flood_risk": average_risk,
        "distance_weighted_flood_risk": distance_weighted_risk,
        "maximum_flood_risk": maximum_risk,
        "high_risk_segments": high_risk_segments,
        "route_segments": route_segments,
    }


def plan_evacuation(
    road_graph: RoadGraph,
    start: Node,
    destination: Node,
    risk_weight: float = 2.0,
    accessibility_weight: float = 1.0,
    heuristic: Optional[Callable[[Node, Node], float]] = None,
) -> EvacuationPlan[Node]:
    """Generate an optimized flood-aware evacuation route."""

    if risk_weight < 0 or not math.isfinite(risk_weight):
        raise ValueError("Risk weight must be finite and non-negative")

    if (
        accessibility_weight < 0
        or not math.isfinite(accessibility_weight)
    ):
        raise ValueError(
            "Accessibility weight must be finite and non-negative"
        )

    if start not in road_graph.adjacency:
        raise ValueError(f"Starting node {start} does not exist")

    if destination not in road_graph.adjacency:
        raise ValueError(f"Destination node {destination} does not exist")

    # Step 1: Build the weighted graph.

    weighted_graph = road_graph.build_weighted_graph(
        risk_weight=risk_weight,
        accessibility_weight=accessibility_weight,
    )

    # Step 2: Execute custom A* search.

    search_result: AStarResult = astar_search(
        graph=weighted_graph,
        start=start,
        goal=destination,
        heuristic=heuristic,
    )

    # Step 3: Prepare the evacuation plan.

    plan = EvacuationPlan(
        start=start,
        destination=destination,
        path=search_result.path,
        found=search_result.found,
        total_cost=search_result.total_cost,
        expanded_nodes=search_result.nodes_explored,
        search_steps=search_result.steps,
        blocked_roads_count=road_graph.get_blocked_edge_count(),
    )

    # Step 4: Calculate route statistics.

    if not search_result.found:
        return plan

    statistics = calculate_route_statistics(
        road_graph,
        search_result.path,
        risk_weight=risk_weight,
        accessibility_weight=accessibility_weight,
    )

    plan.total_distance_km = statistics["total_distance_km"]

    plan.estimated_travel_time_minutes = (
        statistics["estimated_travel_time_minutes"]
    )

    plan.average_flood_risk = statistics["average_flood_risk"]

    plan.distance_weighted_flood_risk = (
        statistics["distance_weighted_flood_risk"]
    )

    plan.maximum_flood_risk = statistics["maximum_flood_risk"]

    plan.high_risk_segments = statistics["high_risk_segments"]

    plan.route_segments = statistics["route_segments"]

    return plan
