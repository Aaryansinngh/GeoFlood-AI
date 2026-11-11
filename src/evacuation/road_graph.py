"""
GeoFlood AI - Flood-Aware Road Graph

Supports:
- Directed road networks
- Parallel road segments
- Dynamic road closures and reopening
- Flood-aware travel costs
"""

import math

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Mapping, Optional, Tuple


Node = Hashable


@dataclass
class RoadEdge:
    """Represents one directed road segment."""

    source: Node
    target: Node

    distance_km: float
    speed_kmph: float = 30.0
    flood_risk: float = 0.0
    accessibility_penalty: float = 0.0

    is_closed: bool = False
    is_flooded: bool = False

    road_name: Optional[str] = None
    segment_id: Optional[Hashable] = None

    @property
    def travel_time_minutes(self) -> float:
        return (self.distance_km / self.speed_kmph) * 60.0

    @property
    def risk_category(self) -> str:
        if self.flood_risk < 0.35:
            return "Low"
        if self.flood_risk < 0.70:
            return "Moderate"
        return "High"

    def validate(self):
        if not math.isfinite(self.distance_km) or self.distance_km <= 0:
            raise ValueError("Road distance must be positive and finite")

        if not math.isfinite(self.speed_kmph) or self.speed_kmph <= 0:
            raise ValueError("Road speed must be positive and finite")

        if (
            not math.isfinite(self.flood_risk)
            or not 0 <= self.flood_risk <= 1
        ):
            raise ValueError("Flood risk must be between 0 and 1")

        if (
            not math.isfinite(self.accessibility_penalty)
            or self.accessibility_penalty < 0
        ):
            raise ValueError("Accessibility penalty must be non-negative")


@dataclass
class RoadGraph:
    """Directed road graph retaining individual road segments."""

    adjacency: Dict[Node, Dict[Node, float]] = field(default_factory=dict)

    # Active representative edge for each directed node pair.
    edges: Dict[Tuple[Node, Node], RoadEdge] = field(default_factory=dict)

    # All road segments, including parallel and currently blocked segments.
    edge_options: Dict[Tuple[Node, Node], List[RoadEdge]] = field(
        default_factory=dict
    )

    blocked_edges: List[Tuple[Node, Node]] = field(default_factory=list)

    def add_node(self, node: Node):
        self.adjacency.setdefault(node, {})

    def _refresh_pair(self, key: Tuple[Node, Node]):
        """Refresh active topology and blocked status for one node pair."""

        source, target = key
        options = self.edge_options.get(key, [])

        active = [
            edge for edge in options
            if not edge.is_closed and not edge.is_flooded
        ]

        if active:
            representative = min(
                active,
                key=lambda edge: edge.travel_time_minutes,
            )

            self.edges[key] = representative
            self.adjacency[source][target] = (
                representative.travel_time_minutes
            )

            if key in self.blocked_edges:
                self.blocked_edges.remove(key)

        else:
            self.edges.pop(key, None)
            self.adjacency[source].pop(target, None)

            if options and key not in self.blocked_edges:
                self.blocked_edges.append(key)

    def add_edge(self, edge: RoadEdge):
        """Add a road segment without discarding parallel segments."""

        edge.validate()

        self.add_node(edge.source)
        self.add_node(edge.target)

        key = (edge.source, edge.target)

        options = self.edge_options.setdefault(key, [])

        if edge not in options:
            options.append(edge)

        self._refresh_pair(key)

    def update_edge_status(
        self,
        source: Node,
        target: Node,
        segment_id: Optional[Hashable] = None,
        is_closed: Optional[bool] = None,
        is_flooded: Optional[bool] = None,
    ):
        """
        Update road conditions without rebuilding the graph.

        If segment_id is omitted, all parallel segments for this
        directed node pair are updated.
        """

        if is_closed is None and is_flooded is None:
            raise ValueError("Specify at least one road-status update")

        key = (source, target)
        options = self.edge_options.get(key)

        if not options:
            raise KeyError(f"Road segment {source} -> {target} does not exist")

        matching = [
            edge for edge in options
            if segment_id is None or edge.segment_id == segment_id
        ]

        if not matching:
            raise KeyError(
                f"Road segment ID {segment_id} was not found "
                f"for {source} -> {target}"
            )

        for edge in matching:
            if is_closed is not None:
                edge.is_closed = bool(is_closed)

            if is_flooded is not None:
                edge.is_flooded = bool(is_flooded)

        self._refresh_pair(key)

    def get_edge_options(
        self,
        source: Node,
        target: Node,
    ) -> List[RoadEdge]:
        """Return all currently usable parallel road segments."""

        return [
            edge
            for edge in self.edge_options.get((source, target), [])
            if not edge.is_closed and not edge.is_flooded
        ]

    def get_edge(
        self,
        source: Node,
        target: Node,
        risk_weight: float = 2.0,
        accessibility_weight: float = 1.0,
    ) -> Optional[RoadEdge]:
        """Return the usable parallel segment with the lowest combined cost."""

        options = self.get_edge_options(source, target)

        if not options:
            return None

        return min(
            options,
            key=lambda edge: self.get_edge_cost(
                edge,
                risk_weight=risk_weight,
                accessibility_weight=accessibility_weight,
            ),
        )

    def get_edge_cost(
        self,
        edge: RoadEdge,
        risk_weight: float = 2.0,
        accessibility_weight: float = 1.0,
    ) -> float:
        """Calculate combined evacuation cost in minutes."""

        if (
            not math.isfinite(risk_weight)
            or not math.isfinite(accessibility_weight)
            or risk_weight < 0
            or accessibility_weight < 0
        ):
            raise ValueError("Cost weights must be finite and non-negative")

        travel_time = edge.travel_time_minutes

        flood_penalty = travel_time * edge.flood_risk * risk_weight

        access_penalty = (
            travel_time
            * edge.accessibility_penalty
            * accessibility_weight
        )

        return travel_time + flood_penalty + access_penalty

    def build_weighted_graph(
        self,
        risk_weight: float = 2.0,
        accessibility_weight: float = 1.0,
    ) -> Dict[Node, Dict[Node, float]]:
        """Build the weighted adjacency graph used by A*."""

        weighted_graph = {
            node: {} for node in self.adjacency
        }

        for (source, target), options in self.edge_options.items():
            active = [
                edge for edge in options
                if not edge.is_closed and not edge.is_flooded
            ]

            if not active:
                continue

            weighted_graph[source][target] = min(
                self.get_edge_cost(
                    edge,
                    risk_weight=risk_weight,
                    accessibility_weight=accessibility_weight,
                )
                for edge in active
            )

        return weighted_graph

    def get_node_count(self) -> int:
        return len(self.adjacency)

    def get_edge_count(self) -> int:
        """Number of active directed node-to-node connections."""
        return len(self.edges)

    def get_segment_count(self) -> int:
        """Number of individually active road segments."""
        return sum(
            len(self.get_edge_options(source, target))
            for source, target in self.edge_options
        )

    def get_blocked_edge_count(self) -> int:
        """Number of directed node pairs with no usable segment."""
        return len(self.blocked_edges)


def build_road_graph(
    road_segments: List[RoadEdge],
    risk_weight: float = 2.0,
    accessibility_weight: float = 1.0,
) -> Tuple[RoadGraph, Dict[Node, Dict[Node, float]]]:
    """Build road metadata and its weighted A* graph."""

    road_graph = RoadGraph()

    for edge in road_segments:
        road_graph.add_edge(edge)

    weighted_graph = road_graph.build_weighted_graph(
        risk_weight=risk_weight,
        accessibility_weight=accessibility_weight,
    )

    return road_graph, weighted_graph
