import math
import pytest

from src.evacuation.road_graph import RoadEdge, RoadGraph
from src.evacuation.planner import (
    calculate_route_statistics,
    plan_evacuation,
)


def create_sample_graph():
    graph = RoadGraph()

    edges = [
        RoadEdge("START", "A", 2, 30, 0.1),
        RoadEdge("A", "SHELTER", 2, 30, 0.1),
        RoadEdge("START", "B", 3, 30, 0.05),
        RoadEdge("B", "SHELTER", 3, 30, 0.05),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


def test_planner_finds_route():
    plan = plan_evacuation(create_sample_graph(), "START", "SHELTER")
    assert plan.found
    assert plan.path[0] == "START"
    assert plan.path[-1] == "SHELTER"


def test_planner_returns_route_statistics():
    plan = plan_evacuation(create_sample_graph(), "START", "SHELTER")
    assert plan.total_distance_km > 0
    assert plan.estimated_travel_time_minutes > 0
    assert plan.average_flood_risk >= 0


def test_planner_selects_lower_risk_route():
    graph = RoadGraph()

    graph.add_edge(RoadEdge("START", "A", 2, 30, 0.8))
    graph.add_edge(RoadEdge("A", "SHELTER", 2, 30, 0.8))

    graph.add_edge(RoadEdge("START", "B", 3, 30, 0.05))
    graph.add_edge(RoadEdge("B", "SHELTER", 3, 30, 0.05))

    plan = plan_evacuation(
        graph, "START", "SHELTER", risk_weight=5.0
    )

    assert plan.found
    assert plan.path == ["START", "B", "SHELTER"]
    assert plan.average_flood_risk == pytest.approx(0.05)


def test_planner_detects_unreachable_shelter():
    graph = RoadGraph()
    graph.add_edge(RoadEdge("START", "A", 2))
    graph.add_node("SHELTER")

    plan = plan_evacuation(graph, "START", "SHELTER")

    assert not plan.found
    assert plan.path == []
    assert math.isinf(plan.total_cost)


def test_planner_rejects_invalid_start():
    with pytest.raises(ValueError):
        plan_evacuation(create_sample_graph(), "UNKNOWN", "SHELTER")


def test_planner_rejects_invalid_destination():
    with pytest.raises(ValueError):
        plan_evacuation(create_sample_graph(), "START", "UNKNOWN")


def test_planner_rejects_negative_risk_weight():
    with pytest.raises(ValueError):
        plan_evacuation(
            create_sample_graph(),
            "START",
            "SHELTER",
            risk_weight=-1,
        )


def test_planner_reports_blocked_roads():
    graph = create_sample_graph()
    graph.add_edge(RoadEdge("START", "CLOSED", 1, is_closed=True))

    plan = plan_evacuation(graph, "START", "SHELTER")

    assert plan.blocked_roads_count == 1


def test_route_statistics():
    statistics = calculate_route_statistics(
        create_sample_graph(),
        ["START", "A", "SHELTER"],
    )

    assert statistics["total_distance_km"] == 4
    assert statistics["estimated_travel_time_minutes"] == 8
    assert statistics["average_flood_risk"] == pytest.approx(0.1)


def test_start_equals_destination():
    graph = RoadGraph()
    graph.add_node("START")

    plan = plan_evacuation(graph, "START", "START")

    assert plan.found
    assert plan.path == ["START"]
    assert plan.total_cost == 0
    assert plan.total_distance_km == 0


def test_planner_returns_search_information():
    plan = plan_evacuation(create_sample_graph(), "START", "SHELTER")

    assert plan.expanded_nodes > 0
    assert len(plan.search_steps) > 0


def test_planner_returns_route_segments():
    plan = plan_evacuation(create_sample_graph(), "START", "SHELTER")

    assert len(plan.route_segments) == 2
    assert "flood_risk" in plan.route_segments[0]
    assert "distance_km" in plan.route_segments[0]

def test_distance_weighted_flood_exposure():
    from src.evacuation.planner import calculate_route_statistics

    graph = RoadGraph()

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        flood_risk=1.0,
    ))

    graph.add_edge(RoadEdge(
        source="B",
        target="C",
        distance_km=9,
        flood_risk=0.0,
    ))

    stats = calculate_route_statistics(graph, ["A", "B", "C"])

    assert stats["average_flood_risk"] == pytest.approx(0.5)
    assert stats["distance_weighted_flood_risk"] == pytest.approx(0.1)


def test_planner_statistics_use_selected_parallel_segment():
    graph = RoadGraph()

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        speed_kmph=60,
        flood_risk=0.9,
        segment_id=1,
    ))

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=2,
        speed_kmph=60,
        flood_risk=0.0,
        segment_id=2,
    ))

    plan = plan_evacuation(
        graph,
        "A",
        "B",
        risk_weight=2.0,
    )

    assert plan.found
    assert plan.route_segments[0]["segment_id"] == 2
    assert plan.total_distance_km == pytest.approx(2.0)
    assert plan.distance_weighted_flood_risk == pytest.approx(0.0)
