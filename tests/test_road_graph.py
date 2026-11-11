import pytest

from src.evacuation.road_graph import (
    RoadEdge,
    RoadGraph,
    build_road_graph,
)


def test_add_road_edge():

    graph = RoadGraph()

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=2,
        speed_kmph=30,
    )

    graph.add_edge(edge)

    assert graph.get_node_count() == 2
    assert graph.get_edge_count() == 1


def test_travel_time_calculation():

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=10,
        speed_kmph=30,
    )

    assert edge.travel_time_minutes == 20


def test_low_risk_classification():

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        flood_risk=0.2,
    )

    assert edge.risk_category == "Low"


def test_moderate_risk_classification():

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        flood_risk=0.5,
    )

    assert edge.risk_category == "Moderate"


def test_high_risk_classification():

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        flood_risk=0.9,
    )

    assert edge.risk_category == "High"


def test_closed_road_excluded():

    graph = RoadGraph()

    graph.add_edge(
        RoadEdge(
            source="A",
            target="B",
            distance_km=2,
            is_closed=True,
        )
    )

    assert graph.get_edge_count() == 0
    assert graph.get_blocked_edge_count() == 1


def test_flooded_road_excluded():

    graph = RoadGraph()

    graph.add_edge(
        RoadEdge(
            source="A",
            target="B",
            distance_km=2,
            is_flooded=True,
        )
    )

    assert graph.get_edge_count() == 0


def test_high_risk_road_has_higher_cost():

    safe = RoadEdge(
        source="A",
        target="B",
        distance_km=5,
        flood_risk=0.1,
    )

    risky = RoadEdge(
        source="B",
        target="C",
        distance_km=5,
        flood_risk=0.9,
    )

    graph = RoadGraph()

    safe_cost = graph.get_edge_cost(safe)
    risky_cost = graph.get_edge_cost(risky)

    assert risky_cost > safe_cost


def test_invalid_flood_risk():

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=2,
        flood_risk=1.5,
    )

    with pytest.raises(ValueError):
        edge.validate()


def test_invalid_distance():

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=-2,
    )

    with pytest.raises(ValueError):
        edge.validate()


def test_build_weighted_graph():

    segments = [
        RoadEdge(
            source="A",
            target="B",
            distance_km=5,
            speed_kmph=30,
            flood_risk=0.1,
        ),
        RoadEdge(
            source="B",
            target="C",
            distance_km=3,
            speed_kmph=30,
            flood_risk=0.2,
        ),
    ]

    road_graph, weighted_graph = build_road_graph(segments)

    assert road_graph.get_node_count() == 3
    assert road_graph.get_edge_count() == 2

    assert weighted_graph["A"]["B"] > 0
    assert weighted_graph["B"]["C"] > 0


def test_negative_cost_weight_rejected():

    graph = RoadGraph()

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=2,
    )

    with pytest.raises(ValueError):
        graph.get_edge_cost(edge, risk_weight=-1)
def test_dynamic_road_closure_and_reopening():
    graph = RoadGraph()

    edge = RoadEdge(
        source="A",
        target="B",
        distance_km=2,
        segment_id=1,
    )

    graph.add_edge(edge)

    assert graph.get_edge_count() == 1

    graph.update_edge_status("A", "B", segment_id=1, is_closed=True)

    assert graph.get_edge_count() == 0
    assert graph.get_blocked_edge_count() == 1

    graph.update_edge_status("A", "B", segment_id=1, is_closed=False)

    assert graph.get_edge_count() == 1
    assert graph.get_blocked_edge_count() == 0


def test_parallel_road_segments_are_preserved():
    graph = RoadGraph()

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        speed_kmph=60,
        segment_id=1,
    ))

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=2,
        speed_kmph=60,
        segment_id=2,
    ))

    assert graph.get_segment_count() == 2
    assert graph.get_edge_count() == 1


def test_closing_one_parallel_segment_keeps_other_available():
    graph = RoadGraph()

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=1,
        speed_kmph=60,
        segment_id=1,
    ))

    graph.add_edge(RoadEdge(
        source="A",
        target="B",
        distance_km=2,
        speed_kmph=60,
        segment_id=2,
    ))

    graph.update_edge_status(
        "A", "B",
        segment_id=1,
        is_closed=True,
    )

    selected = graph.get_edge("A", "B")

    assert selected is not None
    assert selected.segment_id == 2
    assert graph.get_blocked_edge_count() == 0


def test_parallel_segment_selection_considers_flood_risk():
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

    selected = graph.get_edge("A", "B", risk_weight=2.0)

    assert selected.segment_id == 2
