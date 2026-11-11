import math
import pytest

from src.evacuation.astar_engine import astar_search


def test_basic_astar_route():

    graph = {
        "S": {"A": 1, "B": 4},
        "A": {"G": 2},
        "B": {"G": 1},
        "G": {},
    }

    result = astar_search(graph, "S", "G")

    assert result.found
    assert result.path == ["S", "A", "G"]
    assert result.total_cost == 3


def test_risk_weighted_routing():

    graph = {
        "S": {"A": 1, "B": 3},
        "A": {"G": 100},
        "B": {"G": 2},
        "G": {},
    }

    result = astar_search(graph, "S", "G")

    assert result.path == ["S", "B", "G"]
    assert result.total_cost == 5


def test_unreachable_destination():

    graph = {
        "S": {"A": 1},
        "A": {},
        "G": {},
    }

    result = astar_search(graph, "S", "G")

    assert not result.found
    assert math.isinf(result.total_cost)


def test_same_start_and_goal():

    graph = {
        "S": {},
    }

    result = astar_search(graph, "S", "S")

    assert result.path == ["S"]
    assert result.total_cost == 0


def test_invalid_start():

    graph = {
        "S": {},
        "G": {},
    }

    with pytest.raises(ValueError):

        astar_search(graph, "X", "G")


def test_negative_edge_cost():

    graph = {
        "S": {"A": -5},
        "A": {},
        "G": {},
    }

    with pytest.raises(ValueError):

        astar_search(graph, "S", "G")


def test_astar_with_heuristic():

    graph = {
        "S": {"A": 2, "B": 5},
        "A": {"G": 3},
        "B": {"G": 1},
        "G": {},
    }

    heuristic = {
        "S": 3,
        "A": 2,
        "B": 1,
        "G": 0,
    }

    result = astar_search(
        graph,
        "S",
        "G",
        heuristic=lambda node, goal: heuristic[node],
    )

    assert result.path == ["S", "A", "G"]
    assert result.total_cost == 5