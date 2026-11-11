import pytest
import networkx as nx

from src.evacuation.osm_loader import load_osm_road_network


def test_invalid_latitude():

    with pytest.raises(ValueError):

        load_osm_road_network(
            latitude=100,
            longitude=75,
        )


def test_invalid_longitude():

    with pytest.raises(ValueError):

        load_osm_road_network(
            latitude=26,
            longitude=200,
        )


def test_invalid_radius():

    with pytest.raises(ValueError):

        load_osm_road_network(
            latitude=26,
            longitude=75,
            radius_m=-100,
        )


def test_invalid_flood_risk():

    with pytest.raises(ValueError):

        load_osm_road_network(
            latitude=26,
            longitude=75,
            default_flood_risk=1.5,
        )


def test_invalid_default_speed():

    with pytest.raises(ValueError):

        load_osm_road_network(
            latitude=26,
            longitude=75,
            default_speed_kmph=0,
        )