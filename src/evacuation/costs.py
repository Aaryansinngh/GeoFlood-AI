"""Routing cost functions for GeoFlood-AI."""

import math


def edge_travel_time(data):
    return max(float(data.get("travel_time_min", 1.0)), 0.001)


def edge_risk(data):
    value = float(data.get("historical_exposure", 0.0))
    return min(max(value, 0.0), 0.99)


def edge_cost(data, risk_lambda=10.0):
    """Travel time plus historical exposure penalty."""
    travel_time = edge_travel_time(data)
    risk = edge_risk(data)

    penalty = risk_lambda * (-math.log(1.0 - risk))

    return travel_time + penalty
