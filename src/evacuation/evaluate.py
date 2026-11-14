"""Evaluation utilities for evacuation routes."""


def compare_routes(fastest, risk_aware):
    """Compare fastest and risk-aware route results."""

    return {
        "fastest_time_min": fastest["travel_time_min"],
        "risk_aware_time_min": risk_aware["travel_time_min"],
        "time_difference_min": (
            risk_aware["travel_time_min"]
            - fastest["travel_time_min"]
        ),
        "fastest_distance_km": fastest["distance_km"],
        "risk_aware_distance_km": risk_aware["distance_km"],
        "fastest_exposure": fastest["mean_edge_exposure"],
        "risk_aware_exposure": risk_aware["mean_edge_exposure"],
    }
