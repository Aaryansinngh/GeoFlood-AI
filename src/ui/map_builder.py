"""
src/ui/map_builder.py

Folium map construction utilities for GeoFlood AI Phase 4.
Builds the layered Folium map with susceptibility, observation, SAR, and uncertainty layers.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import folium
from folium.raster_layers import ImageOverlay


SUSCEP_CMAP = [
    (0.00, 0.35, [10, 180, 140, 0]),        # Low (transparent at bottom)
    (0.35, 0.35, [10, 180, 140, 170]),      # Low
    (0.35, 0.70, [245, 158, 11, 190]),      # Moderate
    (0.70, 1.00, [239, 68, 68, 215]),       # High
]


def build_susceptibility_rgba(
    ev1: pd.DataFrame,
    h: int,
    w: int,
    prob_col: str,
    opacity_scale: float = 1.0,
) -> np.ndarray:
    """
    Build a pixel-accurate RGBA array for the susceptibility heatmap.
    Missing-data cells remain fully transparent.
    Permanent-water cells use a distinct blue.
    """
    probs = ev1[prob_col].values.reshape((h, w))
    perm  = ev1["is_perm_water"].values.reshape((h, w))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Permanent water: deep blue
    rgba[perm == 1] = [2, 132, 199, int(200 * opacity_scale)]

    # High risk: red
    hi = (probs >= 0.70) & (perm == 0)
    rgba[hi] = [239, 68, 68, int(210 * opacity_scale)]

    # Moderate risk: amber
    mo = (probs >= 0.35) & (probs < 0.70) & (perm == 0)
    rgba[mo] = [245, 158, 11, int(190 * opacity_scale)]

    # Low risk: teal
    lo = (probs < 0.35) & (perm == 0) & (~np.isnan(probs))
    rgba[lo] = [16, 185, 129, int(165 * opacity_scale)]

    # No-data: fully transparent
    nd = np.isnan(probs) & (perm == 0)
    rgba[nd] = [0, 0, 0, 0]

    return rgba


def build_observed_rgba(ev1: pd.DataFrame, h: int, w: int) -> np.ndarray:
    """RGBA for historical MODIS observed flood extent."""
    obs  = ev1["flood_label"].values.reshape((h, w))
    perm = ev1["is_perm_water"].values.reshape((h, w))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[obs == 1.0] = [0, 212, 255, 210]      # Cyan-blue: flooded
    rgba[perm == 1]  = [2, 100, 180, 180]       # Darker blue: perennial river
    return rgba


def build_disagreement_rgba(ev1: pd.DataFrame, h: int, w: int, threshold: float = 0.15) -> np.ndarray:
    """
    Compute pixel-level model disagreement between NN and RF predictions.
    Disagreement = |prob_neural_network - prob_random_forest|.
    Displayed on a yellow-red diverging scale.
    """
    perm = ev1["is_perm_water"].values.reshape((h, w))
    nn_col = "prob_neural_network" if "prob_neural_network" in ev1.columns else "susceptibility_prob"
    rf_col = "prob_random_forest"  if "prob_random_forest"  in ev1.columns else "susceptibility_prob"

    nn_p = ev1[nn_col].values.reshape((h, w))
    rf_p = ev1[rf_col].values.reshape((h, w))

    diff = np.abs(nn_p.astype(float) - rf_p.astype(float))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[perm == 1] = [0, 0, 0, 0]

    valid = (perm == 0) & (~np.isnan(diff))

    # Scale disagreement 0->1 where 1 = diff >= 0.5
    norm = np.clip(diff / 0.5, 0.0, 1.0)

    # Low disagreement: yellow (255, 237, 74)
    # High disagreement: crimson (220, 38, 38)
    r = (255 + (220 - 255) * norm).astype(np.uint8)
    g = (237 + (38  - 237) * norm).astype(np.uint8)
    b = (74  + (38  - 74)  * norm).astype(np.uint8)
    a = (160 * norm).astype(np.uint8)
    a = np.clip(a, 0, 255)

    # Only show significant disagreement
    sig = valid & (diff >= threshold)
    rgba[sig, 0] = r[sig]
    rgba[sig, 1] = g[sig]
    rgba[sig, 2] = b[sig]
    rgba[sig, 3] = np.minimum(a[sig] + 60, 220)

    return rgba


def build_base_map(
    bounds: list,
    center_lat: float,
    center_lon: float,
    tiles: str = "CartoDB dark_matter",
) -> folium.Map:
    """Create a Folium map centred on the study area with dark basemap."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles=tiles,
        control_scale=True,
        prefer_canvas=True,
    )
    m.fit_bounds(bounds, padding=(10, 10))
    return m


def add_study_boundary(m: folium.Map, bounds: list, west, south, east, north) -> folium.Map:
    """Add the study-area boundary rectangle."""
    folium.Rectangle(
        bounds=bounds,
        color="#00d4ff",
        weight=2,
        fill=False,
        popup=folium.Popup(
            f"<b>Study Area</b><br>Kaziranga–Jorhat Corridor<br>"
            f"{west:.3f}°E – {east:.3f}°E<br>{south:.3f}°N – {north:.3f}°N",
            max_width=220,
        ),
        tooltip="GeoFlood AI Study Region",
    ).add_to(m)
    return m


def add_landmarks(m: folium.Map) -> folium.Map:
    """Add regional landmark markers."""
    landmarks = [
        ("Kaziranga National Park", 26.58, 93.17),
        ("Jorhat Corridor (E. Boundary)", 26.75, 94.18),
        ("Golaghat District HQ", 26.52, 93.97),
        ("Bokakhat Sub-Division", 26.63, 93.59),
        ("Tezpur Crossing (W. Boundary)", 26.63, 92.82),
    ]
    for name, lat, lon in landmarks:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#7dd3fc",
            fill=True,
            fill_color="#0d1e35",
            fill_opacity=0.8,
            popup=folium.Popup(f"<b>{name}</b><br>{lat:.2f}°N, {lon:.2f}°E", max_width=180),
            tooltip=name,
        ).add_to(m)
    return m


def add_selected_location(m: folium.Map, lat: float, lon: float, info: dict) -> folium.Map:
    """Add the user-selected analysis location marker."""
    cat = info.get("risk_category", "Unknown")
    prob = info.get("susceptibility_prob")
    prob_str = f"{prob:.1%}" if prob is not None else "N/A"

    color_map = {
        "High Risk": "red",
        "Moderate Risk": "orange",
        "Low Risk": "green",
        "Permanent Water": "blue",
    }
    marker_color = color_map.get(cat, "gray")

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>Selected Location</b><br>{lat:.4f}°N, {lon:.4f}°E<br>"
            f"<b>Category:</b> {cat}<br><b>Susceptibility:</b> {prob_str}",
            max_width=220,
        ),
        tooltip=f"📍 {lat:.4f}°N, {lon:.4f}°E — {cat}",
        icon=folium.Icon(color=marker_color, icon="crosshairs", prefix="fa"),
    ).add_to(m)
    return m
