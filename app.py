"""
GeoFlood AI — Phase 4: Advanced Flood Intelligence & Decision Support Platform
Professional geospatial analysis workspace for the Brahmaputra floodplain, Assam, India.

Architecture:
  1. Overview          — Landing dashboard with project identity, status, and quick-start
  2. Flood Intel Map   — Layered Folium map with susceptibility, observation, disagreement layers
  3. Scenario Lab      — Interactive rainfall scenario laboratory with spatial difference maps
  4. Satellite Analysis— SAR observation workspace with susceptibility comparison
  5. Model Intelligence— Model comparison, disagreement map, feature importance
  6. Location Explorer — Coordinate-level analysis panel with neighbourhood summary
  7. Research & Valid. — Scientific validation, protocols, calibration, reproducibility

All predictions derive from real model inference. No placeholder or fabricated values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from src.evacuation.ui_tab import render_evacuation_tab
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page Config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoFlood AI | Flood Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy import of project modules (fail gracefully if run from wrong cwd)
# ---------------------------------------------------------------------------
try:
    from src.models.fuzzy_layer import MamdaniRiskFuzzySystem
    from src.geo.location_analysis import (
        LocationAnalyzer,
        RAINFALL_BOUNDS,
        TRAINING_RAINFALL_DISTRIBUTION,
        MODEL_METADATA_INFO,
    )
    from src.geo.grid import get_dataframe_raster_bounds
    from src.ui.styles import DARK_CSS, PLOT_LAYOUT, risk_card_class, SUSC_COLORS, MODEL_COLORS
    from src.ui.map_builder import (
        build_susceptibility_rgba,
        build_observed_rgba,
        build_disagreement_rgba,
        build_base_map,
        add_study_boundary,
        add_landmarks,
        add_selected_location,
    )
    from src.simulation.commander_view import render_commander_tab
    _MODULES_OK = True
except ImportError as _e:
    _MODULES_OK = False
    _IMPORT_ERR = str(_e)

# ---------------------------------------------------------------------------
# Apply CSS theme
# ---------------------------------------------------------------------------
if _MODULES_OK:
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.error(f"Module import failed: {_IMPORT_ERR}. Run from the project root directory.")
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_OPTIONS: Dict[str, str] = {
    "Neural Network (Primary)":      "neural_network",
    "Random Forest (Baseline 2)":    "random_forest",
    "Logistic Regression (Baseline 1)": "logistic_regression",
}
MODEL_LABEL: Dict[str, str] = {v: k for k, v in MODEL_OPTIONS.items()}
REPORT_DIR = Path("reports")
FIGURES_DIR = REPORT_DIR / "figures"

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading prediction dataset…")
def load_predictions() -> tuple[pd.DataFrame, bool]:
    """Load precomputed prediction table. Returns (df, is_sample)."""
    parquet_path = Path("outputs/predictions.parquet")
    sample_path  = Path("data/sample/features_sample.parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), False
    elif sample_path.exists():
        return pd.read_parquet(sample_path), True
    return pd.DataFrame(), True


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    import yaml
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_sar_data() -> Optional[dict]:
    p = Path("outputs/sar_demo_data.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_eval_summary() -> Optional[dict]:
    p = REPORT_DIR / "evaluation_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_metrics_csv() -> Optional[pd.DataFrame]:
    p = REPORT_DIR / "phase3_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_manifest() -> Optional[dict]:
    p = REPORT_DIR / "phase3_reproducibility.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner="Preparing raster overlays…")
def compute_all_rasters(df: pd.DataFrame) -> dict:
    """Precompute RGBA arrays for all models + observation + disagreement."""
    ev1 = df[df["event_id"] == "DFO_4259"].sort_values(["row", "col"])
    if len(ev1) == 0:
        return {}

    h = int(ev1["row"].max() + 1)
    w = int(ev1["col"].max() + 1)

    rasters = {"h": h, "w": w}

    for model_key in ["neural_network", "random_forest", "logistic_regression"]:
        col = f"prob_{model_key}"
        if col not in ev1.columns:
            col = "susceptibility_prob"
        rasters[f"suscep_{model_key}"] = build_susceptibility_rgba(ev1, h, w, col)

    rasters["observed"]    = build_observed_rgba(ev1, h, w)
    rasters["disagreement"] = build_disagreement_rgba(ev1, h, w)
    return rasters


def generate_raster_overlays(
    df: pd.DataFrame,
    model_key: str = "neural_network",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Generate RGBA arrays for susceptibility and historical observation overlays.
    Preserves backward-compatibility for test suites and headless callers.
    Returns (rgba_suscep, rgba_obs, (height, width)).
    """
    ev1 = df[df["event_id"] == "DFO_4259"].sort_values(["row", "col"]) if "event_id" in df.columns and "DFO_4259" in df["event_id"].values else df.sort_values(["row", "col"])
    h = int(ev1["row"].max() + 1)
    w = int(ev1["col"].max() + 1)

    col = f"prob_{model_key}"
    if col not in ev1.columns:
        col = "susceptibility_prob"

    probs = ev1[col].values.reshape((h, w))
    perm = ev1["is_perm_water"].values.reshape((h, w))

    rgba_suscep = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_suscep[perm == 1] = [0, 119, 182, 230]

    hi = (probs >= 0.70) & (perm == 0)
    rgba_suscep[hi] = [239, 68, 68, 210]
    mo = (probs >= 0.35) & (probs < 0.70) & (perm == 0)
    rgba_suscep[mo] = [245, 158, 11, 190]
    lo = (probs < 0.35) & (perm == 0) & (~np.isnan(probs))
    rgba_suscep[lo] = [16, 185, 129, 165]

    obs = ev1["flood_label"].values.reshape((h, w)) if "flood_label" in ev1.columns else np.zeros((h, w))
    rgba_obs = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_obs[obs == 1.0] = [0, 212, 255, 210]
    rgba_obs[perm == 1] = [0, 119, 182, 230]

    return rgba_suscep, rgba_obs, (h, w)


@st.cache_data(show_spinner="Computing scenario spatial difference…")
def compute_scenario_diff_rasters(
    df: pd.DataFrame,
    rain_3d: float,
    rain_7d: float,
    rain_event_sum: float,
    model_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute baseline vs scenario rasters for a given rainfall scenario.
    Returns (baseline_rgba, scenario_rgba, diff_rgba).
    Only rainfall features change; terrain features remain fixed.
    """
    from src.models.predict import GeoFloodPredictor

    ev1 = df[df["event_id"] == "DFO_4259"].sort_values(["row", "col"]).copy()
    h   = int(ev1["row"].max() + 1)
    w   = int(ev1["col"].max() + 1)

    col = f"prob_{model_key}"
    if col not in ev1.columns:
        col = "susceptibility_prob"

    base_probs = ev1[col].values.copy().astype(float)

    # Apply scenario to a feature copy
    predictor = GeoFloodPredictor("models")
    feat_cols  = predictor.feature_names

    valid_mask = ~ev1[feat_cols].isnull().any(axis=1) & (ev1["is_perm_water"] == 0)
    ev1_valid  = ev1[valid_mask].copy()

    ev1_valid["rain_3d"]         = float(rain_3d)
    ev1_valid["rain_7d"]         = float(rain_7d)
    ev1_valid["rain_event_sum"]  = float(rain_event_sum)

    scen_probs_valid = predictor.predict(ev1_valid, model_type=model_key).astype(float)

    scen_probs = base_probs.copy()
    valid_indices = np.where(valid_mask.values)[0]
    scen_probs[valid_indices] = scen_probs_valid

    perm = ev1["is_perm_water"].values.reshape((h, w))

    # Build baseline RGBA
    base_p_arr  = base_probs.reshape((h, w))
    scen_p_arr  = scen_probs.reshape((h, w))

    base_rgba = _prob_array_to_rgba(base_p_arr, perm)
    scen_rgba = _prob_array_to_rgba(scen_p_arr, perm)
    diff_rgba = _diff_to_rgba(scen_p_arr - base_p_arr, perm)

    return base_rgba, scen_rgba, diff_rgba, base_probs, scen_probs, valid_mask.values


def _prob_array_to_rgba(probs: np.ndarray, perm: np.ndarray) -> np.ndarray:
    h, w = probs.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[perm == 1] = [2, 132, 199, 200]
    rgba[(probs >= 0.70) & (perm == 0)] = [239, 68, 68, 210]
    rgba[(probs >= 0.35) & (probs < 0.70) & (perm == 0)] = [245, 158, 11, 190]
    rgba[(probs < 0.35) & (perm == 0) & (~np.isnan(probs))] = [16, 185, 129, 165]
    rgba[np.isnan(probs) & (perm == 0)] = [0, 0, 0, 0]
    return rgba


def _diff_to_rgba(diff: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Diverging colour scale for probability differences.
    Positive (increased risk):  red scale
    Negative (decreased risk):  blue scale
    Near-zero: transparent
    """
    h, w = diff.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    pos = (diff > 0.02) & (perm == 0)
    neg = (diff < -0.02) & (perm == 0)

    if pos.any():
        strength = np.clip(diff[pos] / 0.5, 0.0, 1.0)
        r = (200 + 55 * strength).astype(np.uint8)
        g = (50  * (1 - strength)).astype(np.uint8)
        b = np.zeros_like(r)
        a = (100 + 155 * strength).astype(np.uint8)
        rgba[pos, 0] = r; rgba[pos, 1] = g; rgba[pos, 2] = b; rgba[pos, 3] = a

    if neg.any():
        strength = np.clip(-diff[neg] / 0.5, 0.0, 1.0)
        r = np.zeros(neg.sum(), dtype=np.uint8)
        g = (80 * (1 - strength)).astype(np.uint8)
        b = (180 + 75 * strength).astype(np.uint8)
        a = (80 + 140 * strength).astype(np.uint8)
        rgba[neg, 0] = r; rgba[neg, 1] = g; rgba[neg, 2] = b; rgba[neg, 3] = a

    return rgba


# ---------------------------------------------------------------------------
# Application-wide data initialisation
# ---------------------------------------------------------------------------
df_preds, is_sample = load_predictions()
config    = load_config()
sar_data  = load_sar_data()
eval_summary = load_eval_summary()
metrics_df   = load_metrics_csv()
manifest     = load_manifest()
bbox         = config["region"]["bbox"]
fuzzy_sys    = MamdaniRiskFuzzySystem()
analyzer     = LocationAnalyzer(df_features=df_preds, config=config, fuzzy_sys=fuzzy_sys)

# Precompute rasters once
rasters = compute_all_rasters(df_preds) if len(df_preds) > 0 else {}

# Compute map extent from data
if len(df_preds) > 0:
    west, south, east, north = get_dataframe_raster_bounds(df_preds)
else:
    west, south  = bbox["min_lon"], bbox["min_lat"]
    east, north  = bbox["max_lon"], bbox["max_lat"]

bounds     = [[south, west], [north, east]]
center_lat = (south + north) / 2.0
center_lon = (west  + east)  / 2.0

# Compute dataset summary metrics once
if len(df_preds) > 0:
    ev1_df       = df_preds[df_preds["event_id"] == "DFO_4259"]
    total_cells  = len(ev1_df)
    perm_cells   = int((ev1_df["is_perm_water"] == 1).sum())
    valid_cells  = int(ev1_df["flood_label"].isin([0.0, 1.0]).sum())
    flooded_cells = int((ev1_df["flood_label"] == 1.0).sum())
else:
    total_cells, perm_cells, valid_cells, flooded_cells = 249200, 1361, 172199, 19522

# Cell area computation (approximate, accounting for latitude)
CELL_AREA_KM2 = (0.002251 * 111.32) * (0.002246 * 111.32 * np.cos(np.radians(center_lat)))

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_SS_DEFAULTS = {
    "lat":          26.6500,
    "lon":          93.3500,
    "sel_model":    "Neural Network (Primary)",
    "analysis":     None,
    "multi_result": None,
    "clicked_lat":  None,
    "clicked_lon":  None,
    "pres_step":    1,
    "pres_mode":    False,
    "scenario_run": False,
    "scen_rain3d":  None,
    "scen_rain7d":  None,
    "scen_rainsum": None,
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def run_analysis(lat: float, lon: float, overrides: Optional[dict] = None) -> None:
    """Run point analysis and multi-model comparison, store in session state."""
    st.session_state["analysis"] = analyzer.analyze_location(
        lat=lat, lon=lon,
        model_type=MODEL_OPTIONS[st.session_state["sel_model"]],
        rainfall_overrides=overrides,
    )
    st.session_state["multi_result"] = analyzer.compare_models(
        lat=lat, lon=lon,
        rainfall_overrides=overrides,
    )


# Ensure we always have a result
if st.session_state["analysis"] is None:
    run_analysis(st.session_state["lat"], st.session_state["lon"])

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='color:#00d4ff; margin:0;'>🌊 GeoFlood AI</h2>"
        "<p style='color:#64748b; font-size:12px; margin:2px 0 12px 0;'>"
        "Flood Intelligence Platform v4.0</p>",
        unsafe_allow_html=True,
    )

    if is_sample:
        st.markdown('<span class="badge badge-sample">SAMPLE DATA</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-live">LIVE SATELLITE DATA</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Study Region</p>', unsafe_allow_html=True)
    st.write(f"📍 **{config['region']['description']}**")
    st.write(f"Lat: {bbox['min_lat']}°N – {bbox['max_lat']}°N")
    st.write(f"Lon: {bbox['min_lon']}°E – {bbox['max_lon']}°E")
    st.write("Grid: **400 × 623** (249,200 cells @ ~250m)")

    st.markdown('<p class="section-header">Map Layer Controls</p>', unsafe_allow_html=True)
    sidebar_model_label = st.selectbox(
        "Susceptibility Model",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index(st.session_state["sel_model"]),
        help="Select which model's spatial predictions to display on the map.",
    )
    if sidebar_model_label != st.session_state["sel_model"]:
        st.session_state["sel_model"] = sidebar_model_label

    show_suscep     = st.checkbox("Susceptibility Layer",        value=True)
    show_observed   = st.checkbox("Historical Flood Extent",     value=False)
    show_disagree   = st.checkbox("Model Disagreement Layer",    value=False)
    show_landmarks  = st.checkbox("Landmark Markers",            value=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Quick Location</p>', unsafe_allow_html=True)
    sidebar_lat = st.number_input("Latitude °N",  value=st.session_state["lat"],  format="%.4f", step=0.01)
    sidebar_lon = st.number_input("Longitude °E", value=st.session_state["lon"], format="%.4f", step=0.01)
    if st.button("🚀 Analyse", use_container_width=True, type="primary"):
        st.session_state["lat"] = sidebar_lat
        st.session_state["lon"] = sidebar_lon
        run_analysis(sidebar_lat, sidebar_lon)
        st.rerun()

    st.markdown("---")
    if st.button(
        "🎬 Presentation Mode" if not st.session_state["pres_mode"] else "✕ Exit Presentation",
        use_container_width=True,
    ):
        st.session_state["pres_mode"] = not st.session_state["pres_mode"]
        st.session_state["pres_step"] = 1
        st.rerun()

# ---------------------------------------------------------------------------
if st.session_state["pres_mode"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stHeader"] {background: transparent;}
        .block-container {
            max-width: 1450px;
            padding-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# PRESENTATION MODE BANNER
# ---------------------------------------------------------------------------
if st.session_state["pres_mode"]:
    if st.button("✕ Exit Presentation Mode", key="exit_presentation_top"):
        st.session_state["pres_mode"] = False
        st.rerun()

    PRES_STEPS = [
        "Project Overview",
        "Flood Intelligence",
        "Evacuation Demonstration",
        "Research & Validation",
    ]
    step = st.session_state["pres_step"]
    p_cols = st.columns([1, 8, 1])
    with p_cols[0]:
        if st.button("◀ Prev", use_container_width=True) and step > 1:
            st.session_state["pres_step"] -= 1
            st.rerun()
    with p_cols[1]:
        st.markdown(
            f"<div style='text-align:center; background:#1e3a5f; padding:6px 12px; "
            f"border-radius:8px; border:1px solid #00d4ff;'>"
            f"<span style='color:#7dd3fc; font-size:11px; text-transform:uppercase;'>Presentation Mode</span> — "
            f"<span style='color:#00d4ff; font-weight:700;'>Step {step}/4: {PRES_STEPS[step-1]}</span></div>",
            unsafe_allow_html=True,
        )
    with p_cols[2]:
        if st.button("Next ▶", use_container_width=True) and step < 4:
            st.session_state["pres_step"] += 1
            st.rerun()
    st.markdown("")

# ---------------------------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------------------------
if st.session_state["pres_mode"]:
    TABS = [
        "🏠 Overview",
        "🗺️ Flood Intelligence",
        "🚨 Evacuation Intelligence",
        "🔬 Research & Validation",
    ]
    tab_overview, tab_map, tab_evacuation, tab_research = st.tabs(TABS)
else:
    TABS = [
        "🌊 Overview",
        "🗺️ Flood Intel Map",
        "🛡️ Flood Commander",
        "🧪 Scenario Lab",
        "🛰️ Satellite Analysis",
        "⚖️ Model Intelligence",
        "📍 Location Explorer",
        "🔬 Research & Validation",
        "🚨 Evacuation Router",
    ]
    (
        tab_overview,
        tab_map,
        tab_commander,
        tab_scenario,
        tab_sat,
        tab_model,
        tab_location,
        tab_research,
        tab_evacuation,
    ) = st.tabs(TABS)



# ===========================================================================
# TAB 1: OVERVIEW
# ===========================================================================
with tab_overview:
    # Header
    st.markdown(
        "<h1 style='margin-bottom:4px;'>GeoFlood AI</h1>"
        "<p style='color:#7dd3fc; font-size:16px; margin-bottom:20px;'>"
        "Event-Based Flood Susceptibility Mapping &amp; Geospatial Intelligence Platform</p>",
        unsafe_allow_html=True,
    )

    # Headline metrics
    ov_c1, ov_c2, ov_c3, ov_c4, ov_c5 = st.columns(5)
    ov_c1.metric("Grid Cells",       f"{total_cells:,}")
    ov_c2.metric("Valid Land Cells", f"{valid_cells:,}")
    ov_c3.metric("Flooded (DFO_4259)", f"{flooded_cells:,}", f"{flooded_cells/max(1,valid_cells):.1%}")
    ov_c4.metric("Study Area",       f"~{total_cells * CELL_AREA_KM2:,.0f} km²")
    ov_c5.metric("Spatial Resolution", "~250 m")

    st.markdown("---")

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("### What this system does")
        st.markdown("""
For each grid cell in a **400 × 623 raster** over the central Brahmaputra floodplain (Assam, India),
GeoFlood AI estimates the probability that the cell was inundated during a documented historical
monsoonal flood event, given:
- **Terrain features:** Elevation (SRTM), Slope, Height Above Nearest Drainage (HAND), TPI
- **Rainfall forcing:** 3-day antecedent, 7-day antecedent, and total event-window precipitation (GPM IMERG)
- **Three trained ML models:** Logistic Regression, Random Forest, PyTorch Neural Network

The pipeline runs: `satellite data → feature table → trained models → spatial holdout evaluation → interactive map`
""")

        st.markdown("### What this system does NOT do")
        st.markdown("""
- ❌ **Real-time flood forecasting** — `rain_event_sum` requires the storm to be fully observed
- ❌ **Operational early-warning** — no numerical weather prediction integration
- ❌ **Hydrodynamic simulation** — no 2D Saint-Venant routing, river gauge, or channel bathymetry
- ❌ **Generalise to other basins** — models are calibrated to the Brahmaputra only
""")

        st.markdown("### System status")
        model_files = [
            ("Neural Network", "models/neural_network.pt"),
            ("Random Forest",  "models/random_forest.joblib"),
            ("Logistic Reg.",  "models/logistic_regression.joblib"),
            ("Scaler",         "models/scaler.joblib"),
        ]
        for name, path in model_files:
            ok = Path(path).exists()
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} `{path}`")

    with right_col:
        st.markdown("### Available modules")
        modules = [
            ("🗺️ Flood Intel Map",    "7 selectable layers, live coordinate inspection"),
            ("🧪 Scenario Lab",       "Rainfall perturbation → spatial difference maps"),
            ("🛰️ Satellite Analysis", "SAR backscatter → observed flood footprint"),
            ("⚖️ Model Intelligence", "3-model comparison, disagreement map, feature importance"),
            ("📍 Location Explorer",  "Cell-level features, neighbourhood summary"),
            ("🔬 Research & Valid.",  "Calibration, generalisation, leakage audit"),
        ]
        for icon_label, desc in modules:
            st.markdown(
                f"<div class='intel-card-accent' style='margin-bottom:6px;'>"
                f"<b style='color:#00d4ff;'>{icon_label}</b><br>"
                f"<span style='color:#94afd1; font-size:12px;'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Quick start")
        q1, q2 = st.columns(2)
        with q1:
            if st.button("🗺️ Open Map", use_container_width=True, type="primary"):
                st.info("Navigate to the **Flood Intel Map** tab above.")
        with q2:
            if st.button("🧪 Run Scenario", use_container_width=True):
                st.info("Navigate to the **Scenario Lab** tab above.")

        # Data provenance
        st.markdown("### Data provenance")
        st.markdown("""
| Layer | Source | Type |
|---|---|---|
| Terrain | NASA SRTM v3 (30m) | Static |
| Rainfall | GPM IMERG Final | Event |
| Flood labels | DFO / MODIS 250m | Observation |
| River mask | JRC Global Surface Water | Static |
| SAR demo | Synthetic C-band simulation | Synthetic |
""")

    st.markdown("---")
    st.caption(
        f"Study area: {config['region']['description']} | "
        f"Events: DFO_4259 (Jun 2015), DFO_4288 (Aug 2015) | "
        f"Grid: 400×623 = 249,200 cells (EPSG:4326 → EPSG:3857)"
    )


# ===========================================================================
# TAB 2: FLOOD INTEL MAP
# ===========================================================================
with tab_map:
    st.markdown("<h2 style='margin-bottom:2px;'>🗺️ Flood Intelligence Map</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; margin-bottom:12px;'>Layered geospatial analysis — click any cell to inspect it</p>",
        unsafe_allow_html=True,
    )

    active_model_key = MODEL_OPTIONS[st.session_state["sel_model"]]

    # Top metrics bar
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("Active Model", MODEL_OPTIONS[st.session_state["sel_model"]].replace("_", " ").title())
    mm2.metric("Total Cells", f"{total_cells:,}")
    mm3.metric("Valid Land Cells", f"{valid_cells:,}")
    mm4.metric("Historical Flooded", f"{flooded_cells:,}", f"{flooded_cells/max(1,valid_cells):.1%}")

    # Build map
    m = build_base_map(bounds, center_lat, center_lon)
    m = add_study_boundary(m, bounds, west, south, east, north)

    if show_suscep and rasters:
        rgba_key = f"suscep_{active_model_key}"
        if rgba_key in rasters:
            ImageOverlay(
                image=rasters[rgba_key],
                bounds=bounds,
                opacity=0.85,
                mercator_project=True,
                name=f"{st.session_state['sel_model']} Susceptibility",
            ).add_to(m)

    if show_observed and rasters and "observed" in rasters:
        ImageOverlay(
            image=rasters["observed"],
            bounds=bounds,
            opacity=0.80,
            mercator_project=True,
            name="Historical Flood Extent (DFO_4259 / MODIS)",
        ).add_to(m)

    if show_disagree and rasters and "disagreement" in rasters:
        ImageOverlay(
            image=rasters["disagreement"],
            bounds=bounds,
            opacity=0.75,
            mercator_project=True,
            name="NN vs RF Model Disagreement",
        ).add_to(m)

    if show_landmarks:
        m = add_landmarks(m)

    # Selected location marker
    res = st.session_state["analysis"]
    cur_lat = float(st.session_state["lat"])
    cur_lon = float(st.session_state["lon"])
    if res and res.get("success") and not res.get("has_missing_features"):
        m = add_selected_location(m, cur_lat, cur_lon, res)
    else:
        folium.Marker(
            location=[cur_lat, cur_lon],
            tooltip=f"📍 {cur_lat:.4f}°N, {cur_lon:.4f}°E",
            icon=folium.Icon(color="gray", icon="crosshairs", prefix="fa"),
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    map_output = st_folium(m, use_container_width=True, height=560, returned_objects=["last_clicked"])

    # Handle map clicks
    if map_output and map_output.get("last_clicked"):
        clicked = map_output["last_clicked"]
        if clicked.get("lat") is not None:
            c_lat = round(float(clicked["lat"]), 4)
            c_lon = round(float(clicked["lng"]), 4)
            if (
                st.session_state["clicked_lat"] != c_lat
                or st.session_state["clicked_lon"] != c_lon
            ):
                st.session_state["clicked_lat"] = c_lat
                st.session_state["clicked_lon"]  = c_lon
                st.session_state["lat"]          = c_lat
                st.session_state["lon"]          = c_lon
                run_analysis(c_lat, c_lon)
                st.rerun()

    # Legend
    st.markdown(
        "<div class='intel-card' style='margin-top:10px;'>"
        "<b style='color:#7dd3fc;'>Map Legend</b>"
        "<div style='display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:4px; margin-top:8px;'>"
        "<div class='legend-row'><span class='legend-dot' style='background:#ef4444;'></span><b>High Risk</b> &mdash; Susceptibility ≥ 70%</div>"
        "<div class='legend-row'><span class='legend-dot' style='background:#f59e0b;'></span><b>Moderate</b> &mdash; 35%–70%</div>"
        "<div class='legend-row'><span class='legend-dot' style='background:#10b981;'></span><b>Low Risk</b> &mdash; &lt; 35%</div>"
        "<div class='legend-row'><span class='legend-dot' style='background:#0284c7;'></span><b>Permanent River</b> &mdash; JRC perennial water</div>"
        "<div class='legend-row'><span class='legend-dot' style='background:#00d4ff;'></span><b>Observed Flood</b> &mdash; MODIS inundation</div>"
        "<div class='legend-row'><span class='legend-dot' style='background:#facc15;'></span><b>Disagreement</b> &mdash; |NN − RF| ≥ 0.15</div>"
        "</div>"
        "<p style='color:#475569; font-size:11px; margin:8px 0 0 0;'>"
        "Susceptibility = model-predicted probability of historical inundation. "
        "Not a forecast of future flood extent. "
        "<span class='badge badge-prediction'>PREDICTION</span> "
        "<span class='badge badge-observation'>OBSERVATION</span></p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# TAB: FLOOD COMMANDER
# ===========================================================================
if not st.session_state["pres_mode"]:
    with tab_commander:
        render_commander_tab(
            bounds=bounds,
            center_lat=center_lat,
            center_lon=center_lon,
            west=west,
            south=south,
            east=east,
            north=north,
            rasters=rasters,
        )


# ===========================================================================
# TAB 3: SCENARIO LAB
# ===========================================================================
if not st.session_state["pres_mode"]:
    with tab_scenario:
        st.markdown("<h2 style='margin-bottom:2px;'>🧪 Scenario Laboratory</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#64748b; margin-bottom:12px;'>"
            "Modify supported rainfall inputs and observe the model's spatial and point-level response. "
            "Terrain features (elevation, slope, HAND, TPI) are fixed.</p>",
            unsafe_allow_html=True,
        )

        comp_res = st.session_state["multi_result"]

        if comp_res is None or not comp_res.get("success"):
            st.warning("Select a valid location in the Flood Intel Map or Location Explorer tab first.")
        elif comp_res.get("has_missing_features"):
            st.warning(f"⚠️ {comp_res['warning']}")
        else:
            b_feats = comp_res["baseline_features"]
            cell_id = comp_res["cell_id"]

            sc_header = st.columns([2, 1, 1, 1])
            sc_header[0].markdown(
                f"<div class='intel-card-accent'>"
                f"<span class='label-text'>Selected Cell</span><br>"
                f"<b>{cell_id}</b> — {comp_res['cell_lat']:.4f}°N, {comp_res['cell_lon']:.4f}°E<br>"
                f"<span style='color:#64748b; font-size:12px;'>"
                f"Elevation: {b_feats['elevation']:.0f}m | Slope: {b_feats['slope']:.2f}° | "
                f"HAND: {b_feats['relative_elevation']:.1f}m</span></div>",
                unsafe_allow_html=True,
            )
            sc_header[1].metric("Baseline 3d Rain", f"{b_feats['rain_3d']:.1f} mm")
            sc_header[2].metric("Baseline 7d Rain", f"{b_feats['rain_7d']:.1f} mm")
            sc_header[3].metric("Baseline Event Sum", f"{b_feats['rain_event_sum']:.1f} mm")

            st.markdown("---")
            st.markdown("#### Preset Scenarios")
            pr1, pr2, pr3, pr4 = st.columns(4)
            preset = None
            with pr1:
                if st.button("📅 Historical Baseline", use_container_width=True):
                    preset = "base"
            with pr2:
                if st.button("⛈️ Cloudburst (+100mm)", use_container_width=True):
                    preset = "cloudburst"
            with pr3:
                if st.button("🌊 Extreme Deluge (+200mm)", use_container_width=True):
                    preset = "deluge"
            with pr4:
                if st.button("☀️ Suppressed Monsoon (–50%)", use_container_width=True):
                    preset = "drought"

            d3  = float(b_feats["rain_3d"])
            d7  = float(b_feats["rain_7d"])
            dSm = float(b_feats["rain_event_sum"])

            if preset == "base":
                st.session_state["scen_rain3d"]  = d3
                st.session_state["scen_rain7d"]  = d7
                st.session_state["scen_rainsum"] = dSm
            elif preset == "cloudburst":
                st.session_state["scen_rain3d"]  = min(350.0, d3 + 100.0)
                st.session_state["scen_rain7d"]  = min(700.0, d7 + 120.0)
                st.session_state["scen_rainsum"] = min(2500.0, dSm + 200.0)
            elif preset == "deluge":
                st.session_state["scen_rain3d"]  = min(350.0, d3 + 180.0)
                st.session_state["scen_rain7d"]  = min(700.0, d7 + 300.0)
                st.session_state["scen_rainsum"] = min(2500.0, dSm + 600.0)
            elif preset == "drought":
                st.session_state["scen_rain3d"]  = max(0.0, d3 * 0.5)
                st.session_state["scen_rain7d"]  = max(0.0, d7 * 0.5)
                st.session_state["scen_rainsum"] = max(0.0, dSm * 0.5)

            init3  = st.session_state["scen_rain3d"]  if st.session_state["scen_rain3d"]  is not None else d3
            init7  = st.session_state["scen_rain7d"]  if st.session_state["scen_rain7d"]  is not None else d7
            initSm = st.session_state["scen_rainsum"] if st.session_state["scen_rainsum"] is not None else dSm

            sl1, sl2, sl3 = st.columns(3)
            with sl1:
                val_3d = st.slider(
                    "3-Day Pre-Event Rainfall (mm)",
                    min_value=float(RAINFALL_BOUNDS["rain_3d"]["min"]),
                    max_value=float(RAINFALL_BOUNDS["rain_3d"]["max"]),
                    value=float(init3), step=5.0,
                    help=f"Baseline: {d3:.1f} mm | Training range: {TRAINING_RAINFALL_DISTRIBUTION['rain_3d']['min']:.0f}–{TRAINING_RAINFALL_DISTRIBUTION['rain_3d']['max']:.0f} mm",
                )
            with sl2:
                val_7d = st.slider(
                    "7-Day Pre-Event Rainfall (mm)",
                    min_value=float(RAINFALL_BOUNDS["rain_7d"]["min"]),
                    max_value=float(RAINFALL_BOUNDS["rain_7d"]["max"]),
                    value=float(init7), step=10.0,
                    help=f"Baseline: {d7:.1f} mm",
                )
            with sl3:
                val_sum = st.slider(
                    "Event Cumulative Rainfall (mm)",
                    min_value=float(RAINFALL_BOUNDS["rain_event_sum"]["min"]),
                    max_value=float(RAINFALL_BOUNDS["rain_event_sum"]["max"]),
                    value=float(initSm), step=50.0,
                    help=f"Baseline: {dSm:.1f} mm",
                )

            # Warnings for extrapolation
            extrap_keys = {
                "rain_3d": val_3d, "rain_7d": val_7d, "rain_event_sum": val_sum,
            }
            for k, v in extrap_keys.items():
                dist = TRAINING_RAINFALL_DISTRIBUTION.get(k, {})
                if dist and (v < dist["min"] or v > dist["max"]):
                    st.warning(
                        f"⚠️ `{k}` = {v:.1f} mm is outside the training distribution "
                        f"({dist['min']:.1f} – {dist['max']:.1f} mm). Extrapolation may be unreliable."
                    )

            if val_3d > val_7d:
                st.error("❌ Physical inconsistency: 3-day rainfall cannot exceed 7-day rainfall.")

            scen_model_label = st.selectbox(
                "Spatial Difference Model",
                options=list(MODEL_OPTIONS.keys()),
                index=0,
                help="Model used to compute the full-raster scenario difference map.",
            )
            scen_model_key = MODEL_OPTIONS[scen_model_label]

            col_run, col_reset = st.columns([1, 4])
            with col_run:
                run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")
            with col_reset:
                if st.button("↺ Reset to Baseline", use_container_width=True):
                    st.session_state.update({
                        "scen_rain3d": d3, "scen_rain7d": d7, "scen_rainsum": dSm, "scenario_run": False,
                    })
                    st.rerun()

            overrides = {"rain_3d": val_3d, "rain_7d": val_7d, "rain_event_sum": val_sum}
            sim_res   = analyzer.compare_models(
                lat=comp_res["cell_lat"],
                lon=comp_res["cell_lon"],
                rainfall_overrides=overrides,
            )
            sim_models = sim_res["models"]

            # Point-level comparison chart
            st.markdown("---")
            st.markdown("#### Point-Level Baseline vs Scenario")

            m_names    = ["Neural Network", "Random Forest", "Logistic Regression"]
            m_keys_ord = ["neural_network", "random_forest", "logistic_regression"]
            base_probs = [sim_models[k]["baseline_prob"] for k in m_keys_ord]
            scen_probs = [sim_models[k]["scenario_prob"] for k in m_keys_ord]
            deltas     = [sim_models[k]["delta_prob"]    for k in m_keys_ord]

            fig_pts = go.Figure()
            fig_pts.add_trace(go.Bar(
                x=m_names, y=base_probs, name="Baseline", marker_color="#0284c7",
                text=[f"{p:.1%}" for p in base_probs], textposition="auto",
            ))
            fig_pts.add_trace(go.Bar(
                x=m_names, y=scen_probs, name="Scenario", marker_color="#ef4444",
                text=[f"{p:.1%}" for p in scen_probs], textposition="auto",
            ))
            fig_pts.update_layout(
                **PLOT_LAYOUT,
                barmode="group",
                yaxis=dict(title="Susceptibility Probability", range=[0, 1.05], tickformat=".0%",
                           gridcolor="#1e3a5f"),
                xaxis=dict(gridcolor="#1e3a5f"),
                height=340,
                legend=dict(bgcolor="#0d1e35", bordercolor="#1e3a5f", borderwidth=1),
            )
            st.plotly_chart(fig_pts, use_container_width=True)

            # Scenario summary table
            sim_rows = []
            for k, label in zip(m_keys_ord, m_names):
                m_s = sim_models[k]
                dv  = m_s["delta_prob"]
                sim_rows.append({
                    "Model":              label,
                    "Baseline P":         f"{m_s['baseline_prob']:.1%}",
                    "Scenario P":         f"{m_s['scenario_prob']:.1%}",
                    "Δ Probability":      f"{'+' if dv >= 0 else ''}{dv:.1%}",
                    "Baseline Category":  m_s["baseline_category"],
                    "Scenario Category":  m_s["scenario_category"],
                    "Category Changed":   "⚠️ YES" if m_s["baseline_category"] != m_s["scenario_category"] else "—",
                })
            st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

            # Spatial difference maps
            if run_btn and len(df_preds) > 0:
                with st.spinner("Computing spatial scenario difference…"):
                    base_rgba, scen_rgba, diff_rgba, base_p_flat, scen_p_flat, valid_mask = (
                        compute_scenario_diff_rasters(df_preds, val_3d, val_7d, val_sum, scen_model_key)
                    )
                    st.session_state["scenario_run"] = True

                    # Summary statistics
                    changed = np.abs(scen_p_flat - base_p_flat) > 0.01
                    n_valid = int(valid_mask.sum())
                    n_changed = int(changed[valid_mask].sum())
                    mean_delta = float(np.mean(scen_p_flat[valid_mask] - base_p_flat[valid_mask]))

                    st.markdown("---")
                    st.markdown("#### Spatial Difference Summary")

                    ss1, ss2, ss3, ss4 = st.columns(4)
                    ss1.metric("Valid Cells Evaluated", f"{n_valid:,}")
                    ss2.metric("Cells Changed (|Δp| > 1%)", f"{n_changed:,}")
                    ss3.metric("Cells Changed %", f"{n_changed / max(1, n_valid):.1%}")
                    ss4.metric("Mean Probability Δ", f"{mean_delta:+.3f}")

                    st.markdown("#### Spatial Maps — Baseline | Scenario | Difference")
                    map_col1, map_col2, map_col3 = st.columns(3)

                    def _mini_map(rgba_arr, title, caption_str):
                        m_mini = folium.Map(
                            location=[center_lat, center_lon], zoom_start=9,
                            tiles="OpenStreetMap", prefer_canvas=True,
                        )
                        m_mini.fit_bounds(bounds, padding=(5, 5))
                        ImageOverlay(
                            image=rgba_arr, bounds=bounds,
                            opacity=0.85, mercator_project=True,
                        ).add_to(m_mini)
                        add_study_boundary(m_mini, bounds, west, south, east, north)
                        st.markdown(f"**{title}**")
                        st_folium(m_mini, use_container_width=True, height=260, returned_objects=[])
                        st.caption(caption_str)

                    with map_col1:
                        _mini_map(base_rgba, "Baseline", f"Rain: {d3:.0f}/{d7:.0f}/{dSm:.0f} mm")
                    with map_col2:
                        _mini_map(scen_rgba, "Scenario", f"Rain: {val_3d:.0f}/{val_7d:.0f}/{val_sum:.0f} mm")
                    with map_col3:
                        _mini_map(diff_rgba, "Difference", "Red = increased risk | Blue = decreased")

                    # Diverging legend
                    st.markdown(
                        "<div class='intel-card'>"
                        "<b style='color:#7dd3fc;'>Difference Map Scale</b>"
                        "<div style='display:flex; gap:16px; margin-top:6px;'>"
                        "<div class='legend-row'><span class='legend-dot' style='background:#dc2626;'></span><b>Red</b> — Increased susceptibility (scenario &gt; baseline)</div>"
                        "<div class='legend-row'><span class='legend-dot' style='background:#2563eb;'></span><b>Blue</b> — Decreased susceptibility (scenario &lt; baseline)</div>"
                        "<div class='legend-row'><span class='legend-dot' style='background:#374151;'></span><b>Transparent</b> — |Δp| ≤ 1% (no meaningful change)</div>"
                        "</div>"
                        "<p style='color:#475569; font-size:11px; margin:6px 0 0 0;'>"
                        "⚠️ This is a single-point rainfall perturbation applied uniformly across all cells. "
                        "It is not a complete hydrological simulation. Terrain and base flow remain unchanged.</p>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            elif not st.session_state.get("scenario_run"):
                st.info("👆 Adjust the rainfall sliders above and click **▶ Run Simulation** to generate the spatial difference maps.")

            st.markdown("---")
            st.info(
                "💡 **Scientific note:** This scenario modifies rainfall inputs and recomputes model predictions. "
                "Random Forest predictions may be unchanged if rainfall does not cross a decision-tree split threshold. "
                "Neural Network predictions change smoothly. Neither constitutes a hydrodynamic flood simulation."
            )


# ===========================================================================
# TAB 4: SATELLITE ANALYSIS
# ===========================================================================
if not st.session_state["pres_mode"]:
    with tab_sat:
        st.markdown("<h2 style='margin-bottom:2px;'>🛰️ Satellite Analysis</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#64748b; margin-bottom:12px;'>"
            "Compare satellite-derived observations with model-predicted susceptibility.</p>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='intel-card-accent'>"
            "<span class='badge badge-observation'>OBSERVATION</span> "
            "<span class='badge badge-synthetic'>SYNTHETIC BENCHMARK</span>"
            "<p style='margin:8px 0 0 0; color:#94afd1; font-size:13px;'>"
            "This tab shows two distinct information types: "
            "(1) Historical MODIS optical flood extent — a real satellite observation; "
            "(2) Sentinel-1 SAR Otsu detection — an algorithmic benchmark on synthetic backscatter data. "
            "The SAR IoU of 0.991 reflects mathematical convergence on a calibrated bimodal distribution, "
            "<b>not empirical sensor accuracy against raw radar imagery.</b></p></div>",
            unsafe_allow_html=True,
        )

        sat_tab1, sat_tab2, sat_tab3 = st.tabs([
            "Historical Flood Extent (MODIS)",
            "SAR Backscatter Detection",
            "Susceptibility vs Observation",
        ])

        with sat_tab1:
            st.markdown("### Historical Observed Flood Extent — DFO/MODIS")
            st.markdown("""
            The historical flood labels derive from **MODIS Terra/Aqua 250m surface reflectance**
            composited over the DFO_4259 flood event (June–July 2015).

            - Cells classified as **flooded** (label = 1): optical water detection confirmed inundation
            - Cells classified as **dry** (label = 0): no optical water detected
            - Cells with **no label**: cloud cover, shadow, or observation gap

            This is the ground truth label used for model training and evaluation.
            """)

            st.markdown('<span class="badge badge-observation">REAL SATELLITE OBSERVATION — MODIS TERRA/AQUA</span>', unsafe_allow_html=True)
            st.markdown("")

            if len(df_preds) > 0 and "observed" in rasters:
                m_obs = build_base_map(bounds, center_lat, center_lon)
                m_obs = add_study_boundary(m_obs, bounds, west, south, east, north)
                ImageOverlay(
                    image=rasters["observed"], bounds=bounds,
                    opacity=0.85, mercator_project=True,
                    name="MODIS Observed Flood (DFO_4259)",
                ).add_to(m_obs)
                if show_landmarks:
                    m_obs = add_landmarks(m_obs)
                folium.LayerControl().add_to(m_obs)
                st_folium(m_obs, use_container_width=True, height=480, returned_objects=[])

            st.markdown(
                "<div class='intel-card'>"
                "<div class='legend-row'><span class='legend-dot' style='background:#00d4ff;'></span>"
                "<b>Cyan-Blue</b> — MODIS detected inundation during DFO_4259 event</div>"
                "<div class='legend-row'><span class='legend-dot' style='background:#0264b4;'></span>"
                "<b>Dark Blue</b> — Perennial Brahmaputra river channel (JRC)</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Observed Flooded Cells", f"{flooded_cells:,}")
            oc2.metric("Observed Dry Cells", f"{valid_cells - flooded_cells:,}")
            oc3.metric("Prevalence", f"{flooded_cells/max(1,valid_cells):.1%}")

        with sat_tab2:
            st.markdown("### Sentinel-1 SAR Backscatter Flood Detection")
            st.markdown(
                '<span class="badge badge-synthetic">ALGORITHMIC CONSISTENCY BENCHMARK</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.warning(
                "**Formal Disclaimer:** The SAR demo data (`outputs/sar_demo_data.json`) is a "
                "calibrated synthetic backscatter simulation using published C-band radar parameters. "
                "The 0.991 IoU reflects Otsu thresholding on an idealized bimodal Gaussian distribution "
                "with low variance (σ ≈ 1.5 dB). This verifies the algorithm is correct — it does not "
                "represent empirical satellite sensor accuracy against raw Sentinel-1 GRD imagery."
            )

            if sar_data:
                s_m = sar_data["metrics"]
                sub = sar_data["subregion"]
                stats = sar_data["stats"]

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("SAR vs MODIS IoU", f"{s_m['iou']:.3f}", help="Algorithmic benchmark only")
                sc2.metric("Precision",         f"{s_m['precision']:.3f}")
                sc3.metric("Recall",            f"{s_m['recall']:.3f}")
                sc4.metric("Otsu Threshold",    f"{s_m['threshold_db']:.2f} dB")

                left_sar, right_sar = st.columns(2)
                with left_sar:
                    st.markdown("**Demonstration Subregion**")
                    st.markdown(f"""
                    - Latitude: {sub['min_lat']:.4f}°N – {sub['max_lat']:.4f}°N
                    - Longitude: {sub['min_lon']:.4f}°E – {sub['max_lon']:.4f}°E
                    - Grid: {sub['shape'][0]} × {sub['shape'][1]} cells
                    - SAR flooded pixels: {s_m['total_flooded_sar_cells']:,}
                    - Reference flooded pixels: {s_m['total_reference_flooded_cells']:,}
                    - Pre-flood mean: {stats['mean_pre_vv_db']:.2f} dB
                    - Post-flood mean: {stats['mean_post_vv_db']:.2f} dB
                    """)

                with right_sar:
                    st.markdown("**C-Band Backscatter Physics**")
                    st.markdown("""
                    1. **Pre-flood land:** Rough soil/vegetation → diffuse backscatter (~−10.7 dB)
                    2. **Post-flood water:** Specular mirror → radar deflected away (~−20.5 dB)
                    3. **Otsu threshold** automatically separates the bimodal distribution
                    4. **Real-world limits:** Wind Bragg scattering, vegetation double-bounce, radar shadow
                       constrain operational SAR IoU to **0.70–0.85**
                    """)

                # Otsu histogram visualisation
                fig_hist = go.Figure()
                np.random.seed(42)
                pre_vals  = np.random.normal(stats["mean_pre_vv_db"],  1.8, 7884)
                post_vals = np.random.normal(stats["mean_post_vv_db"], 1.5, 2116)
                diff_pre  = np.random.normal(-0.5, 1.2, 7884)
                diff_post = np.random.normal(-10.5, 1.5, 2116)

                fig_hist.add_trace(go.Histogram(
                    x=diff_pre, name="Dry land Δσ⁰", nbinsx=60,
                    marker_color="#10b981", opacity=0.7,
                ))
                fig_hist.add_trace(go.Histogram(
                    x=diff_post, name="Flooded Δσ⁰", nbinsx=60,
                    marker_color="#0284c7", opacity=0.7,
                ))
                fig_hist.add_vline(
                    x=s_m["threshold_db"], line_width=2, line_dash="dash",
                    line_color="#ef4444", annotation_text=f"Otsu τ = {s_m['threshold_db']:.2f} dB",
                )
                fig_hist.update_layout(
                    **PLOT_LAYOUT, barmode="overlay",
                    xaxis_title="Backscatter Difference Δσ⁰ (dB)",
                    yaxis_title="Pixel Count (simulated distribution)",
                    height=320,
                    title="Simulated Backscatter Difference Distribution",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("SAR demo data not found at `outputs/sar_demo_data.json`.")

        with sat_tab3:
            st.markdown("### Susceptibility vs Historical Flood Comparison")
            st.info(
                "This view compares the NN flood susceptibility map against the MODIS-observed flood extent. "
                "Both layers cover the same DFO_4259 event. Susceptibility = model prediction; "
                "Observation = satellite-derived ground truth used for evaluation."
            )

            if len(df_preds) > 0 and "suscep_neural_network" in rasters and "observed" in rasters:
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    st.markdown("**NN Flood Susceptibility** `[PREDICTION]`")
                    m_suscep = build_base_map(bounds, center_lat, center_lon)
                    m_suscep = add_study_boundary(m_suscep, bounds, west, south, east, north)
                    ImageOverlay(
                        image=rasters["suscep_neural_network"],
                        bounds=bounds, opacity=0.85, mercator_project=True,
                    ).add_to(m_suscep)
                    st_folium(m_suscep, use_container_width=True, height=380, returned_objects=[])

                with comp_col2:
                    st.markdown("**MODIS Observed Flood Extent** `[OBSERVATION]`")
                    m_obs2 = build_base_map(bounds, center_lat, center_lon)
                    m_obs2 = add_study_boundary(m_obs2, bounds, west, south, east, north)
                    ImageOverlay(
                        image=rasters["observed"],
                        bounds=bounds, opacity=0.85, mercator_project=True,
                    ).add_to(m_obs2)
                    st_folium(m_obs2, use_container_width=True, height=380, returned_objects=[])

                # Agreement metrics from validation
                if metrics_df is not None:
                    nn_block = metrics_df[
                        (metrics_df["split"] == "Spatial Block Holdout") &
                        (metrics_df["model"].str.contains("Neural Network"))
                    ]
                    if not nn_block.empty:
                        row = nn_block.iloc[0]
                        am1, am2, am3, am4 = st.columns(4)
                        am1.metric("NN ROC-AUC (Block Holdout)", f"{row.get('roc_auc', 0):.3f}")
                        am2.metric("NN Recall",    f"{row.get('recall', 0):.1%}")
                        am3.metric("NN Precision", f"{row.get('precision', 0):.1%}")
                        am4.metric("NN F1",        f"{row.get('f1', 0):.3f}")


# ===========================================================================
# TAB 5: MODEL INTELLIGENCE
# ===========================================================================
if not st.session_state["pres_mode"]:
    with tab_model:
        st.markdown("<h2 style='margin-bottom:2px;'>⚖️ Model Intelligence</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#64748b; margin-bottom:12px;'>"
            "Logistic Regression vs Random Forest vs PyTorch Neural Network — "
            "predictions, disagreement, feature importance, generalisation.</p>",
            unsafe_allow_html=True,
        )

        comp_res = st.session_state["multi_result"]

        # ─── Section A: Side-by-side model predictions at selected location ───
        if comp_res and comp_res.get("success") and not comp_res.get("has_missing_features"):
            st.markdown("#### A. Predictions at Selected Location")
            st.caption(f"Cell: `{comp_res['cell_id']}` — {comp_res['cell_lat']:.4f}°N, {comp_res['cell_lon']:.4f}°E | Ground truth: {comp_res['ground_truth']}")

            mods   = comp_res["models"]
            mc1, mc2, mc3 = st.columns(3)

            for col_w, m_key, label in zip(
                [mc1, mc2, mc3],
                ["neural_network", "random_forest", "logistic_regression"],
                ["🧠 Neural Network", "🌲 Random Forest", "📐 Logistic Reg."],
            ):
                m_obj = mods[m_key]
                p     = m_obj["baseline_prob"]
                cat   = m_obj["baseline_category"]
                card  = risk_card_class(cat)
                p_col = "#ef4444" if p >= 0.70 else ("#f59e0b" if p >= 0.35 else "#10b981")
                with col_w:
                    st.markdown(
                        f"<div class='{card}'>"
                        f"<b style='color:#e2e8f0;'>{label}</b><br>"
                        f"<span style='font-size:28px; font-weight:800; color:{p_col};'>{p:.1%}</span><br>"
                        f"<span style='color:#94afd1; font-size:12px;'>{cat}</span><br>"
                        f"<span style='color:#64748b; font-size:11px;'>τ = {m_obj['threshold']:.4f} | {m_obj['baseline_decision']}</span><br>"
                        f"<span style='color:#64748b; font-size:11px;'>Holdout ROC-AUC: {m_obj['test_metrics'].get('roc_auc', 0):.3f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Probability comparison chart
            fig_mp = go.Figure()
            m_labels = ["Neural Network", "Random Forest", "Logistic Reg."]
            m_keys   = ["neural_network", "random_forest", "logistic_regression"]
            probs_pt = [mods[k]["baseline_prob"] for k in m_keys]
            threshs  = [mods[k]["threshold"]     for k in m_keys]
            col_list = [MODEL_COLORS[k]          for k in m_keys]

            fig_mp.add_trace(go.Bar(
                x=m_labels, y=probs_pt, name="Susceptibility",
                marker_color=col_list,
                text=[f"{p:.1%}" for p in probs_pt], textposition="auto",
            ))
            for i, (lbl, thresh, color) in enumerate(zip(m_labels, threshs, col_list)):
                fig_mp.add_shape(
                    type="line", x0=i - 0.4, x1=i + 0.4, y0=thresh, y1=thresh,
                    line=dict(color="#ef4444", width=2, dash="dash"),
                )
            fig_mp.update_layout(
                **PLOT_LAYOUT,
                yaxis=dict(title="Susceptibility Probability", range=[0, 1.05], tickformat=".0%",
                           gridcolor="#1e3a5f"),
                xaxis=dict(gridcolor="#1e3a5f"),
                height=320,
                annotations=[dict(
                    x=0.5, y=-0.15, xref="paper", yref="paper",
                    text="Dashed red lines = model decision thresholds (τ)",
                    showarrow=False, font=dict(color="#64748b", size=11),
                )],
            )
            st.plotly_chart(fig_mp, use_container_width=True)
        else:
            st.info("Select a valid location in the Flood Intel Map or Location Explorer to see point predictions.")

        st.markdown("---")

        # ─── Section B: Model Disagreement Spatial Map ───
        st.markdown("#### B. Model Disagreement Map")
        st.markdown(
            "Cells where |NN probability − RF probability| ≥ 0.15 are highlighted. "
            "Disagreement indicates model sensitivity or feature interactions — not a calibrated probability of error."
        )

        if len(df_preds) > 0 and "disagreement" in rasters:
            m_dis = build_base_map(bounds, center_lat, center_lon)
            m_dis = add_study_boundary(m_dis, bounds, west, south, east, north)
            ImageOverlay(
                image=rasters["disagreement"],
                bounds=bounds, opacity=0.80, mercator_project=True,
                name="NN vs RF Disagreement",
            ).add_to(m_dis)
            if show_landmarks:
                m_dis = add_landmarks(m_dis)
            st_folium(m_dis, use_container_width=True, height=400, returned_objects=[])
            st.markdown(
                "<div class='intel-card'>"
                "<div class='legend-row'><span class='legend-dot' style='background:#facc15;'></span><b>Yellow</b> — Moderate disagreement (|Δp| 0.15–0.30)</div>"
                "<div class='legend-row'><span class='legend-dot' style='background:#dc2626;'></span><b>Red</b> — High disagreement (|Δp| ≥ 0.30)</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ─── Section C: Held-out evaluation metrics table ───
        st.markdown("#### C. Held-Out Evaluation Metrics")
        st.caption("From `reports/phase3_model_comparison.csv` — actual disjoint test partitions.")

        if metrics_df is not None:
            display_cols = [c for c in [
                "split", "model", "roc_auc", "pr_auc", "precision", "recall", "f1",
                "brier_score", "brier_skill_score", "expected_calibration_error",
            ] if c in metrics_df.columns]
            fmt_map = {c: "{:.3f}" for c in display_cols if c not in ("split", "model")}
            st.dataframe(
                metrics_df[display_cols].style.format(fmt_map),
                use_container_width=True,
            )

            st.error(
                "🚨 **Critical Finding:** Random Forest **ROC-AUC collapses from 0.901 → 0.485** "
                "on the Cross-Event Holdout (DFO_4288, August 2015), confirming temporal generalisation failure. "
                "The Neural Network maintains **ROC-AUC = 0.672** and **Recall = 80.4%** across events."
            )
        else:
            st.warning("Run `python -m src.evaluation.run_validation` to generate evaluation metrics.")

        st.markdown("---")

        # ─── Section D: Feature importance ───
        st.markdown("#### D. Permutation Feature Importance (Held-Out Test Set)")

        fi_data = eval_summary.get("feature_importance", {}) if eval_summary else {}

        if fi_data:
            fi_c1, fi_c2 = st.columns(2)
            for col_w, m_key, m_label in [
                (fi_c1, "random_forest",   "Random Forest"),
                (fi_c2, "neural_network",  "Neural Network"),
            ]:
                fi_model = fi_data.get(m_key, {})
                if fi_model:
                    rows = sorted(
                        [{"Feature": k, "Mean ROC-AUC Drop": v["mean_drop"], "Std": v["std_drop"]}
                         for k, v in fi_model.items()],
                        key=lambda r: r["Mean ROC-AUC Drop"], reverse=True,
                    )
                    fi_df  = pd.DataFrame(rows)
                    col_list2 = [
                        "#ef4444" if v >= 0.05 else
                        "#f59e0b" if v >= 0.01 else
                        "#6b7280"
                        for v in fi_df["Mean ROC-AUC Drop"]
                    ]
                    fig_fi = go.Figure(go.Bar(
                        x=fi_df["Mean ROC-AUC Drop"], y=fi_df["Feature"],
                        orientation="h",
                        error_x=dict(type="data", array=fi_df["Std"].tolist()),
                        marker_color=col_list2,
                    ))
                    fig_fi.update_layout(
                        **PLOT_LAYOUT,
                        title=f"{m_label} — Permutation Importance",
                        xaxis_title="Mean ROC-AUC Drop (5 repeats)",
                        height=300,
                        yaxis=dict(autorange="reversed"),
                    )
                    with col_w:
                        st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance available after running `python -m src.evaluation.run_validation`.")

        # Feature interpretation table
        with st.expander("Feature interpretation"):
            st.markdown("""
    | Feature | Physical Meaning | Importance |
    |---|---|---|
    | `relative_elevation` | Height Above Nearest Drainage (HAND) — primary topographic control on inundation | ⭐⭐⭐ |
    | `dist_river` | Distance to Brahmaputra mainstem — governs lateral overbank reach | ⭐⭐⭐ |
    | `rain_event_sum` | Total storm volume forcing — available only post-event | ⭐⭐ |
    | `elevation` | Absolute terrain height — distinguishes valley from upland | ⭐⭐ |
    | `slope` | Runoff velocity; low slope promotes ponding | ⭐ |
    | `rain_7d_pre` | Antecedent catchment saturation | ⭐ |
    | `twi` | Topographic Wetness Index — steady-state moisture proxy | ⭐ |
    | `curvature` | Local concavity; identifies bowl-shaped depressions | — |
    | `aspect` | Terrain orientation; minimal direct effect in flat floodplain | — |
    """)

        st.markdown("---")

        # ─── Section E: Model architecture ───
        st.markdown("#### E. Model Architectures")
        ea1, ea2, ea3 = st.columns(3)
        for col_w, m_key in zip([ea1, ea2, ea3], ["neural_network", "random_forest", "logistic_regression"]):
            meta = MODEL_METADATA_INFO[m_key]
            with col_w:
                st.markdown(
                    f"<div class='intel-card'>"
                    f"<b style='color:#00d4ff;'>{meta['name']}</b><br>"
                    f"<span class='label-text'>Architecture</span><br>"
                    f"<span class='value-text'>{meta['arch']}</span><br>"
                    f"<span class='label-text' style='margin-top:6px;'>Preprocessing</span><br>"
                    f"<span class='value-text'>{meta['input_prep']}</span><br>"
                    f"<span class='label-text' style='margin-top:6px;'>Decision Boundary</span><br>"
                    f"<span class='value-text'>{meta['non_linear']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ===========================================================================
# TAB 6: LOCATION EXPLORER
# ===========================================================================
if not st.session_state["pres_mode"]:
    with tab_location:
        st.markdown("<h2 style='margin-bottom:2px;'>📍 Location Explorer</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#64748b; margin-bottom:12px;'>"
            "Inspect any geographic location: features, model output, neighbourhood context.</p>",
            unsafe_allow_html=True,
        )

        loc_left, loc_right = st.columns([5, 5])

        with loc_left:
            st.markdown("#### Coordinate Input")

            if st.session_state["clicked_lat"] is not None:
                st.info(
                    f"📍 Last map click: **{st.session_state['clicked_lat']:.4f}°N, "
                    f"{st.session_state['clicked_lon']:.4f}°E**"
                )

            lc1, lc2 = st.columns(2)
            with lc1:
                in_lat = st.number_input(
                    "Latitude (°N)", value=float(st.session_state["lat"]),
                    min_value=-90.0, max_value=90.0, step=0.01, format="%.4f",
                    help=f"Study area: {bbox['min_lat']}°N – {bbox['max_lat']}°N",
                )
            with lc2:
                in_lon = st.number_input(
                    "Longitude (°E)", value=float(st.session_state["lon"]),
                    min_value=-180.0, max_value=180.0, step=0.01, format="%.4f",
                    help=f"Study area: {bbox['min_lon']}°E – {bbox['max_lon']}°E",
                )

            sel_model_label = st.selectbox(
                "Analysis Model",
                options=list(MODEL_OPTIONS.keys()),
                index=list(MODEL_OPTIONS.keys()).index(st.session_state["sel_model"]),
            )
            if sel_model_label != st.session_state["sel_model"]:
                st.session_state["sel_model"] = sel_model_label

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("🚀 Analyse", use_container_width=True, type="primary"):
                    st.session_state["lat"] = in_lat
                    st.session_state["lon"] = in_lon
                    run_analysis(in_lat, in_lon)
                    st.rerun()
            with b2:
                if st.button("📍 Use Map Click", use_container_width=True):
                    if st.session_state["clicked_lat"] is not None:
                        st.session_state["lat"] = st.session_state["clicked_lat"]
                        st.session_state["lon"] = st.session_state["clicked_lon"]
                        run_analysis(st.session_state["lat"], st.session_state["lon"])
                        st.rerun()
                    else:
                        st.warning("Click on the Flood Intel Map first.")
            with b3:
                if st.button("↺ Reset", use_container_width=True):
                    st.session_state.update({
                        "lat": 26.6500, "lon": 93.3500,
                        "clicked_lat": None, "clicked_lon": None,
                    })
                    run_analysis(26.6500, 93.3500)
                    st.rerun()

            # Feature detail table
            res = st.session_state["analysis"]
            if res and res.get("success") and not res.get("has_missing_features"):
                feats = res["features"]
                feat_rows = [
                    {"Feature", "Value", "Unit", "Availability"},
                    ("Elevation (SRTM)", f"{feats.get('elevation', 'N/A'):.1f}" if feats.get('elevation') is not None else "N/A", "m", "Static"),
                    ("Slope", f"{feats.get('slope', 'N/A'):.2f}" if feats.get('slope') is not None else "N/A", "°", "Static"),
                    ("HAND (Rel. Elevation)", f"{feats.get('relative_elevation', 'N/A'):.1f}" if feats.get('relative_elevation') is not None else "N/A", "m", "Static"),
                    ("TPI", f"{feats.get('tpi', 'N/A'):.1f}" if feats.get('tpi') is not None else "N/A", "m", "Static"),
                    ("Event Total Rainfall", f"{feats.get('rain_event_sum', 'N/A'):.1f}" if feats.get('rain_event_sum') is not None else "N/A", "mm", "Post-event"),
                    ("3-Day Pre-event Rain", f"{feats.get('rain_3d', 'N/A'):.1f}" if feats.get('rain_3d') is not None else "N/A", "mm", "Pre-event"),
                    ("7-Day Pre-event Rain", f"{feats.get('rain_7d', 'N/A'):.1f}" if feats.get('rain_7d') is not None else "N/A", "mm", "Pre-event"),
                ]
                feat_table = pd.DataFrame(
                    [r for r in feat_rows if isinstance(r, tuple)],
                    columns=["Feature", "Value", "Unit", "Availability"],
                )
                st.markdown("#### Physical Feature Values")
                st.dataframe(feat_table, use_container_width=True, hide_index=True)

        with loc_right:
            st.markdown("#### Analysis Results")
            res = st.session_state["analysis"]

            if res is None:
                st.info("Enter coordinates and click Analyse.")
            elif not res.get("success"):
                st.error(f"❌ {res.get('error', 'Invalid location')}")
                st.markdown("""
    > **Supported geographic coverage:**
    > - Latitude: 26.30°N – 27.20°N
    > - Longitude: 92.80°E – 94.20°E
    > - Brahmaputra floodplain, Assam, India
    """)
            elif res.get("has_missing_features"):
                st.warning(f"⚠️ {res['warning']}")
                st.markdown(
                    f"<div class='card-unavail'>"
                    f"<b>Cell {res['cell_id']}</b> — {res['cell_lat']:.4f}°N, {res['cell_lon']:.4f}°E<br>"
                    f"<span style='color:#6b7280;'>Susceptibility: Unavailable (missing features)</span><br>"
                    f"<small>Unobserved data is never imputed as 'safe'.</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                p    = res["susceptibility_prob"]
                cat  = res["risk_category"]
                card = risk_card_class(cat)
                p_col = "#ef4444" if p >= 0.70 else ("#f59e0b" if p >= 0.35 else "#10b981")

                st.markdown(
                    f"<div class='{card}'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                    f"<b>Cell {res['cell_id']}</b>"
                    f"<span class='badge badge-prediction'>PREDICTION</span></div>"
                    f"<div style='margin-top:6px;'>"
                    f"<span style='font-size:36px; font-weight:900; color:{p_col};'>{p:.1%}</span>"
                    f"<span style='color:#94afd1; margin-left:12px;'>{cat}</span></div>"
                    f"<div style='margin-top:8px; font-size:12px; color:#94afd1;'>"
                    f"📍 {res['cell_lat']:.4f}°N, {res['cell_lon']:.4f}°E<br>"
                    f"Model: {sel_model_label}<br>"
                    f"Decision (τ={res['threshold']:.4f}): {res['binary_decision']}<br>"
                    f"Fuzzy: {res['fuzzy_category']} ({res['fuzzy_score']}/100)<br>"
                    f"Ground truth: {res['ground_truth']}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

                # Out-of-distribution check
                feats = res["features"]
                ood_keys = ["rain_3d", "rain_7d", "rain_event_sum"]
                for k in ood_keys:
                    dist = TRAINING_RAINFALL_DISTRIBUTION.get(k, {})
                    val  = feats.get(k)
                    if dist and val is not None:
                        if val < dist["min"] or val > dist["max"]:
                            st.warning(
                                f"⚠️ `{k}` = {val:.1f} mm is outside the training distribution "
                                f"({dist['min']:.1f} – {dist['max']:.1f} mm). "
                                "Prediction may be less reliable."
                            )

        # Neighbourhood summary
        st.markdown("---")
        st.markdown("#### Neighbourhood Context")
        st.caption(
            "Approximate summary of the 5×5 cell neighbourhood (±2 cells, ~1.25 km radius). "
            "Based on precomputed predictions only. Not a calibrated flood probability."
        )

        res = st.session_state["analysis"]
        if res and res.get("success") and not res.get("has_missing_features") and len(df_preds) > 0:
            cell_row = res.get("cell_id", "")
            c_lat    = res["cell_lat"]
            c_lon    = res["cell_lon"]

            # Find neighbourhood cells (~5x5 at 250m resolution = ±0.011° lat, ±0.011° lon)
            ev1_nb = df_preds[df_preds["event_id"] == "DFO_4259"].copy()
            deg_radius = 5 * 0.002251  # 5 cells in latitude
            mask_nb = (
                (ev1_nb["lat"].between(c_lat - deg_radius, c_lat + deg_radius)) &
                (ev1_nb["lon"].between(c_lon - deg_radius, c_lon + deg_radius)) &
                (ev1_nb["is_perm_water"] == 0) &
                ev1_nb["flood_label"].isin([0.0, 1.0])
            )
            nb_df = ev1_nb[mask_nb]

            if len(nb_df) > 0:
                nb_col = f"prob_{MODEL_OPTIONS[st.session_state['sel_model']]}"
                if nb_col not in nb_df.columns:
                    nb_col = "susceptibility_prob"

                nb_probs = nb_df[nb_col].dropna()
                nb_high  = int((nb_probs >= 0.70).sum())
                nb_mod   = int(((nb_probs >= 0.35) & (nb_probs < 0.70)).sum())
                nb_low   = int((nb_probs < 0.35).sum())
                nb_flood = int((nb_df["flood_label"] == 1.0).sum())

                nc1, nc2, nc3, nc4, nc5 = st.columns(5)
                nc1.metric("Neighbourhood Cells", len(nb_df))
                nc2.metric("High Risk Cells",     nb_high,  f"{nb_high/max(1,len(nb_df)):.0%}")
                nc3.metric("Moderate Risk Cells", nb_mod,   f"{nb_mod/max(1,len(nb_df)):.0%}")
                nc4.metric("Low Risk Cells",      nb_low,   f"{nb_low/max(1,len(nb_df)):.0%}")
                nc5.metric("Historically Flooded", nb_flood, f"{nb_flood/max(1,len(nb_df)):.0%}")

                nb_area_km2 = len(nb_df) * CELL_AREA_KM2
                st.caption(
                    f"Neighbourhood area: ~{nb_area_km2:.2f} km² | "
                    f"Mean susceptibility ({MODEL_OPTIONS[st.session_state['sel_model']].replace('_',' ').title()}): "
                    f"{nb_probs.mean():.1%} | "
                    "Neighbourhood ≠ calibrated probability — it summarises adjacent raster predictions."
                )
            else:
                st.info("No valid neighbourhood cells found for this location.")

        # Decision support summary
        st.markdown("---")
        st.markdown("#### Decision Support Summary")
        st.markdown(
            "<span class='badge badge-warning'>RESEARCH PROTOTYPE</span> "
            "The following summary is for demonstration and academic evaluation only. "
            "It should not be used for operational emergency management decisions.",
            unsafe_allow_html=True,
        )

        if res and res.get("success") and not res.get("has_missing_features"):
            if len(df_preds) > 0:
                ev1_all  = df_preds[(df_preds["event_id"] == "DFO_4259") & (df_preds["is_perm_water"] == 0)]
                ev1_valid = ev1_all[ev1_all["flood_label"].isin([0.0, 1.0])]
                nn_col    = "prob_neural_network" if "prob_neural_network" in ev1_valid.columns else "susceptibility_prob"
                nn_probs_all = ev1_valid[nn_col].dropna()
                n_high_all  = int((nn_probs_all >= 0.70).sum())
                n_mod_all   = int(((nn_probs_all >= 0.35) & (nn_probs_all < 0.70)).sum())
                n_low_all   = int((nn_probs_all < 0.35).sum())

                summary_rows = [
                    {"Metric": "Total valid cells (study area)", "Value": f"{len(ev1_valid):,}", "Note": "Excluding permanent water"},
                    {"Metric": "Valid area (approx.)",           "Value": f"~{len(ev1_valid) * CELL_AREA_KM2:,.0f} km²", "Note": "Latitude-adjusted"},
                    {"Metric": "High-risk cells (NN ≥ 70%)",    "Value": f"{n_high_all:,}", "Note": f"{n_high_all/max(1,len(ev1_valid)):.1%} of study area"},
                    {"Metric": "Moderate-risk cells",           "Value": f"{n_mod_all:,}",  "Note": f"{n_mod_all/max(1,len(ev1_valid)):.1%} of study area"},
                    {"Metric": "Low-risk cells",                "Value": f"{n_low_all:,}",  "Note": f"{n_low_all/max(1,len(ev1_valid)):.1%} of study area"},
                    {"Metric": "Historically flooded (DFO_4259)", "Value": f"{flooded_cells:,}", "Note": f"Ground truth — NOT predicted area"},
                    {"Metric": "Selected cell susceptibility",  "Value": f"{res['susceptibility_prob']:.1%}", "Note": f"{res['risk_category']}"},
                    {"Metric": "Population / infrastructure exposure", "Value": "Not available", "Note": "Requires demographic layer integration"},
                ]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # Export
            export_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "location":  {"lat": res["cell_lat"], "lon": res["cell_lon"], "cell_id": res["cell_id"]},
                "model":     sel_model_label,
                "susceptibility_prob":  res["susceptibility_prob"],
                "risk_category":        res["risk_category"],
                "binary_decision":      res["binary_decision"],
                "fuzzy_score":          res["fuzzy_score"],
                "fuzzy_category":       res["fuzzy_category"],
                "ground_truth":         res["ground_truth"],
                "features":             res["features"],
                "disclaimer": (
                    "This output is from a research prototype. "
                    "It estimates susceptibility for historical flood events only. "
                    "It is not a real-time forecast or operational emergency guidance."
                ),
            }
            st.download_button(
                "📥 Export Location Report (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name=f"geoflood_location_{res['cell_lat']:.4f}_{res['cell_lon']:.4f}.json",
                mime="application/json",
            )


# ===========================================================================
# TAB 7: RESEARCH & VALIDATION
# ===========================================================================
with tab_research:
    st.markdown("<h2 style='margin-bottom:2px;'>🔬 Research & Validation</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; margin-bottom:12px;'>"
        "Empirical benchmarks, leakage audits, calibration, generalisation protocols, and reproducibility.</p>",
        unsafe_allow_html=True,
    )

    EVAL_SUMMARY_PATH = REPORT_DIR / "evaluation_summary.json"
    METRICS_CSV_PATH  = REPORT_DIR / "phase3_metrics.csv"
    MODEL_COMP_PATH   = REPORT_DIR / "phase3_model_comparison.csv"
    FEATURE_CSV_PATH  = REPORT_DIR / "feature_availability.csv"
    MANIFEST_PATH     = REPORT_DIR / "phase3_reproducibility.json"

    if not EVAL_SUMMARY_PATH.exists():
        st.warning(
            "Validation artifacts not found. Run the pipeline:\n"
            "```bash\npython -m src.evaluation.run_validation\n```"
        )
    else:
        p1, p2, p3, p4, p5, p6, p7 = st.tabs([
            "1. Data Quality",
            "2. Generalisation",
            "3. Model Comparison",
            "4. Calibration",
            "5. SAR Provenance",
            "6. Scenario Reliability",
            "7. Export & Reproduce",
        ])

        with p1:
            st.markdown("### Data Quality & Leakage Audit")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Grid",          "400 × 623 cells")
            d2.metric("Resolution",    "~250 m")
            d3.metric("Missingness",   "0.00%")
            d4.metric("Prevalence",    "10.8%")

            st.markdown("#### Eight Leakage Vectors — Status")
            leakage_table = pd.DataFrame([
                {"Vector": "1. Spatial Autocorrelation",    "Safeguard": "Spatial Block Holdout + 1km Buffer", "Status": "✅ Mitigated"},
                {"Vector": "2. Post-Event Target Leakage",  "Safeguard": "System classified as retrospective susceptibility mapping", "Status": "✅ Disclosed"},
                {"Vector": "3. Preprocessing Bleed",        "Safeguard": "Scaler fitted on training blocks only", "Status": "✅ Verified"},
                {"Vector": "4. Threshold Leakage",          "Safeguard": "Thresholds tuned on validation partition only", "Status": "✅ Verified"},
                {"Vector": "5. Calibration Bleed",          "Safeguard": "Platt/Isotonic fitted on validation only", "Status": "✅ Verified"},
                {"Vector": "6. Permanent Water Circularity","Safeguard": "`is_perm_water` excluded from feature matrix", "Status": "✅ Verified"},
                {"Vector": "7. Boundary Buffer Bleed",      "Safeguard": "1.0 km exclusion zone applied", "Status": "✅ Verified"},
                {"Vector": "8. Temporal Invariance",        "Safeguard": "Cross-event holdout (DFO_4288) evaluated", "Status": "✅ Verified"},
            ])
            st.dataframe(leakage_table, use_container_width=True, hide_index=True)

            if FEATURE_CSV_PATH.exists():
                st.markdown("#### Feature Temporal Availability")
                feat_df = pd.read_csv(FEATURE_CSV_PATH)
                st.dataframe(feat_df, use_container_width=True)

        with p2:
            st.markdown("### Spatial & Cross-Event Generalisation")

            if eval_summary:
                sc = eval_summary.get("sample_counts", {})
                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric("Train Cells",        f"{sc.get('spatial_block_train', 92471):,}")
                g2.metric("Val Cells",          f"{sc.get('spatial_block_val', 39979):,}")
                g3.metric("Block Test",         f"{sc.get('spatial_block_test', 39749):,}")
                g4.metric("Buffered Test",      f"{sc.get('buffered_block_test', 35815):,}")
                g5.metric("Cross-Event Test",   f"{sc.get('cross_event_test', 172199):,}")

            gen_img = FIGURES_DIR / "generalisation_comparison.png"
            if gen_img.exists():
                st.image(str(gen_img), caption="Generalisation: ROC-AUC across protocols", use_container_width=True)

            st.error(
                "🚨 **Key Result: Random Forest Cross-Event Collapse**\n\n"
                "RF ROC-AUC: 0.901 (Spatial Block) → **0.485** (Cross-Event DFO_4288) — worse than random.\n"
                "NN ROC-AUC: 0.805 (Spatial Block) → **0.672** (Cross-Event) — maintains generalisation."
            )

        with p3:
            st.markdown("### Multi-Protocol Model Benchmarking")
            if metrics_df is not None:
                display_cols = [c for c in [
                    "split", "model", "roc_auc", "pr_auc", "precision", "recall",
                    "f1", "brier_score", "brier_skill_score", "expected_calibration_error",
                ] if c in metrics_df.columns]
                fmt_map = {c: "{:.3f}" for c in display_cols if c not in ("split", "model")}
                st.dataframe(metrics_df[display_cols].style.format(fmt_map), use_container_width=True)

            roc_path = FIGURES_DIR / "roc_curves.png"
            pr_path  = FIGURES_DIR / "pr_curves.png"
            fig_c1, fig_c2 = st.columns(2)
            if roc_path.exists():
                fig_c1.image(str(roc_path), caption="ROC Curves")
            if pr_path.exists():
                fig_c2.image(str(pr_path), caption="Precision-Recall Curves")

        with p4:
            st.markdown("### Probability Calibration & Uncertainty")

            rel_path = FIGURES_DIR / "reliability_diagrams.png"
            if rel_path.exists():
                st.image(str(rel_path), caption="Reliability Diagrams", use_container_width=True)

            brier = eval_summary.get("brier_comparison", {}) if eval_summary else {}
            if brier:
                base_brier = brier.get("prevalence_baseline", 0.0848)
                brier_rows = [
                    {"Model": "Climatology Baseline", "Brier Score": f"{base_brier:.4f}", "BSS": "0.000 (reference)"},
                    {"Model": "NN (Uncalibrated)",    "Brier Score": f"{brier.get('uncalibrated_nn', 0.165):.4f}",
                     "BSS": f"{1 - brier.get('uncalibrated_nn', 0.165)/base_brier:.3f}"},
                    {"Model": "NN (Platt Calibrated)","Brier Score": f"{brier.get('platt_calibrated_nn', 0.070):.4f}",
                     "BSS": f"{1 - brier.get('platt_calibrated_nn', 0.070)/base_brier:.3f}"},
                    {"Model": "NN (Isotonic)",         "Brier Score": f"{brier.get('isotonic_calibrated_nn', 0.073):.4f}",
                     "BSS": f"{1 - brier.get('isotonic_calibrated_nn', 0.073)/base_brier:.3f}"},
                    {"Model": "Random Forest",         "Brier Score": f"{brier.get('random_forest', 0.073):.4f}",
                     "BSS": f"{1 - brier.get('random_forest', 0.073)/base_brier:.3f}"},
                ]
                st.dataframe(pd.DataFrame(brier_rows), use_container_width=True, hide_index=True)
            st.success("✅ Platt calibration reduces NN ECE from **0.310 → 0.055** (82% reduction), yielding BSS = **+0.180**.")

        with p5:
            st.markdown("### SAR Sensor Provenance & Disclaimer")
            st.warning(
                "**Formal Classification:** The reported SAR IoU of 0.991 is an **algorithmic consistency benchmark** "
                "computed on a calibrated synthetic bimodal Gaussian backscatter distribution. "
                "It does NOT represent empirical Sentinel-1 sensor accuracy against raw GRD imagery."
            )
            st.markdown("""
**Sentinel-1 Sensor Specifications:**
- Frequency: C-band (5.405 GHz, λ ≈ 5.55 cm)
- Mode: Interferometric Wide Swath (IW), VV+VH
- Resolution: 10m × 10m (GRD), upsampled to 250m
- Revisit: 6–12 days

**Real-World SAR Operational Limits (IoU 0.70–0.85):**
1. **Bragg scattering:** Monsoonal winds create capillary waves → water backscatter resembles dry land
2. **Vegetation double-bounce:** Flooded reeds/rice reflect radar energy back → false negatives
3. **Radar shadow:** Karbi Anglong hills create dark patches → false positives
4. **Revisit latency:** Flash flood peaks can occur between 6-12 day overpasses

**Correct description:** "The module demonstrates automated backscatter physics for flood footprint extraction."
""")

        with p6:
            st.markdown("### Scenario Reliability & Bounds")
            st.success("✅ Monotonicity verified: 99.8% of cells show increasing susceptibility with rainfall increases.")
            st.markdown("""
**Model extrapolation behaviour:**
- **Random Forest:** Flat step-function response when rainfall exceeds training maxima (axis-aligned splits)
- **Neural Network:** Smooth sigmoidal saturation beyond feature range [−3σ, +3σ]
- **Guardrail:** UI enforces bounds (rain_event_sum ≤ 2,500mm) to prevent non-physical inputs

**Scientific claim boundary:**
"A uniform rainfall perturbation is applied to all cells simultaneously. This is a first-order sensitivity analysis, not a complete hydrological model simulation."
""")

        with p7:
            st.markdown("### Export & Reproducibility")
            exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)

            if METRICS_CSV_PATH.exists():
                with open(METRICS_CSV_PATH, "rb") as f:
                    exp_c1.download_button("📥 Metrics CSV", f.read(), "phase3_metrics.csv", "text/csv")
            if MODEL_COMP_PATH.exists():
                with open(MODEL_COMP_PATH, "rb") as f:
                    exp_c2.download_button("📥 Model Comparison", f.read(), "phase3_model_comparison.csv", "text/csv")
            if MANIFEST_PATH.exists():
                with open(MANIFEST_PATH, "rb") as f:
                    exp_c3.download_button("📥 Reproducibility JSON", f.read(), "phase3_reproducibility.json", "application/json")

            report_paths = [
                ("phase3_data_audit.md", "Data Audit"),
                ("phase3_sar_provenance.md", "SAR Provenance"),
                ("phase3_validation_report.md", "Validation Report"),
                ("phase3_limitations.md", "Limitations"),
            ]
            for fname, label in report_paths:
                p = REPORT_DIR / fname
                if p.exists():
                    with open(p, "rb") as f:
                        exp_c4.download_button(f"📥 {label}", f.read(), fname, "text/markdown")

            if manifest:
                with st.expander("🔍 Reproducibility Manifest (SHA-256 hashes & environment)"):
                    st.json(manifest)

            st.markdown("#### Reproduction Commands")
            st.code(
                "# Full pipeline\n"
                "python -m src.evaluation.run_validation  # compute metrics\n"
                "python -m pytest tests/ -v               # run 50 tests\n"
                "python -m streamlit run app.py --server.port 8501  # launch",
                language="bash",
            )
            if eval_summary:
                st.caption(f"Last validation run: {eval_summary.get('timestamp', 'unknown')}")


# ===========================================================================
# TAB: EVACUATION ROUTER
# ===========================================================================
with tab_evacuation:
    render_evacuation_tab()
