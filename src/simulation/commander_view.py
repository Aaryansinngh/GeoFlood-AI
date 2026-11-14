"""
src/simulation/commander_view.py

Streamlit view and interactive map renderer for the Flood Commander simulation tab.
"""

from __future__ import annotations

import folium
from folium.raster_layers import ImageOverlay
import streamlit as st
import numpy as np
import pandas as pd
from streamlit_folium import st_folium

from src.simulation.disaster_commander import (
    FloodCommanderSim,
    MissionState,
    KEY_SECTORS,
    RESOURCE_COSTS,
)
from src.ui.map_builder import build_base_map, add_study_boundary


def render_commander_map(
    sim: FloodCommanderSim,
    bounds: list,
    center_lat: float,
    center_lon: float,
    west: float,
    south: float,
    east: float,
    north: float,
    rasters: dict,
) -> None:
    """Render interactive map with sectors, resource markers, and optional flood layers."""
    m = build_base_map(bounds, center_lat, center_lon, tiles="OpenStreetMap")
    m = add_study_boundary(m, bounds, west, south, east, north)

    # Optional model susceptibility layer
    if "suscep_neural_network" in rasters:
        ImageOverlay(
            image=rasters["suscep_neural_network"],
            bounds=bounds,
            opacity=0.45,
            name="Susceptibility Prior (Advisory)",
        ).add_to(m)

    # In Phase >= 3 or after mission, display observed extent
    if sim.state.current_phase >= 3 and "observed" in rasters:
        ImageOverlay(
            image=rasters["observed"],
            bounds=bounds,
            opacity=0.65,
            name="Confirmed Ground Inundation",
        ).add_to(m)

    # Render sector markers and resource icons
    for s_id, sector in KEY_SECTORS.items():
        allocs = sim.state.resources_allocated.get(s_id, {})
        r_teams = allocs.get("rescue_teams", 0)
        barriers = allocs.get("flood_barriers", 0)
        meds = allocs.get("medical_units", 0)
        rations = allocs.get("relief_rations", 0)

        # Color based on ML susceptibility
        if sector.base_susceptibility >= 0.70:
            circle_color = "#ef4444"  # Red
        elif sector.base_susceptibility >= 0.40:
            circle_color = "#f59e0b"  # Amber
        else:
            circle_color = "#10b981"  # Emerald

        popup_html = (
            f"<b>{sector.name}</b><br>"
            f"Population: {sector.population:,}<br>"
            f"Terrain: {sector.terrain_desc}<br>"
            f"Susceptibility: {sector.base_susceptibility:.1%}<br>"
            f"<hr style='margin:4px 0;'>"
            f"<b>Deployed Assets:</b><br>"
            f"• Rescue Teams: {r_teams}<br>"
            f"• Barriers (km): {barriers}<br>"
            f"• Medical Clinics: {meds}<br>"
            f"• Relief Kits: {rations}"
        )

        folium.CircleMarker(
            location=[sector.lat, sector.lon],
            radius=12 + (r_teams + barriers) * 2,
            color=circle_color,
            fill=True,
            fill_color=circle_color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{sector.name} ({sector.population:,} pop)",
        ).add_to(m)

        # If resources are deployed, add an asset marker
        total_assets = r_teams + barriers + meds + rations
        if total_assets > 0:
            folium.Marker(
                location=[sector.lat, sector.lon],
                icon=folium.Icon(color="blue", icon="shield", prefix="fa"),
                tooltip=f"Command Post: {total_assets} units deployed",
            ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    st_folium(m, use_container_width=True, height=480, returned_objects=[])


def render_commander_tab(
    bounds: list,
    center_lat: float,
    center_lon: float,
    west: float,
    south: float,
    east: float,
    north: float,
    rasters: dict,
) -> None:
    """Main tab renderer for Flood Commander interface."""
    # Ensure sim engine in session state
    if "commander_sim" not in st.session_state:
        st.session_state["commander_sim"] = FloodCommanderSim()

    sim: FloodCommanderSim = st.session_state["commander_sim"]

    # Header & Command Banner
    st.markdown(
        "<div style='background:#0f172a; border-left: 5px solid #00d4ff; padding: 14px 18px; border-radius: 6px; margin-bottom: 16px;'>"
        "<h2 style='color:#38bdf8; margin:0;'>🛡️ Disaster Response Commander (Scenario Lab)</h2>"
        "<p style='color:#94a3b8; font-size:13px; margin:4px 0 0 0;'>"
        "Professional Incident Simulation • Event: <b>Historical Brahmaputra Surge (June 2015, DFO_4259)</b> • "
        "<span style='color:#f43f5e; font-weight:600;'>Academic Simulation Only</span></p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Disclaimer note
    st.info(
        "ℹ️ **Operational Distinction**: This simulation uses genuine historical flood extent and "
        "trained ML susceptibility priors to simulate emergency resource staging decisions. "
        "It is designed for training and scenario evaluation, not real-time civil protection dispatch."
    )

    # Metrics ribbon
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Timeline", f"Phase {sim.state.current_phase}/4")
    spent = sim.state.spent_budget
    total = sim.state.total_budget
    rem = total - spent
    c2.metric("Emergency Budget", f"${rem:,}", f"Spent: ${spent:,}")
    total_assets = sum(
        sum(v.values()) for v in sim.state.resources_allocated.values()
    )
    c3.metric("Deployed Assets", f"{total_assets} units")
    status_str = "MISSION COMPLETED" if sim.state.mission_complete else "OPERATIONS ACTIVE"
    c4.metric("Incident Status", status_str)

    st.markdown("---")

    # Layout: Left column is Map + Situation Report; Right column is Command Action Panel
    col_map, col_ops = st.columns([3, 2])

    with col_map:
        st.markdown("#### Tactical Geospatial Overview")
        render_commander_map(
            sim, bounds, center_lat, center_lon, west, south, east, north, rasters
        )

        # Timeline indicator
        phase_names = [
            "1. Pre-Monsoon Watch (t - 48h)",
            "2. Inundation Onset (t 0h)",
            "3. River Crest Peak (t + 24h)",
            "4. Post-Flood Stabilization (t + 72h)",
        ]
        curr_p = sim.state.current_phase
        st.markdown(f"**Current Phase**: `{phase_names[curr_p - 1]}`")

    with col_ops:
        st.markdown("#### Incident Action Panel")

        if not sim.state.mission_complete:
            st.markdown(
                "<p style='font-size:12px; color:#94a3b8;'>Allocate emergency response assets to vulnerable sectors based on "
                "terrain slope, elevation, and ML flood susceptibility priors.</p>",
                unsafe_allow_html=True,
            )

            # Sector selector
            selected_s_id = st.selectbox(
                "Select Operational Sector",
                options=list(KEY_SECTORS.keys()),
                format_func=lambda s: KEY_SECTORS[s].name,
            )
            s_info = KEY_SECTORS[selected_s_id]

            st.markdown(
                f"<div style='background:#1e293b; padding:10px 14px; border-radius:6px; margin-bottom:12px; font-size:13px;'>"
                f"<b>Sector Profile:</b> {s_info.name}<br>"
                f"• Population: <b>{s_info.population:,}</b><br>"
                f"• Key Infrastructure: {s_info.critical_infra}<br>"
                f"• Terrain: {s_info.terrain_desc}<br>"
                f"• Model Susceptibility: <b style='color:#38bdf8;'>{s_info.base_susceptibility:.1%}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

            current_allocs = sim.state.resources_allocated.get(selected_s_id, {})

            # Resource sliders
            new_boats = st.slider(
                f"Rescue Boat Teams (${RESOURCE_COSTS['rescue_teams']:,} ea)",
                min_value=0, max_value=5, value=current_allocs.get("rescue_teams", 0),
                key=f"b_{selected_s_id}",
            )
            new_barriers = st.slider(
                f"Flood Defense Barriers km (${RESOURCE_COSTS['flood_barriers']:,} ea)",
                min_value=0, max_value=4, value=current_allocs.get("flood_barriers", 0),
                key=f"bar_{selected_s_id}",
            )
            new_med = st.slider(
                f"Mobile Medical Units (${RESOURCE_COSTS['medical_units']:,} ea)",
                min_value=0, max_value=3, value=current_allocs.get("medical_units", 0),
                key=f"med_{selected_s_id}",
            )
            new_rations = st.slider(
                f"Relief Supply Kits (${RESOURCE_COSTS['relief_rations']:,} ea)",
                min_value=0, max_value=5, value=current_allocs.get("relief_rations", 0),
                key=f"rat_{selected_s_id}",
            )

            if st.button("💾 Commit Sector Allocation", type="secondary", use_container_width=True):
                ok1, msg1 = sim.allocate_resource(selected_s_id, "rescue_teams", new_boats)
                ok2, msg2 = sim.allocate_resource(selected_s_id, "flood_barriers", new_barriers)
                ok3, msg3 = sim.allocate_resource(selected_s_id, "medical_units", new_med)
                ok4, msg4 = sim.allocate_resource(selected_s_id, "relief_rations", new_rations)
                if not (ok1 and ok2 and ok3 and ok4):
                    st.error("Budget limit reached! Adjust units.")
                else:
                    st.success("Sector deployment committed.")
                    st.rerun()

            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("⏩ Advance Incident Phase", type="primary", use_container_width=True):
                    res = sim.advance_phase()
                    if res["advanced"]:
                        st.info(res["message"])
                    st.rerun()
            with btn_col2:
                if st.button("🔄 Reset Scenario", use_container_width=True):
                    sim.reset()
                    st.rerun()

        else:
            # Mission complete - Show Scorecard & AAR
            rep = sim.state.eval_report
            st.markdown(
                f"<div style='background:#1e293b; border: 2px solid #38bdf8; padding: 14px; border-radius:8px; text-align:center;'>"
                f"<h3 style='margin:0; color:#38bdf8;'>Mission Score: {rep.get('overall_score', 0)} / 100</h3>"
                f"<p style='margin:4px 0 0 0; color:#cbd5e1; font-weight:600;'>{rep.get('rank', 'Evaluating...')}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(f"**Vulnerable Citizens Protected:** `{rep.get('citizens_protected', 0):,}`")
            st.markdown(f"**Emergency Capital Spent:** `${rep.get('budget_spent', 0):,}`")

            if st.button("🔄 Play Again / Re-run Simulation", use_container_width=True):
                sim.reset()
                st.rerun()

    # Bottom After-Action Report (AAR) when mission complete
    if sim.state.mission_complete:
        st.markdown("---")
        st.markdown("### 📋 Incident After-Action Report (AAR)")

        rep = sim.state.eval_report
        col_aar1, col_aar2, col_aar3 = st.columns(3)

        with col_aar1:
            st.markdown("##### 🟢 Tactical Strengths")
            for s in rep.get("strengths", []):
                st.markdown(f"- {s}")
            if not rep.get("strengths"):
                st.caption("No notable strengths recorded.")

        with col_aar2:
            st.markdown("##### 🔴 Strategic Vulnerabilities")
            for m in rep.get("mistakes", []):
                st.markdown(f"- {m}")
            if not rep.get("mistakes"):
                st.caption("No critical allocation errors detected.")

        with col_aar3:
            st.markdown("##### 💡 Advisory Recommendations")
            for r in rep.get("recommendations", []):
                st.markdown(f"- {r}")

        # Detailed sector breakdown table
        st.markdown("##### Sector-by-Sector Inundation & Deployment Analysis")
        details = rep.get("sector_details", [])
        if details:
            rows = []
            for d in details:
                rows.append({
                    "Sector": d["sector_name"],
                    "Actually Flooded": "YES" if d["was_flooded"] else "NO",
                    "ML Susceptibility Prior": f"{d['model_susceptibility']:.1%}",
                    "Rescue Boats": d["resources"].get("rescue_teams", 0),
                    "Barriers (km)": d["resources"].get("flood_barriers", 0),
                    "Medical Clinics": d["resources"].get("medical_units", 0),
                    "Relief Kits": d["resources"].get("relief_rations", 0),
                    "Performance": d["feedback"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
