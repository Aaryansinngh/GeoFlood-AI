"""
GeoFlood AI - Data and Methodology
Page 3: Data Integrity, Leakage Prevention, & Methodology
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(
    page_title="GeoFlood AI | Data & Methodology",
    page_icon="🔬",
    layout="wide"
)

@st.cache_data
def load_feature_data():
    proc_file = Path("data/processed/features.parquet")
    sample_file = Path("data/sample/features_sample.parquet")
    is_sample = False

    if proc_file.exists():
        df = pd.read_parquet(proc_file)
    elif sample_file.exists():
        df = pd.read_parquet(sample_file)
        is_sample = True
    else:
        df = pd.DataFrame()

    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)

    return df, config, is_sample


df, config, is_sample = load_feature_data()

st.title("🔬 Data & Methodology")
st.markdown(
    "Transparent documentation of the end-to-end data processing, feature engineering, "
    "spatial partitioning, leakage controls, and operational limitations."
)

# Prominent Badges
badge_col1, badge_col2, badge_col3, badge_col4 = st.columns(4)
with badge_col1:
    st.info(f"**Study Region:**\n{config['region']['description']}")
with badge_col2:
    st.info("**Resolution & CRS:**\n~250m Grid (EPSG:4326)")
with badge_col3:
    st.info("**Date Ranges:**\nJune 2–29, 2015 & Aug 13–Sep 11, 2015")
with badge_col4:
    if is_sample:
        st.warning("**Dataset Mode:**\nSAMPLE DATA (Subset active)")
    else:
        st.success("**Dataset Mode:**\nFULL SATELLITE REPOSITORY")

# Section 1: Dataset Preview & Distributions
st.subheader("1. Feature Table Preview & Statistics")

if len(df) > 0:
    st.write(f"Displaying sample records from feature table (`{len(df):,}` total records):")
    st.dataframe(df.head(100), use_container_width=True)

    # Missing Value Summary Table
    st.markdown("#### Missing Values & Nodata Accounting")
    st.markdown(
        "*Rule Enforcement: Unobserved or cloud-obscured cells are explicitly labeled as **Unavailable (NaN)** "
        "and never imputed as 'safe' or 0.*"
    )

    missing_stats = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        missing_stats.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Missing / NaN Cells": n_missing,
            "Missing Ratio": f"{n_missing / len(df):.2%}",
            "Integrity Status": "Masked as Unavailable" if n_missing > 0 else "Complete"
        })
    st.dataframe(pd.DataFrame(missing_stats), use_container_width=True, hide_index=True)

    # Feature Distributions
    st.markdown("#### Feature Distributions")
    num_cols = ["elevation", "slope", "relative_elevation", "tpi", "rain_event_sum", "rain_3d"]
    sel_feat = st.selectbox("Select feature to view distribution:", num_cols, index=0)

    fig_dist = px.histogram(
        df[df[sel_feat].notna()],
        x=sel_feat,
        color="flood_label",
        barmode="overlay",
        marginal="box",
        labels={"flood_label": "Ground Truth (1=Flooded, 0=Dry)"},
        title=f"Distribution of {sel_feat} conditioned on Flood Inundation",
        color_discrete_map={0.0: "#2b5c8f", 1.0: "#d90429", -1.0: "#0077b6"}
    )
    fig_dist.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_dist, use_container_width=True)

# Section 2: Methodology Walkthrough
st.markdown("---")
st.subheader("2. Methodology & Academic Design Principles")

col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown("""
    #### A. Single Fixed Prediction Target
    - **Target Definition:** For each ~250m grid cell in the Brahmaputra study region, estimate the probability that the cell was inundated during the documented historical event, given antecedent rainfall and terrain characteristics.
    - **Classification:** Binary ($y_i \in \{0, 1\}$).
    - **Scope Boundary:** This is an event-based susceptibility assessment, **NOT** an early warning system or advance real-time forecast.

    #### B. Spatial Block Holdout Split
    - Random cell-level train/test splits suffer from severe **spatial autocorrelation (near-duplicate leakage)**, inflating validation scores artificially.
    - To prevent this, the study region is partitioned into **12 contiguous spatial blocks** (3 rows × 4 columns).
    - Blocks `[2, 7, 10]` are reserved exclusively for the test set, guaranteeing zero geographic overlap with training cells.
    """)

with col_m2:
    st.markdown("""
    #### C. Strict Leakage Controls
    1. **Zero Label-Derived Input Features:** No distance-to-flooded-area, duration, or downstream flood extent features enter the models.
    2. **Permanent Water Masking:** Perennial river channels (`is_perm_water == 1`, from JRC Surface Water) are excluded from binary classification training so models do not merely learn open water optics.
    3. **Antecedent Rainfall Separation:** 3-day and 7-day rainfall metrics are calculated strictly prior to the flood event onset date ($t < t_{\text{start}}$).
    4. **Threshold Selection on Validation Only:** All decision thresholds are tuned exclusively on the validation set.
    """)

# Section 3: Limitations Box
st.markdown("---")
st.subheader("3. Limitations & Constraints")

st.markdown("""
> [!WARNING] Honest Limitations of this Prototype
> 1. **Spatial Resolution (~250m):** While appropriate for regional macro-susceptibility, a 250m grid does not capture micro-topographic features such as roadway embankments, culverts, drainage ditches, and localized flood protection levees.
> 2. **Optical MODIS Cloud Obscuration:** Peak monsoon deluge in Assam is accompanied by thick overcast clouds. Although the Global Flood Database utilizes 3-day composites, brief flash-flood pulses under heavy cloud cover may be omitted from optical observation.
> 3. **Hydraulic Dynamics Not Simulated:** This is an ML susceptibility model, not a physical 2D hydrodynamic solver (e.g. HEC-RAS or LISFLOOD-FP). It does not explicitly model mass conservation, upstream hydrograph discharge waves, or backwater friction.
> 4. **No Real-Time Lead Time:** Features are conditioned on observed event rainfall. It cannot forecast inundation weeks in advance without forecasted numerical weather prediction (NWP) precipitation grids.
""")

st.markdown("---")
st.subheader("4. Primary References")
st.markdown("""
1. **Tellman, B. et al.** (2021). *Satellite imaging reveals increased proportion of population at risk of floods.* **Nature**, 596, 355–359.
2. **Farr, T. G. et al.** (2007). *The Shuttle Radar Topography Mission.* **Reviews of Geophysics**, 45(2).
3. **Pekel, J. F. et al.** (2016). *High-resolution mapping of global surface water and its long-term changes.* **Nature**, 540, 418–422.
4. **Huffman, G. J. et al.** (2020). *Integrated Multi-satellitE Retrievals for GPM (IMERG) Technical Documentation.* NASA GES DISC.
""")
