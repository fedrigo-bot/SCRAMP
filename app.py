"""
Streamlit Main Application (`app.py`)
SCRAM-Predictor: IRRS & SMR Cooling-Water Hazard Screening Tool
"""

from __future__ import annotations

import copy
import importlib
import json
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import advisor_hook as _advisor_mod
import config as _config_mod
import satellite_engine as _sat_mod

importlib.reload(_config_mod)
importlib.reload(_sat_mod)
importlib.reload(_advisor_mod)

from advisor_hook import (
    answer_irrs_inspector_question,
    evaluate_pass_dual_gate,
    generate_site_baseline_report,
    resolve_gemini_api_key,
    run_rag_regulatory_analysis,
    synthesize_hybrid_rag_cards,
)
from config import (
    DEFAULT_CDSE_CLIENT_ID,
    DEFAULT_CDSE_CLIENT_SECRET,
    MIN_VALID_WATER_PCT,
    NDCI_THRESHOLDS,
    NDTI_THRESHOLDS,
    PRESET_SITES,
    SST_THRESHOLDS,
    TOOL_NAME,
    TOOL_VERSION,
    ZSCORE_THRESHOLDS,
    compute_bounding_box,
)
from satellite_engine import CopernicusWaterMonitor, resolve_cdse_credentials


st.set_page_config(
    page_title="SCRAM-Predictor | IAEA IRRS & SMR Environmental Screening",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_theme_css(dark_mode: bool) -> None:
    """
    Inject CSS to allow seamless switching between Dark Mode and Light Mode
    directly from the sidebar toggle, including full contrast fixes for Streamlit Cloud
    when base theme is light.
    """
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            [data-testid="stSidebar"] {
                background-color: #161B22;
                color: #E6EDF3;
            }
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div {
                color: #E6EDF3;
            }
            [data-testid="stHeader"] {
                background-color: rgba(14, 17, 23, 0.85);
            }
            /* Soften inline code pills instead of glaring white boxes */
            .stApp code, .stApp pre {
                background-color: #1E293B !important;
                color: #93C5FD !important;
                border: 1px solid #334155 !important;
                border-radius: 5px;
                padding: 0.12rem 0.4rem;
            }
            /* Ensure KPI cards & markdown remain legible on dark background */
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"] {
                color: #F8FAFC !important;
            }
            [data-testid="stCaptionContainer"] {
                color: #94A3B8 !important;
            }
            /* Soften sidebar status alerts in dark mode */
            [data-testid="stSidebar"] [data-testid="stAlert"],
            [data-testid="stSidebar"] [data-testid="stAlert"] > div,
            [data-testid="stSidebar"] [role="alert"] {
                background-color: rgba(16, 185, 129, 0.14) !important;
                border-color: rgba(16, 185, 129, 0.40) !important;
            }
            [data-testid="stSidebar"] [data-testid="stAlert"] p {
                color: #6EE7B7 !important;
            }
            /* Darken sidebar & main inputs/widgets with crisp visible borders */
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-baseweb="input"],
            [data-testid="stSidebar"] [data-baseweb="base-input"] {
                background-color: #1E293B !important;
                color: #F1F5F9 !important;
                border: 1.5px solid #64748B !important;
                border-radius: 6px !important;
                -webkit-text-fill-color: #F1F5F9 !important;
            }
            [data-testid="stSidebar"] input {
                background-color: #1E293B !important;
                color: #F1F5F9 !important;
                -webkit-text-fill-color: #F1F5F9 !important;
            }
            [data-testid="stRadio"] p,
            [data-testid="stRadio"] label,
            [data-testid="stWidgetLabel"] p {
                color: #E2E8F0 !important;
            }
            [data-testid="stTextInput"] button,
            button[aria-label*="password"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #F8FAFC;
                color: #0F172A;
            }
            [data-testid="stSidebar"] {
                background-color: #EEF2F6;
                color: #0F172A;
            }
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] div,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #0F172A !important;
            }
            [data-testid="stHeader"] {
                background-color: rgba(248, 250, 252, 0.90);
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li {
                color: #0F172A;
            }
            .stApp code, .stApp pre {
                background-color: #E2E8F0 !important;
                color: #0F172A !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 5px;
                padding: 0.12rem 0.4rem;
            }
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"] {
                color: #0F172A !important;
            }
            [data-testid="stMetricDelta"] {
                background-color: #E2E8F0 !important;
                color: #0F172A !important;
                border: 1px solid #CBD5E1 !important;
                border-radius: 999px !important;
                padding: 0.15rem 0.55rem !important;
            }
            [data-testid="stMetricDelta"] > div,
            [data-testid="stMetricDelta"] span {
                color: #0F172A !important;
            }
            [data-testid="stMetricDelta"] svg {
                fill: #0F172A !important;
                color: #0F172A !important;
            }
            [data-testid="stCaptionContainer"] {
                color: #475569 !important;
            }
            [data-testid="stRadio"] p,
            [data-testid="stRadio"] label,
            [data-testid="stWidgetLabel"] p,
            [data-testid="stTabs"] button p,
            [data-testid="stExpander"] summary p {
                color: #0F172A !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-baseweb="input"],
            [data-testid="stSidebar"] [data-baseweb="base-input"] {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1.5px solid #64748B !important;
                border-radius: 6px !important;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
                -webkit-text-fill-color: #0F172A !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] > div > div,
            [data-testid="stSidebar"] input {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                -webkit-text-fill-color: #0F172A !important;
            }
            [data-testid="stSidebar"] [data-baseweb="select"] svg {
                fill: #0F172A !important;
                color: #0F172A !important;
            }
            [data-testid="stSidebar"] [role="switch"] {
                background-color: #94A3B8 !important;
            }
            /* Expander cards & headers in Light Mode */
            [data-testid="stExpander"],
            [data-testid="stExpander"] details {
                background-color: #FFFFFF !important;
                border-color: #CBD5E1 !important;
            }
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] details > summary {
                background-color: #E2E8F0 !important;
                color: #0F172A !important;
            }
            [data-testid="stExpander"] summary svg {
                fill: #0F172A !important;
                color: #0F172A !important;
            }
            /* Main area info/warning banners, download buttons, and JSON viewer */
            .stApp [data-testid="stAlert"],
            .stApp [data-testid="stAlert"] > div {
                background-color: #EFF6FF !important;
                border-color: #93C5FD !important;
            }
            .stApp [data-testid="stAlert"] p,
            .stApp [data-testid="stAlert"] div {
                color: #1E3A8A !important;
            }
            [data-testid="stDownloadButton"] button {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: 1px solid #CBD5E1 !important;
            }
            [data-testid="stJson"] {
                background-color: #F1F5F9 !important;
                color: #0F172A !important;
                border-radius: 6px;
                padding: 0.5rem;
            }
            [data-testid="stSidebar"] [data-testid="stAlert"],
            [data-testid="stSidebar"] [data-testid="stAlert"] > div,
            [data-testid="stSidebar"] [role="alert"] {
                background-color: #ECFDF5 !important;
                border-color: #6EE7B7 !important;
            }
            [data-testid="stSidebar"] [data-testid="stAlert"] p {
                color: #065F46 !important;
            }
            [data-testid="stTextInput"] button,
            button[aria-label*="password"] {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def _build_intake_map(
    lat: float,
    lon: float,
    buffer_deg: float,
    site_name: str,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Render an interactive map Plotly figure displaying the offshore
    cooling-water intake monitoring coordinate and the Sentinel-2 extraction bounding box.
    Compatible with both Plotly 6+/7+ (`Scattermap`) and Plotly 5.x (`Scattermapbox`).
    """
    bbox = compute_bounding_box(lat=lat, lon=lon, buffer_deg=buffer_deg)
    box_lats = [bbox["south"], bbox["south"], bbox["north"], bbox["north"], bbox["south"]]
    box_lons = [bbox["west"], bbox["east"], bbox["east"], bbox["west"], bbox["west"]]

    fig = go.Figure()
    use_scattermap = hasattr(go, "Scattermap")
    TraceClass = go.Scattermap if use_scattermap else go.Scattermapbox

    # Sentinel-2 L2A spatial aggregation polygon (high-contrast cyan over satellite imagery)
    fig.add_trace(
        TraceClass(
            lat=box_lats,
            lon=box_lons,
            mode="lines",
            fill="toself",
            fillcolor="rgba(0, 240, 255, 0.18)",
            line={"color": "#00F0FF", "width": 2.5},
            name=f"Sentinel-2 Offshore ROI (±{buffer_deg}° Buffer)",
            hoverinfo="name",
        )
    )

    # Central offshore cooling-water intake coordinate
    fig.add_trace(
        TraceClass(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            marker={"size": 11, "color": "#FF3366"},
            text=[site_name.split(" (")[0]],
            textposition="top right",
            textfont={"color": "#FFFFFF", "size": 12},
            name="Offshore Monitoring Point",
            hovertemplate=(
                f"<b>{site_name}</b><br>"
                f"Latitude: {lat:.4f}°<br>"
                f"Longitude: {lon:.4f}°<extra></extra>"
            ),
        )
    )

    map_config = {
        "style": "white-bg",
        "center": {"lat": lat, "lon": lon},
        "zoom": 11.8,
        "layers": [
            {
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                "source": [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
            }
        ],
    }

    layout_kwargs = {
        "template": "plotly_dark" if dark_mode else "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "height": 270,
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 0.02,
            "bgcolor": "rgba(15, 23, 42, 0.85)",
            "bordercolor": "rgba(0, 240, 255, 0.4)",
            "borderwidth": 1,
            "font": {"color": "#F8FAFC", "size": 11},
        },
    }
    if use_scattermap:
        layout_kwargs["map"] = map_config
    else:
        layout_kwargs["mapbox"] = map_config

    fig.update_layout(**layout_kwargs)
    return fig


def _build_dual_hazard_chart(
    df: pd.DataFrame,
    site_name: str,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Build an interactive dual-subplot Plotly time-series chart for marine-calibrated
    NDCI (Trophic/Chlorophyll precursor) and NDTI (Turbidity/Detritus precursor)
    with horizontal MODERATE/HIGH thresholds, site median baseline, and Z-score tooltips.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Normalized Difference Chlorophyll Index (NDCI) — Trophic & Phytoplankton Precursor Proxy",
            "Normalized Difference Turbidity Index (NDTI) — Suspended Sediment & Organic Detritus Proxy",
            "Sea Surface Temperature (SST °C) — Sentinel-3 SLSTR / Copernicus Marine Thermal Proxy",
        ),
    )

    # --- Top Subplot: NDCI ---
    y_max_ndci = max(float(df["ndci"].max()) + 0.04, NDCI_THRESHOLDS["HIGH"] + 0.04)
    y_min_ndci = min(float(df["ndci"].min()) - 0.03, NDCI_THRESHOLDS["MODERATE"] - 0.12)
    median_ndci = float(df["ndci"].median())

    fig.add_hrect(
        y0=NDCI_THRESHOLDS["MODERATE"],
        y1=NDCI_THRESHOLDS["HIGH"],
        fillcolor="rgba(255, 165, 0, 0.11)",
        line_width=0,
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=NDCI_THRESHOLDS["HIGH"],
        y1=y_max_ndci,
        fillcolor="rgba(230, 57, 70, 0.12)",
        line_width=0,
        row=1,
        col=1,
    )

    fig.add_hline(
        y=median_ndci,
        line_dash="dot",
        line_color="#8ECAE6",
        annotation_text=f"Site Median Baseline ({median_ndci:+.3f})",
        annotation_position="bottom right",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=NDCI_THRESHOLDS["MODERATE"],
        line_dash="dash",
        line_color="#F4A261",
        annotation_text=f"MODERATE ({NDCI_THRESHOLDS['MODERATE']:+.2f})",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=NDCI_THRESHOLDS["HIGH"],
        line_dash="dash",
        line_color="#E63946",
        annotation_text=f"HIGH / Trophic Surge Watch ({NDCI_THRESHOLDS['HIGH']:+.2f})",
        annotation_position="top left",
        row=1,
        col=1,
    )

    ndci_colors = [
        "#E63946"
        if (v >= NDCI_THRESHOLDS["HIGH"] or z >= ZSCORE_THRESHOLDS["HIGH"])
        else (
            "#F4A261"
            if (v >= NDCI_THRESHOLDS["MODERATE"] or z >= ZSCORE_THRESHOLDS["MODERATE"])
            else "#2A9D8F"
        )
        for v, z in zip(df["ndci"], df["ndci_zscore"])
    ]

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ndci"],
            mode="lines+markers",
            name="NDCI (Chlorophyll-a Precursor)",
            line={"color": "#2A9D8F", "width": 2.5},
            marker={
                "size": 8,
                "color": ndci_colors,
                "line": {"width": 1, "color": "#1D3557"},
            },
            customdata=df[["ndci_zscore", "valid_water_coverage_pct"]],
            hovertemplate=(
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>NDCI:</b> %{y:+.4f} (<b>%{customdata[0]:+.2f}σ</b>)<br>"
                "<b>Valid Water Pixels:</b> %{customdata[1]:.1f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --- Bottom Subplot: NDTI ---
    y_max_ndti = max(float(df["ndti"].max()) + 0.05, NDTI_THRESHOLDS["HIGH"] + 0.05)
    y_min_ndti = min(float(df["ndti"].min()) - 0.04, NDTI_THRESHOLDS["MODERATE"] - 0.22)
    median_ndti = float(df["ndti"].median())

    fig.add_hrect(
        y0=NDTI_THRESHOLDS["MODERATE"],
        y1=NDTI_THRESHOLDS["HIGH"],
        fillcolor="rgba(255, 165, 0, 0.11)",
        line_width=0,
        row=2,
        col=1,
    )
    fig.add_hrect(
        y0=NDTI_THRESHOLDS["HIGH"],
        y1=y_max_ndti,
        fillcolor="rgba(230, 57, 70, 0.12)",
        line_width=0,
        row=2,
        col=1,
    )

    fig.add_hline(
        y=median_ndti,
        line_dash="dot",
        line_color="#8ECAE6",
        annotation_text=f"Site Median Baseline ({median_ndti:+.3f})",
        annotation_position="bottom right",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=NDTI_THRESHOLDS["MODERATE"],
        line_dash="dash",
        line_color="#F4A261",
        annotation_text=f"MODERATE ({NDTI_THRESHOLDS['MODERATE']:+.2f})",
        annotation_position="top left",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=NDTI_THRESHOLDS["HIGH"],
        line_dash="dash",
        line_color="#E63946",
        annotation_text=f"HIGH / Sediment Plume Watch ({NDTI_THRESHOLDS['HIGH']:+.2f})",
        annotation_position="top left",
        row=2,
        col=1,
    )

    ndti_colors = [
        "#E63946"
        if (v >= NDTI_THRESHOLDS["HIGH"] or z >= ZSCORE_THRESHOLDS["HIGH"])
        else (
            "#F4A261"
            if (v >= NDTI_THRESHOLDS["MODERATE"] or z >= ZSCORE_THRESHOLDS["MODERATE"])
            else "#457B9D"
        )
        for v, z in zip(df["ndti"], df["ndti_zscore"])
    ]

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ndti"],
            mode="lines+markers",
            name="NDTI (Turbidity Precursor)",
            line={"color": "#457B9D", "width": 2.5},
            marker={
                "size": 8,
                "color": ndti_colors,
                "line": {"width": 1, "color": "#1D3557"},
            },
            customdata=df[["ndti_zscore", "valid_water_coverage_pct"]],
            hovertemplate=(
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>NDTI:</b> %{y:+.4f} (<b>%{customdata[0]:+.2f}σ</b>)<br>"
                "<b>Valid Water Pixels:</b> %{customdata[1]:.1f}%<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --- Bottom Subplot: SST (thermal trigger) ---
    if "sst_celsius" in df.columns and df["sst_celsius"].notna().any():
        sst_vals = pd.to_numeric(df["sst_celsius"], errors="coerce")
        sst_z = (
            pd.to_numeric(df["sst_zscore"], errors="coerce")
            if "sst_zscore" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        y_max_sst = max(float(sst_vals.max()) + 1.5, SST_THRESHOLDS["HIGH"] + 1.5)
        y_min_sst = min(float(sst_vals.min()) - 1.5, SST_THRESHOLDS["MODERATE"] - 6.0)

        fig.add_hrect(
            y0=SST_THRESHOLDS["MODERATE"], y1=SST_THRESHOLDS["HIGH"],
            fillcolor="rgba(255, 165, 0, 0.11)", line_width=0, row=3, col=1,
        )
        fig.add_hrect(
            y0=SST_THRESHOLDS["HIGH"], y1=y_max_sst,
            fillcolor="rgba(230, 57, 70, 0.12)", line_width=0, row=3, col=1,
        )
        sst_base_med = df.attrs.get("sst_baseline_median", float(sst_vals.median()))
        sst_base_win = df.attrs.get("sst_baseline_window", "Same-Window Past-Years Baseline")
        fig.add_hline(
            y=float(sst_base_med), line_dash="dot", line_color="#8ECAE6",
            annotation_text=f"{sst_base_win} ({float(sst_base_med):.1f} °C)",
            annotation_position="bottom right", row=3, col=1,
        )
        fig.add_hline(
            y=SST_THRESHOLDS["MODERATE"], line_dash="dash", line_color="#F4A261",
            annotation_text=f"Moderate Thermal Watch ({SST_THRESHOLDS['MODERATE']:.1f} °C)",
            annotation_position="top left", row=3, col=1,
        )
        fig.add_hline(
            y=SST_THRESHOLDS["HIGH"], line_dash="dash", line_color="#E63946",
            annotation_text=f"High Thermal / Jellyfish Swarm Trigger ({SST_THRESHOLDS['HIGH']:.1f} °C)",
            annotation_position="top left", row=3, col=1,
        )
        sst_src = (
            df["sst_source"].astype(str)
            if "sst_source" in df.columns
            else pd.Series("n/a", index=df.index)
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=sst_vals,
                mode="lines+markers",
                name="SST °C (Thermal Trigger)",
                line={"color": "#f97316", "width": 2.5},
                marker={"size": 7, "color": "#ff7f50", "line": {"width": 1, "color": "#7c2d12"}},
                customdata=pd.concat([sst_z, sst_src], axis=1).to_numpy(),
                hovertemplate=(
                    "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                    "<b>SST:</b> %{y:.2f} °C (<b>%{customdata[0]:+.2f}σ</b>)<br>"
                    "<b>Source:</b> %{customdata[1]}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="SST (°C)", range=[y_min_sst, y_max_sst], row=3, col=1)
    fig.update_yaxes(title_text="NDCI ((B05-B04)/(B05+B04))", range=[y_min_ndci, y_max_ndci], row=1, col=1)
    fig.update_yaxes(title_text="NDTI ((B04-B03)/(B04+B03))", range=[y_min_ndti, y_max_ndti], row=2, col=1)
    fig.update_xaxes(title_text="Satellite Revisit Date (S2 MSI + S3 SLSTR)", row=3, col=1)

    fig.update_layout(
        title={
            "text": f"Coupled Optical (Sentinel-2 L2A) + Thermal (SST) Hazard Time Series — {site_name}",
            "x": 0.0,
            "xanchor": "left",
            "font": {"color": "#FAFAFA" if dark_mode else "#1F2937", "size": 17},
        },
        height=900,
        hovermode="x unified",
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22, 27, 34, 0.65)" if dark_mode else "rgba(255, 255, 255, 1.0)",
        font={"color": "#FAFAFA" if dark_mode else "#1F2937"},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.07,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 30, "t": 75, "b": 85},
    )
    return fig


def _build_zscore_hazard_chart(
    df: pd.DataFrame,
    site_name: str,
    dark_mode: bool = True,
) -> go.Figure:
    """
    Build a standardized non-parametric Robust Z-Score (MAD + IQR) time-series chart
    displaying both NDCI and NDTI relative anomalies against +1.50σ (MODERATE) and
    +2.20σ (HIGH) Dual-Gate threshold lines.
    """
    fig = go.Figure()

    z_cols = [c for c in ("ndci_zscore", "ndti_zscore", "sst_zscore") if c in df.columns]
    y_max_z = max(
        float(df[z_cols].max().max()) + 0.5,
        ZSCORE_THRESHOLDS["HIGH"] + 0.6,
    )
    y_min_z = min(
        float(df[z_cols].min().min()) - 0.4,
        -1.5,
    )

    fig.add_hrect(
        y0=ZSCORE_THRESHOLDS["MODERATE"],
        y1=ZSCORE_THRESHOLDS["HIGH"],
        fillcolor="rgba(255, 165, 0, 0.11)",
        line_width=0,
    )
    fig.add_hrect(
        y0=ZSCORE_THRESHOLDS["HIGH"],
        y1=y_max_z,
        fillcolor="rgba(230, 57, 70, 0.12)",
        line_width=0,
    )

    fig.add_hline(
        y=0.0,
        line_dash="dot",
        line_color="#8ECAE6",
        annotation_text="Site Median Baseline (0.0σ)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=ZSCORE_THRESHOLDS["MODERATE"],
        line_dash="dash",
        line_color="#F4A261",
        annotation_text=f"MODERATE Anomaly Gate (+{ZSCORE_THRESHOLDS['MODERATE']:.2f}σ)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=ZSCORE_THRESHOLDS["HIGH"],
        line_dash="dash",
        line_color="#E63946",
        annotation_text=f"HIGH Anomaly Gate (+{ZSCORE_THRESHOLDS['HIGH']:.2f}σ)",
        annotation_position="top left",
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ndci_zscore"],
            mode="lines+markers",
            name="NDCI Robust Z-Score (+σ Trophic Anomaly)",
            line={"color": "#2A9D8F", "width": 2.5},
            marker={"size": 8, "color": "#2A9D8F"},
            customdata=df["ndci"],
            hovertemplate=(
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>NDCI Z-Score:</b> %{y:+.2f}σ (Raw NDCI: %{customdata:+.4f})<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ndti_zscore"],
            mode="lines+markers",
            name="NDTI Robust Z-Score (+σ Turbidity Anomaly)",
            line={"color": "#457B9D", "width": 2.5},
            marker={"size": 8, "color": "#457B9D"},
            customdata=df["ndti"],
            hovertemplate=(
                "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                "<b>NDTI Z-Score:</b> %{y:+.2f}σ (Raw NDTI: %{customdata:+.4f})<extra></extra>"
            ),
        )
    )

    if "sst_zscore" in df.columns and df["sst_zscore"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["sst_zscore"],
                mode="lines+markers",
                name="SST Robust Z-Score (+σ Thermal Anomaly)",
                line={"color": "#f97316", "width": 2.5},
                marker={"size": 8, "color": "#f97316"},
                customdata=df["sst_celsius"],
                hovertemplate=(
                    "<b>Date:</b> %{x|%Y-%m-%d}<br>"
                    "<b>SST Z-Score:</b> %{y:+.2f}σ (SST: %{customdata:.2f} °C)<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title={
            "text": f"Standardized Robust Z-Score (MAD + IQR) — NDCI, NDTI & SST — {site_name}",
            "x": 0.0,
            "xanchor": "left",
            "font": {"color": "#FAFAFA" if dark_mode else "#1F2937", "size": 17},
        },
        yaxis_title="Robust Relative Anomaly (+σ above Site Median)",
        xaxis_title="Sentinel-2 L2A Revisit Date",
        yaxis={"range": [y_min_z, y_max_z]},
        height=510,
        hovermode="x unified",
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22, 27, 34, 0.65)" if dark_mode else "rgba(255, 255, 255, 1.0)",
        font={"color": "#FAFAFA" if dark_mode else "#1F2937"},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 30, "t": 65, "b": 85},
    )
    return fig


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_live_cached(
    site_name: str,
    lat_rounded: float,
    lon_rounded: float,
    start_str: str,
    end_str: str,
    buffer_deg: float,
    _client_id: str = "",
    _client_secret: str = "",
):
    """
    24-hour (86400 s) cached Copernicus CDSE retrieval keyed on site and rounded
    coordinates/dates so repeated IAEA reviewer visits consume zero duplicate API quota.
    Leading underscores on credentials exclude them from cache key hashing.
    """
    mon = CopernicusWaterMonitor()
    result_df = mon.fetch_live_openeo_data(
        lat=lat_rounded,
        lon=lon_rounded,
        buffer_deg=buffer_deg,
        start_date=start_str,
        end_date=end_str,
        client_id=_client_id or None,
        client_secret=_client_secret or None,
        site_name=site_name,
    )
    return result_df, mon.last_warning


@st.cache_data(ttl=86400, show_spinner=False)
def _synthesize_rag_cards_cached(
    site_name: str,
    start_str: str,
    end_str: str,
    risk_level: str,
    max_ndci: float,
    max_ndti: float,
    max_sst: float,
    has_gemini_key: bool,
    _report_dict: dict,
    _site_meta: dict,
    _explicit_gemini_key: str = "",
) -> dict:
    """24-hour cached Hybrid RAG synthesis for the 4 IAEA regulatory cards."""
    return synthesize_hybrid_rag_cards(
        report_dict=_report_dict,
        site_meta=_site_meta,
        explicit_gemini_key=_explicit_gemini_key or None,
    )


@st.cache_data(ttl=86400, show_spinner=False)
def _answer_inspector_qa_cached(
    question: str,
    site_name: str,
    start_str: str,
    end_str: str,
    risk_level: str,
    has_gemini_key: bool,
    _report_dict: dict,
    _site_meta: dict,
    _explicit_gemini_key: str = "",
) -> dict:
    """24-hour cached IRRS Inspector Q&A answer."""
    return answer_irrs_inspector_question(
        question=question,
        report_dict=_report_dict,
        site_meta=_site_meta,
        explicit_gemini_key=_explicit_gemini_key or None,
    )


def _sync_dates_from_months() -> None:
    """Update Start/End calendar inputs ONLY when the user clicks + / - on Last X Months."""
    months = int(st.session_state.get("lookback_months", 3))
    today = date.today()
    st.session_state["start_date_input"] = today - timedelta(days=int(months * 30.4375))
    st.session_state["end_date_input"] = today


def _load_historical_benchmark() -> None:
    """Load the Apr–Sep 2025 historical Gravelines/SMR validation benchmark into the calendars."""
    st.session_state["start_date_input"] = date(2025, 4, 1)
    st.session_state["end_date_input"] = date(2025, 9, 30)


def main() -> None:
    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown("## 🛰️ SCRAM-Predictor")
        st.caption(f"Version `{TOOL_VERSION}` | IAEA IRRS & NHSI SMR Screening")
        dark_mode = st.toggle(
            "🌙 Dark Mode",
            value=True,
            help="Passa istantaneamente dal tema scuro (Dark Mode) al tema chiaro (Light Mode).",
        )
        st.divider()

        cloud_id, cloud_secret, cred_source = resolve_cdse_credentials()
        has_cloud_secrets = cred_source in ("st.secrets", "os.environ")

        data_mode = st.radio(
            "1. Satellite Data Source",
            options=[
                "Demo / Calibrated Historical Simulation (Instant)",
                "Live Copernicus CDSE API (openEO)",
            ],
            index=1 if has_cloud_secrets else 0,
            help=(
                "Choose between instant marine-calibrated Sentinel-2 simulation "
                "or live execution against Copernicus Data Space Ecosystem (CDSE)."
            ),
        )

        client_id = cloud_id or ""
        client_secret = cloud_secret or ""
        if data_mode == "Live Copernicus CDSE API (openEO)":
            if has_cloud_secrets:
                st.success("Copernicus CDSE Live API: Connected (Protected Cloud Secrets)")
            else:
                st.info(
                    "Copernicus Data Space Ecosystem (CDSE) OAuth2 credentials required for live queries. "
                    f"Applies cloud filter (≤40%), water coverage gate (≥{MIN_VALID_WATER_PCT:.0f}%), "
                    "and dark-pixel radiometric guard."
                )
                client_id = st.text_input(
                    "CDSE Client ID",
                    value=DEFAULT_CDSE_CLIENT_ID,
                    type="password",
                    help=(
                        "Generate an OIDC Client ID in the Copernicus Data Space Dashboard "
                        "(https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings)."
                    ),
                )
                client_secret = st.text_input(
                    "CDSE Client Secret",
                    value=DEFAULT_CDSE_CLIENT_SECRET,
                    type="password",
                    help=(
                        "Copy the corresponding Client Secret generated under OAuth Clients "
                        "in your Copernicus Data Space Ecosystem account settings."
                    ),
                )

        st.divider()
        selected_site_name = st.selectbox(
            "2. Nuclear Facility / SMR Intake Site",
            options=list(PRESET_SITES.keys()),
            index=0,
        )

        site_meta = copy.deepcopy(PRESET_SITES[selected_site_name])
        is_custom_smr = "Custom" in selected_site_name

        col_lat, col_lon = st.columns(2)
        with col_lat:
            custom_lat = st.number_input(
                "Latitude (°N)",
                min_value=-85.0,
                max_value=85.0,
                value=float(site_meta["lat"]),
                step=0.005,
                format="%.4f",
                disabled=not is_custom_smr,
            )
        with col_lon:
            custom_lon = st.number_input(
                "Longitude (°E)",
                min_value=-180.0,
                max_value=180.0,
                value=float(site_meta["lon"]),
                step=0.005,
                format="%.4f",
                disabled=not is_custom_smr,
            )

        if is_custom_smr:
            site_meta["lat"] = float(custom_lat)
            site_meta["lon"] = float(custom_lon)

        st.divider()
        st.markdown("**3. Screening Window (Sentinel-2 L2A + SST)**")
        today_dt = date.today()
        st.session_state.setdefault("start_date_input", today_dt - timedelta(days=90))
        st.session_state.setdefault("end_date_input", today_dt)

        st.number_input(
            "Quick Window: Last X Months",
            min_value=1,
            max_value=24,
            value=3,
            step=1,
            key="lookback_months",
            on_change=_sync_dates_from_months,
            help="Use – / + to set the lookback window relative to today without wiping manual calendar edits on reruns.",
        )

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            start_date = st.date_input(
                "Start Date",
                key="start_date_input",
                format="DD/MM/YYYY",
            )
        with d_col2:
            end_date = st.date_input(
                "End Date",
                key="end_date_input",
                format="DD/MM/YYYY",
            )

        if end_date < start_date:
            st.warning("End Date must be on or after Start Date.")
        elif (end_date - start_date).days < 85:
            st.caption(
                "⚠️ *Windows under 3 months (<85 days) are suited for short-term operational watch "
                "but provide a limited sample size (N < 10 clear passes) for seasonal baseline medians.*"
            )

        st.button(
            "Load Historical Benchmark (Apr–Sep 2025)",
            on_click=_load_historical_benchmark,
            use_container_width=True,
        )

        run_screening = st.button(
            "Run Environmental Hazard Screening",
            type="primary",
            use_container_width=True,
        )

        gemini_env_key, gemini_source = resolve_gemini_api_key()
        explicit_gemini_key = ""
        with st.expander("🧠 Phase-2 AI RAG Settings (Optional)", expanded=False):
            if gemini_source in ("st.secrets", "os.environ"):
                st.caption("Gemini Flash RAG: Connected via Protected Secrets.")
            else:
                explicit_gemini_key = st.text_input(
                    "Gemini API Key (Optional)",
                    value="",
                    type="password",
                    help="Optional: enables Gemini Flash synthesis. Leave blank for offline Deterministic IAEA Corpus RAG.",
                )

    # --- Strict Button-Gated Execution Lifecycle ---
    pending_signature = (
        data_mode,
        selected_site_name,
        round(float(site_meta["lat"]), 4),
        round(float(site_meta["lon"]), 4),
        str(start_date),
        str(end_date),
    )

    st.session_state.setdefault("live_api_calls", 0)
    st.session_state.setdefault("live_cache_keys", set())

    is_initial_load = "active_screening_result" not in st.session_state
    if not is_initial_load and pending_signature != st.session_state["active_screening_result"]["signature"]:
        with st.sidebar:
            st.info("⏳ Window updated — click 'Run Environmental Hazard Screening' to execute scan.")

    monitor = CopernicusWaterMonitor()
    if (run_screening and end_date >= start_date) or is_initial_load:
        with st.spinner("Processing Sentinel-2 L2A reflectances, Sentinel-3/marine SST fusion & Dual-Gate Z-Scores..."):
            if data_mode == "Live Copernicus CDSE API (openEO)":
                live_key = (
                    selected_site_name,
                    round(float(site_meta["lat"]), 4),
                    round(float(site_meta["lon"]), 4),
                    str(start_date),
                    str(end_date),
                )
                is_session_cached = live_key in st.session_state["live_cache_keys"]
                if not is_session_cached and st.session_state["live_api_calls"] >= 15:
                    df = monitor.generate_demo_timeseries(
                        lat=site_meta["lat"],
                        lon=site_meta["lon"],
                        start_date=start_date,
                        end_date=end_date,
                        site_name=selected_site_name,
                    )
                    monitor.last_warning = (
                        "Session rate limit reached (15/15 uncached live Copernicus API queries used). "
                        "Displaying calibrated historical fallback data to protect shared API quota."
                    )
                else:
                    df, live_warn = _fetch_live_cached(
                        site_name=selected_site_name,
                        lat_rounded=round(float(site_meta["lat"]), 4),
                        lon_rounded=round(float(site_meta["lon"]), 4),
                        start_str=str(start_date),
                        end_str=str(end_date),
                        buffer_deg=float(site_meta["buffer_deg"]),
                        _client_id=client_id,
                        _client_secret=client_secret,
                    )
                    monitor.last_warning = live_warn
                    if not is_session_cached and not live_warn:
                        st.session_state["live_api_calls"] += 1
                    st.session_state["live_cache_keys"].add(live_key)
            else:
                df = monitor.generate_demo_timeseries(
                    lat=site_meta["lat"],
                    lon=site_meta["lon"],
                    start_date=start_date,
                    end_date=end_date,
                    site_name=selected_site_name,
                )

            report = generate_site_baseline_report(
                site_name=selected_site_name,
                site_meta=site_meta,
                df=df,
                date_range=(start_date, end_date),
            )

            st.session_state["active_screening_result"] = {
                "df": df,
                "report": report,
                "warning": monitor.last_warning,
                "signature": pending_signature,
                "site_meta": copy.deepcopy(site_meta),
            }

    active_res = st.session_state["active_screening_result"]
    df: pd.DataFrame = active_res["df"]
    report: dict = active_res["report"]
    fallback_warning = active_res.get("warning")
    active_site_meta: dict = active_res.get("site_meta", site_meta)

    _apply_theme_css(dark_mode)

    # --- Main Dashboard Header ---
    st.title(f"🛰️ {TOOL_NAME}")
    st.markdown(
        "**Copernicus Sentinel-2 L2A Trophic Precursors + Sentinel-3 SLSTR / Marine SST Thermal Trigger Screening** "
        "| Built for **IAEA IRRS Missions** (`SSR-1 Req. 20` / `SSG-18 Sec. 5`) and the **NHSI SMR Hub**."
    )

    if fallback_warning:
        if "Session rate limit reached" in fallback_warning:
            st.info(fallback_warning, icon="ℹ️")
        else:
            st.warning(fallback_warning, icon="⚠️")

    # Site Metadata & Map Row
    profile = report["site_profile"]
    metrics = report["hazard_metrics"]
    risk_info = report["risk_classification"]

    header_left, header_right = st.columns([1.25, 1.0])
    with header_left:
        st.subheader(f"📍 {profile['site_name']}")
        st.markdown(
            f"- **Facility / Deployment Type:** `{profile['reactor_type']}`\n"
            f"- **Cooling System Architecture:** `{profile['cooling_system']}`\n"
            f"- **Offshore ROI Coordinates:** `{profile['coordinates']['lat']:.4f}° N, "
            f"{profile['coordinates']['lon']:.4f}° E` (Buffer: `±{active_site_meta['buffer_deg']}°`)\n"
            f"- **Active Data Feed:** `{report['report_metadata']['data_source_mode']}`\n"
            f"- **Thermal SST Feed:** `{metrics.get('sst_data_source', 'n/a')}` "
            f"(Baseline `0.0σ`: `{metrics.get('sst_baseline_window', 'n/a')}` = `{metrics.get('baseline_median_sst', 0.0):.2f} °C`, `σ = {metrics.get('sigma_robust_sst', 0.0):.2f} °C`)\n"
            f"- **Analysis Window:** `{report['report_metadata']['analysis_period']['start_date']}` to "
            f"`{report['report_metadata']['analysis_period']['end_date']}` "
            f"(`{metrics['total_revisit_observations']}` passes ≥`{MIN_VALID_WATER_PCT:.0f}%` water, "
            f"mean coverage `{metrics['mean_valid_water_coverage_pct']:.1f}%`)\n"
            f"- **Site Operating Context:** *{profile['historical_note']}*"
        )
    with header_right:
        st.plotly_chart(
            _build_intake_map(
                lat=profile["coordinates"]["lat"],
                lon=profile["coordinates"]["lon"],
                buffer_deg=active_site_meta["buffer_deg"],
                site_name=profile["site_name"],
                dark_mode=dark_mode,
            ),
            use_container_width=True,
        )

    st.divider()

    # --- Conditional Characterization Banner & Epistemological KPI Strip ---
    is_present = bool(metrics.get("is_present_day_window", False))
    regime_profile = str(metrics.get("window_regime_profile", "SEASONAL / MULTI-PHASE REGIME"))
    n_anom = int(metrics.get("anomalous_passes_count", 0))
    n_tot = int(metrics.get("total_valid_passes", metrics["total_revisit_observations"]))
    exc_pct = float(metrics.get("exceedance_pct", 0.0))
    max_streak = int(metrics.get("max_consecutive_streak", 0))
    peak_date_str = str(metrics.get("peak_event_date", "n/a"))
    days_since_last = int(metrics.get("days_since_last_pass", 0))
    latest_pass_date = str(metrics.get("latest_valid_pass_date", "n/a"))
    cur_op_risk = str(metrics.get("current_operational_risk", "LOW"))

    short_regime = (
        "Stable"
        if "STABLE" in regime_profile
        else (
            "Episodic"
            if "EPISODIC" in regime_profile
            else ("Chronic" if "CHRONIC" in regime_profile else "Seasonal")
        )
    )

    if is_present:
        if days_since_last > 18 or cur_op_risk == "CLOUD_OBSCURED":
            op_banner_text = "⚪ RECENT OPTICAL DATA OBSCURED BY CLOUDS (Thermal Watch Only)"
            op_kpi_value = "⚪ OBSCURED"
            st.warning(
                f"**CURRENT OPERATIONAL RISK:** {op_banner_text}  \n"
                f"Latest clear Sentinel-2 pass: **{days_since_last} days ago** (`{latest_pass_date}`) | "
                f"Window Regime: **{short_regime}** (`{regime_profile}`)",
                icon="☁️",
            )
        elif cur_op_risk == "HIGH":
            op_kpi_value = "🔴 HIGH"
            st.error(
                f"**CURRENT OPERATIONAL RISK (21-Day Tail): 🔴 HIGH**  \n"
                f"Latest clear Sentinel-2 pass: **{days_since_last} days ago** (`{latest_pass_date}`) | "
                f"Window Regime: **{short_regime}** (`{n_anom}/{n_tot}` passes, `{exc_pct:.1f}%`)",
                icon="🚨",
            )
        elif cur_op_risk == "MODERATE":
            op_kpi_value = "🟡 MODERATE"
            st.warning(
                f"**CURRENT OPERATIONAL RISK (21-Day Tail): 🟡 MODERATE**  \n"
                f"Latest clear Sentinel-2 pass: **{days_since_last} days ago** (`{latest_pass_date}`) | "
                f"Window Regime: **{short_regime}** (`{n_anom}/{n_tot}` passes, `{exc_pct:.1f}%`)",
                icon="⚠️",
            )
        else:
            op_kpi_value = "🟢 LOW"
            st.success(
                f"**CURRENT OPERATIONAL RISK (21-Day Tail): 🟢 LOW**  \n"
                f"Latest clear Sentinel-2 pass: **{days_since_last} days ago** (`{latest_pass_date}`) | "
                f"Window Regime: **{short_regime}** (`{n_anom}/{n_tot}` passes, `{exc_pct:.1f}%`)",
                icon="✅",
            )
    else:
        st.info(
            f"📊 **HISTORICAL SITE CHARACTERIZATION (IAEA SSR-1): `{regime_profile}`**  \n"
            f"Anomalous Passes: **{n_anom} of {n_tot} clear passes ({exc_pct:.1f}%)** | "
            f"Max Persistence: **{max_streak} consecutive passes** | "
            f"Peak Anomaly Date: **`{peak_date_str}`**",
        )

    kpi_1, kpi_2, kpi_3, kpi_4, kpi_5 = st.columns(5)

    if is_present:
        if days_since_last > 18 or cur_op_risk == "CLOUD_OBSCURED":
            op_kpi_value = "⚪ OBSCURED"
        elif cur_op_risk == "HIGH":
            op_kpi_value = "🔴 ACTIVE"
        elif cur_op_risk == "MODERATE":
            op_kpi_value = "🟡 ACTIVE"
        else:
            op_kpi_value = "🟢 BASELINE"

        with kpi_1:
            st.metric(
                label="Window Regime (Lookback)",
                value=short_regime,
                delta=f"{n_anom}/{n_tot} ({exc_pct:.0f}%) | Streak {max_streak}",
                delta_color="off",
                help=f"Full-window regime: {regime_profile}. Peak event on {peak_date_str}.",
            )
        with kpi_2:
            st.metric(
                label="Recent Operational Status (21d)",
                value=op_kpi_value,
                delta=f"Latest pass: {days_since_last}d ago ({latest_pass_date})",
                delta_color="inverse" if cur_op_risk in ("HIGH", "MODERATE") else "normal",
                help=(
                    "Evaluated strictly from the Recent Operational Tail (last 21 days). "
                    "Displays ⚪ OBSCURED if >18 days without a clear Sentinel-2 pass."
                ),
            )
    else:
        with kpi_1:
            st.metric(
                label="Site Regime Profile (SSR-1)",
                value=short_regime,
                delta=f"Peak date: {peak_date_str}",
                delta_color="off",
                help=f"Historical ecological temperament: {regime_profile}.",
            )
        with kpi_2:
            st.metric(
                label="Exceedance Frequency",
                value=f"{n_anom}/{n_tot} passes",
                delta=f"{exc_pct:.1f}% | Max streak {max_streak}",
                delta_color="off",
                help="Share of cloud-free passes in this historical window exceeding Dual-Gate thresholds.",
            )
    with kpi_3:
        st.metric(
            label="Peak NDCI (Trophic Precursor)",
            value=f"{metrics['max_ndci']:+.3f} ({metrics['max_ndci_zscore']:+.2f}σ)",
            delta=f"Latest: {metrics['latest_ndci']:+.3f} ({metrics['latest_ndci_zscore']:+.2f}σ) | High ≥ {NDCI_THRESHOLDS['HIGH']:+.2f}",
            delta_color="inverse"
            if (
                metrics["max_ndci"] >= NDCI_THRESHOLDS["HIGH"]
                or metrics["max_ndci_zscore"] >= ZSCORE_THRESHOLDS["HIGH"]
            )
            else "normal",
        )
    with kpi_4:
        st.metric(
            label="Peak NDTI (Turbidity Precursor)",
            value=f"{metrics['max_ndti']:+.3f} ({metrics['max_ndti_zscore']:+.2f}σ)",
            delta=f"Latest: {metrics['latest_ndti']:+.3f} ({metrics['latest_ndti_zscore']:+.2f}σ) | High ≥ {NDTI_THRESHOLDS['HIGH']:+.2f}",
            delta_color="inverse"
            if (
                metrics["max_ndti"] >= NDTI_THRESHOLDS["HIGH"]
                or metrics["max_ndti_zscore"] >= ZSCORE_THRESHOLDS["HIGH"]
            )
            else "normal",
        )
    with kpi_5:
        has_sst_kpi = pd.notna(metrics.get("max_sst"))
        st.metric(
            label="Peak SST (Thermal Trigger)",
            value=(
                f"{metrics['max_sst']:.1f} °C ({metrics['max_sst_zscore']:+.2f}σ)"
                if has_sst_kpi
                else "n/a"
            ),
            delta=(
                f"Latest: {metrics['latest_sst']:.1f} °C ({metrics['latest_sst_zscore']:+.2f}σ) | "
                f"High ≥ {SST_THRESHOLDS['HIGH']:.1f} °C"
                if has_sst_kpi
                else "No thermal data"
            ),
            delta_color="inverse"
            if (
                has_sst_kpi
                and (
                    metrics.get("thermal_high_pass_detections", 0) >= 1
                    or metrics.get("trophic_lag_jellyfish_precursor_detected")
                )
            )
            else "normal",
            help=(
                f"SST Z-Score is evaluated against {metrics.get('sst_baseline_window', 'past years')} "
                f"(median {metrics.get('baseline_median_sst', 0.0):.2f} °C, σ = {metrics.get('sigma_robust_sst', 0.0):.2f} °C). "
                f"Peak SST on {metrics.get('max_sst_date')}; P90 = {metrics.get('p90_sst')} °C. "
                f"Trophic-lag jellyfish precursor: "
                f"{'DETECTED' if metrics.get('trophic_lag_jellyfish_precursor_detected') else 'not detected'}. "
                f"Source: {metrics.get('sst_data_source', 'n/a')}."
            ),
        )

    st.caption(
        f"**Primary Hazard Driver Identified:** {risk_info['primary_hazard_driver']} — "
        "*Note: Optical NDCI/NDTI metrics act as trophic and physical water-mass precursors "
        "(phytoplankton blooms, organic detritus, suspended sediment plumes); transparent jellyfish are "
        "not imaged directly. SST is the physical thermal trigger for medusa aggregation and stratification. "
        "Together they screen gelatinous zooplankton ingress and traveling screen ΔP loading under IAEA SSR-1 & SSG-18.*"
    )

    # --- Chart View Mode Selector & Interactive Plotly Charts ---
    chart_view = st.radio(
        "Time-Series Visualization Mode",
        options=[
            "📈 Coupled Spectral + Thermal Indices (NDCI, NDTI & SST)",
            "📊 Standardized Robust Z-Score Time Series (+1.50σ Moderate / +2.20σ High)",
            "🔀 Show Both Charts (Spectral/Thermal + Robust Z-Score)",
        ],
        index=0,
        horizontal=True,
    )

    if "Spectral" in chart_view or "Both" in chart_view:
        st.plotly_chart(
            _build_dual_hazard_chart(
                df=df,
                site_name=profile["site_name"],
                dark_mode=dark_mode,
            ),
            use_container_width=True,
        )

    if "Robust Z-Score" in chart_view or "Both" in chart_view:
        st.plotly_chart(
            _build_zscore_hazard_chart(
                df=df,
                site_name=profile["site_name"],
                dark_mode=dark_mode,
            ),
            use_container_width=True,
        )

    st.divider()

    # --- Regulatory & NHSI Hub Output Section (2 Tabs) ---
    st.subheader("🏛️ IAEA Regulatory Context & NHSI SMR Hub Export")
    tab_irrs, tab_export = st.tabs(
        [
            "📋 Tab 1: Preliminary IRRS & Safety Standards Checklist",
            "📦 Tab 2: Standardized NHSI SMR Site Data (JSON/CSV Export)",
        ]
    )

    with tab_irrs:
        active_gemini_key, _ = resolve_gemini_api_key(explicit_gemini_key)
        rag_bundle = _synthesize_rag_cards_cached(
            site_name=profile["site_name"],
            start_str=report["report_metadata"]["analysis_period"]["start_date"],
            end_str=report["report_metadata"]["analysis_period"]["end_date"],
            risk_level=risk_info["overall_scram_clogging_risk"],
            max_ndci=float(metrics["max_ndci"]),
            max_ndti=float(metrics["max_ndti"]),
            max_sst=float(metrics.get("max_sst", 0.0) or 0.0),
            has_gemini_key=bool(active_gemini_key),
            _report_dict=report,
            _site_meta=active_site_meta,
            _explicit_gemini_key=explicit_gemini_key,
        )

        st.caption(f"🔎 **{rag_bundle['mode_badge']}**")
        st.info(
            f"🧠 **Contextual Regulatory Advisory:** {rag_bundle['ai_contextual_advisory']}",
            icon="🏛️",
        )

        for item in rag_bundle["cards"]:
            status_tag = item["status"]
            icon = (
                "🚨"
                if "ACTION" in status_tag
                else ("⚠️" if ("REVIEW" in status_tag or "WATCH" in status_tag) else "✅")
            )
            with st.expander(f"{icon} [{status_tag}] {item['standard']}", expanded=True):
                if item.get("cited_paragraphs"):
                    st.caption(f"**Cited IAEA Clauses:** `{item['cited_paragraphs']}`")
                st.markdown(f"**Satellite Empirical Finding:** {item['finding']}")
                st.markdown(f"**IRRS Reviewer Verification Prompt:** *{item['irrs_reviewer_prompt']}*")

        if risk_info["high_risk_triggered_dates"]:
            st.markdown(
                "**Dates Triggering Dual-Gate HIGH Risk:** "
                + ", ".join(f"`{d}`" for d in risk_info["high_risk_triggered_dates"])
            )
        if risk_info.get("moderate_risk_triggered_dates"):
            st.markdown(
                "**Dates Triggering Dual-Gate MODERATE Risk:** "
                + ", ".join(f"`{d}`" for d in risk_info["moderate_risk_triggered_dates"])
            )

        st.divider()
        st.markdown("#### 💬 Ask the IAEA RAG Regulatory Advisor about this Site Screening")
        preset_questions = [
            "Why did the coupled optical-thermal metrics trigger this risk level?",
            "What specific intake screen & backwash checks should an IRRS team request at this site?",
            "How does IAEA SSR-1 Req. 20 apply to SMR deployment here?",
        ]
        q_cols = st.columns(3)
        for idx, pq in enumerate(preset_questions):
            with q_cols[idx]:
                if st.button(pq, key=f"rag_preset_{idx}", use_container_width=True):
                    st.session_state["irrs_qa_question"] = pq

        user_q = st.text_input(
            "Or enter a custom IRRS / NHSI regulatory question:",
            value=st.session_state.get("irrs_qa_question", ""),
            placeholder="e.g., How should traveling screen ΔP setpoints be verified for this water body?",
            key="irrs_qa_input",
        )
        active_q = (user_q or st.session_state.get("irrs_qa_question", "")).strip()
        if active_q:
            qa_resp = _answer_inspector_qa_cached(
                question=active_q,
                site_name=profile["site_name"],
                start_str=report["report_metadata"]["analysis_period"]["start_date"],
                end_str=report["report_metadata"]["analysis_period"]["end_date"],
                risk_level=risk_info["overall_scram_clogging_risk"],
                has_gemini_key=bool(active_gemini_key),
                _report_dict=report,
                _site_meta=active_site_meta,
                _explicit_gemini_key=explicit_gemini_key,
            )
            st.caption(
                f"Source: `{qa_resp['mode']}` | Retrieved Clauses: "
                + ", ".join(f"`{c}`" for c in qa_resp["citations"])
            )
            st.markdown(qa_resp["answer"])

    with tab_export:
        st.markdown(
            "Standardized machine-readable baseline package ready for ingestion into the "
            "**IAEA NHSI SMR Regulation Hub** and pre-mission **IRRS Site Evaluation** dossiers."
        )

        json_payload = json.dumps(report, indent=2)
        export_df = df.copy()
        export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
        csv_payload = export_df.to_csv(index=False)

        safe_slug = (
            profile["site_name"]
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ Download NHSI Site Baseline (JSON)",
                data=json_payload,
                file_name=f"scram_predictor_{safe_slug}_baseline.json",
                mime="application/json",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="⬇️ Download Filtered Satellite Time-Series (CSV)",
                data=csv_payload,
                file_name=f"scram_predictor_{safe_slug}_timeseries.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("#### Structured NHSI & IRRS Baseline Payload (`JSON`)")
        st.json(report)


if __name__ == "__main__":
    main()
