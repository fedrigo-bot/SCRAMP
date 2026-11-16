"""
Regulatory & Advisory Hook Module (`advisor_hook.py`).

Bridges satellite-derived water quality hazard indices (NDCI, NDTI) and
outlier-immune non-parametric Robust Z-Score anomalies with IAEA Safety Standards
(SSR-1, SSG-18) for Integrated Regulatory Review Service (IRRS) missions and the
Nuclear Harmonization and Standardization Initiative (NHSI) SMR Hub.

Designed with a clean extension hook so Phase 2 LLM/RAG Knowledge Management
modules can be plugged in without refactoring the data pipeline or Streamlit UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    CLIMATOLOGICAL_BASELINES,
    IAEA_REFERENCES,
    INDEX_CLAMP_RANGE,
    MIN_VALID_WATER_PCT,
    NDCI_THRESHOLDS,
    NDTI_THRESHOLDS,
    SIGMA_MIN,
    SST_CLAMP_RANGE,
    SST_COUPLED_GATES,
    SST_THRESHOLDS,
    TOOL_NAME,
    TOOL_VERSION,
    ZSCORE_THRESHOLDS,
    compute_bounding_box,
    resolve_site_key,
)
from satellite_engine import (
    compute_historical_window_sst_baseline,
    compute_robust_zscore_series,
)

TROPHIC_THERMAL_DRIVER = (
    "Coupled Trophic-Thermal Swarm Regime (Spring Phytoplankton Precursor + Mid-Summer SST Spike "
    "-> High Gelatinous Zooplankton / Jellyfish Ingress Risk under IAEA SSG-18 Section 5)"
)


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def evaluate_thermal_gate(
    sst_celsius: Optional[float],
    sst_zscore: Optional[float],
    prior_max_ndci_zscore: Optional[float],
) -> str:
    """
    Coupled "Trophic-Lag + Thermal Trigger" jellyfish gate for a single pass.

    HIGH:
      - sst_z >= 1.65 AND sst >= 19.0 °C AND an earlier bloom (NDCI Z >= 1.50 at least
        20 days before this pass) occurred in the window   [trophic-lag precursor]
      - OR sst_z >= 2.20 AND sst >= 20.0 °C                [severe thermal anomaly]
    MODERATE:
      - sst_z >= 1.50 AND sst >= 18.5 °C
    """
    if not (_is_finite(sst_celsius) and _is_finite(sst_zscore)):
        return "LOW"
    g = SST_COUPLED_GATES
    t, z = float(sst_celsius), float(sst_zscore)
    prior_bloom = _is_finite(prior_max_ndci_zscore) and float(prior_max_ndci_zscore) >= g["PRIOR_BLOOM_NDCI_Z"]
    if (z >= g["TROPHIC_LAG_Z"] and t >= g["TROPHIC_LAG_C"] and prior_bloom) or (
        z >= g["SEVERE_Z"] and t >= g["SEVERE_C"]
    ):
        return "HIGH"
    if z >= g["MODERATE_Z"] and t >= g["MODERATE_C"]:
        return "MODERATE"
    return "LOW"


def evaluate_pass_dual_gate(
    ndci: float,
    ndti: float,
    ndci_zscore: float,
    ndti_zscore: float,
    sst_celsius: Optional[float] = None,
    sst_zscore: Optional[float] = None,
    prior_max_ndci_zscore: Optional[float] = None,
) -> str:
    """
    Evaluate a single satellite pass using the Dual-Gate Coastal Hybrid Decision Matrix
    (combining Site-Relative Robust Z-Score Gate and Marine-Calibrated Absolute Gate),
    extended with the coupled Trophic-Lag + Thermal Trigger gate (`evaluate_thermal_gate`).

    HIGH Risk:
      - ndci_zscore >= 2.20 OR ndti_zscore >= 2.20
      - OR (ndci_zscore >= 1.50 AND ndti_zscore >= 1.50)
      - OR ndci >= 0.03 OR ndti >= -0.10
      - OR thermal gate == HIGH
    MODERATE Risk (if not HIGH):
      - ndci_zscore >= 1.50 OR ndti_zscore >= 1.50
      - OR ndci >= -0.02 OR ndti >= -0.18
      - OR thermal gate == MODERATE
    LOW Risk:
      - Otherwise
    """
    thermal = evaluate_thermal_gate(sst_celsius, sst_zscore, prior_max_ndci_zscore)
    is_high = (
        thermal == "HIGH"
        or ndci_zscore >= ZSCORE_THRESHOLDS["HIGH"]
        or ndti_zscore >= ZSCORE_THRESHOLDS["HIGH"]
        or (
            ndci_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
            and ndti_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
        )
        or ndci >= NDCI_THRESHOLDS["HIGH"]
        or ndti >= NDTI_THRESHOLDS["HIGH"]
    )
    if is_high:
        return "HIGH"

    is_mod = (
        thermal == "MODERATE"
        or ndci_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
        or ndti_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
        or ndci >= NDCI_THRESHOLDS["MODERATE"]
        or ndti >= NDTI_THRESHOLDS["MODERATE"]
    )
    if is_mod:
        return "MODERATE"

    return "LOW"


def _classify_overall_risk(
    max_ndci: float,
    max_ndti: float,
    max_ndci_zscore: float,
    max_ndti_zscore: float,
    high_pass_count: int,
    moderate_pass_count: int,
    thermal_high_count: int = 0,
    thermal_moderate_count: int = 0,
    optical_high_count: Optional[int] = None,
    optical_moderate_count: Optional[int] = None,
    trophic_lag_detected: bool = False,
    max_sst: Optional[float] = None,
    max_sst_zscore: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Determine the overall cooling-water intake clogging / SCRAM risk level
    using the Dual-Gate Coastal Hybrid Decision Matrix (optical + thermal) and identify
    the primary trophic, thermal or hydrological hazard driver.
    """
    overall_gate = evaluate_pass_dual_gate(
        ndci=max_ndci,
        ndti=max_ndti,
        ndci_zscore=max_ndci_zscore,
        ndti_zscore=max_ndti_zscore,
    )
    if optical_high_count is None:
        optical_high_count = high_pass_count
    if optical_moderate_count is None:
        optical_moderate_count = moderate_pass_count

    if high_pass_count >= 1 or overall_gate == "HIGH":
        level = "HIGH"
    elif moderate_pass_count >= 1 or overall_gate == "MODERATE":
        level = "MODERATE"
    else:
        level = "LOW"

    sst_txt = (
        f"SST {max_sst:.1f} °C ({max_sst_zscore:+.2f}σ)"
        if _is_finite(max_sst) and _is_finite(max_sst_zscore)
        else "SST n/a"
    )

    # 1) Coupled trophic-thermal regime takes precedence as primary driver
    if trophic_lag_detected:
        return "HIGH", TROPHIC_THERMAL_DRIVER

    # 2) Thermal-only escalation (optical channels did not reach this level)
    if level == "HIGH" and thermal_high_count >= 1 and optical_high_count == 0 and overall_gate != "HIGH":
        return (
            "HIGH",
            f"Severe Thermal Anomaly / Stratification Trigger (jellyfish aggregation window): {sst_txt}",
        )
    if level == "MODERATE" and thermal_moderate_count >= 1 and optical_moderate_count == 0 and overall_gate == "LOW":
        return (
            "MODERATE",
            f"Elevated Thermal Watch (SST above seasonal envelope): {sst_txt}",
        )

    if level == "HIGH":
        compound_z = (
            max_ndci_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
            and max_ndti_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
        )
        high_bio = (
            max_ndci_zscore >= ZSCORE_THRESHOLDS["HIGH"]
            or max_ndci >= NDCI_THRESHOLDS["HIGH"]
        )
        high_sed = (
            max_ndti_zscore >= ZSCORE_THRESHOLDS["HIGH"]
            or max_ndti >= NDTI_THRESHOLDS["HIGH"]
        )

        if compound_z or (high_bio and high_sed):
            return (
                "HIGH",
                (
                    f"Compound Trophic & Physical Water-Mass Anomaly: "
                    f"NDCI {max_ndci:+.3f} ({max_ndci_zscore:+.2f}σ) & "
                    f"NDTI {max_ndti:+.3f} ({max_ndti_zscore:+.2f}σ)"
                ),
            )
        if high_bio:
            return (
                "HIGH",
                (
                    f"Acute Trophic/Chlorophyll Precursor Surge: "
                    f"NDCI {max_ndci:+.3f} ({max_ndci_zscore:+.2f}σ above site median)"
                ),
            )
        return (
            "HIGH",
            (
                f"Acute Suspended Sediment & Detritus Plume: "
                f"NDTI {max_ndti:+.3f} ({max_ndti_zscore:+.2f}σ above site median)"
            ),
        )

    if level == "MODERATE":
        mod_bio = (
            max_ndci_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
            or max_ndci >= NDCI_THRESHOLDS["MODERATE"]
        )
        mod_sed = (
            max_ndti_zscore >= ZSCORE_THRESHOLDS["MODERATE"]
            or max_ndti >= NDTI_THRESHOLDS["MODERATE"]
        )
        if mod_bio and mod_sed:
            return (
                "MODERATE",
                (
                    f"Combined Moderate Trophic ({max_ndci_zscore:+.2f}σ) & "
                    f"Turbidity ({max_ndti_zscore:+.2f}σ) Water-Mass Loading"
                ),
            )
        if mod_bio:
            return (
                "MODERATE",
                (
                    f"Elevated Trophic/Phytoplankton Precursor Activity: "
                    f"NDCI {max_ndci:+.3f} ({max_ndci_zscore:+.2f}σ)"
                ),
            )
        return (
            "MODERATE",
            (
                f"Elevated Suspended Sediment / Water Turbidity: "
                f"NDTI {max_ndti:+.3f} ({max_ndti_zscore:+.2f}σ)"
            ),
        )

    return (
        "LOW",
        "Baseline Coastal Water Conditions Within Normal Operating Envelope",
    )


def _build_irrs_checklist(
    risk_level: str,
    hazard_metrics: Dict[str, Any],
    site_meta: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Generate deterministic, rule-based preliminary regulatory review items
    mapped to IAEA SSR-1 (Requirement 20) and IAEA SSG-18 (Section 5).
    Explicitly frames optical NDCI/NDTI metrics as trophic and physical water-mass
    precursors rather than direct optical photography of transparent subsurface jellyfish.
    """
    bio_high = (
        hazard_metrics["max_ndci"] >= NDCI_THRESHOLDS["HIGH"]
        or hazard_metrics["max_ndci_zscore"] >= ZSCORE_THRESHOLDS["HIGH"]
    )
    bio_mod = (
        hazard_metrics["max_ndci"] >= NDCI_THRESHOLDS["MODERATE"]
        or hazard_metrics["max_ndci_zscore"] >= ZSCORE_THRESHOLDS["MODERATE"]
    )
    sed_high = (
        hazard_metrics["max_ndti"] >= NDTI_THRESHOLDS["HIGH"]
        or hazard_metrics["max_ndti_zscore"] >= ZSCORE_THRESHOLDS["HIGH"]
    )
    sed_mod = (
        hazard_metrics["max_ndti"] >= NDTI_THRESHOLDS["MODERATE"]
        or hazard_metrics["max_ndti_zscore"] >= ZSCORE_THRESHOLDS["MODERATE"]
    )

    checklist: List[Dict[str, str]] = [
        {
            "standard": f"{IAEA_REFERENCES['SSR-1']['code']} ({IAEA_REFERENCES['SSR-1']['requirement']})",
            "status": (
                "ACTION REQUIRED"
                if (risk_level == "HIGH" or bio_high)
                else ("REVIEW RECOMMENDED" if (risk_level == "MODERATE" or bio_mod) else "COMPLIANT BASELINE")
            ),
            "finding": (
                f"Peak NDCI reached {hazard_metrics['max_ndci']:+.3f} (site median: "
                f"{hazard_metrics['baseline_median_ndci']:+.3f}, peak relative anomaly: "
                f"{hazard_metrics['max_ndci_zscore']:+.2f}σ, robust scale σ={hazard_metrics['sigma_robust_ndci']:.4f}). "
                "Optical chlorophyll-a surges serve as trophic water-mass precursors (phytoplankton blooms "
                "and warm nutrient-enriched coastal fronts that support gelatinous zooplankton/jellyfish "
                "swarms and macro-algal detachment), rather than direct optical imaging of transparent subsurface organisms."
            ),
            "irrs_reviewer_prompt": (
                "Verify under IAEA SSR-1 Requirement 20 that the site safety analysis correlates "
                "satellite trophic precursors (NDCI > -0.02 or Z >= +1.5σ) with in-situ intake monitoring, "
                "seasonal jellyfish/zooplankton ingress models, and Ultimate Heat Sink (UHS) margin assessments."
            ),
        },
        {
            "standard": f"{IAEA_REFERENCES['SSG-18']['code']} ({IAEA_REFERENCES['SSG-18']['requirement']})",
            "status": (
                "ACTION REQUIRED"
                if sed_high
                else ("REVIEW RECOMMENDED" if sed_mod else "COMPLIANT BASELINE")
            ),
            "finding": (
                f"Peak NDTI reached {hazard_metrics['max_ndti']:+.3f} (site median: "
                f"{hazard_metrics['baseline_median_ndti']:+.3f}, peak relative anomaly: "
                f"{hazard_metrics['max_ndti_zscore']:+.2f}σ, robust scale σ={hazard_metrics['sigma_robust_ndti']:.4f}), "
                "reflecting physical water-mass turbidity, organic detritus, and suspended sediment plumes "
                "capable of accelerating fine-screen biofouling and particulate loading."
            ),
            "irrs_reviewer_prompt": (
                "Confirm under IAEA SSG-18 Section 5 the adequacy of coarse trash racks, traveling band/drum "
                "screen differential-pressure (ΔP) alarm and trip setpoints, high-pressure backwash capacity, "
                "and debris-return prevention during compound trophic and turbidity anomalies."
            ),
        },
        {
            "standard": f"{IAEA_REFERENCES['SSG-18']['code']} (Section 5 — Thermal Trigger & Jellyfish Ingress)",
            "status": (
                "ACTION REQUIRED"
                if (
                    hazard_metrics.get("trophic_lag_jellyfish_precursor_detected")
                    or hazard_metrics.get("thermal_high_pass_detections", 0) >= 1
                )
                else (
                    "REVIEW RECOMMENDED"
                    if hazard_metrics.get("thermal_moderate_pass_detections", 0) >= 1
                    else "COMPLIANT BASELINE"
                )
            ),
            "finding": (
                f"Peak SST {hazard_metrics.get('max_sst', float('nan')):.1f} °C "
                f"(P90 {hazard_metrics.get('p90_sst', float('nan')):.1f} °C, latest "
                f"{hazard_metrics.get('latest_sst', float('nan')):.1f} °C, peak anomaly "
                f"{hazard_metrics.get('max_sst_zscore', float('nan')):+.2f}σ vs site median "
                f"{hazard_metrics.get('baseline_median_sst', float('nan')):.1f} °C). "
                f"Trophic-lag jellyfish precursor detected: "
                f"{'YES' if hazard_metrics.get('trophic_lag_jellyfish_precursor_detected') else 'NO'}"
                + (
                    f" (bloom {hazard_metrics.get('trophic_lag_bloom_date')} -> thermal spike "
                    f"{hazard_metrics.get('trophic_lag_thermal_date')})."
                    if hazard_metrics.get("trophic_lag_jellyfish_precursor_detected")
                    else "."
                )
                + " SST is the physical trigger for medusa aggregation and stratification; an earlier "
                "phytoplankton bloom supplies the food web that sustains gelatinous zooplankton. "
                f"Thermal source: {hazard_metrics.get('sst_data_source', 'n/a')}."
            ),
            "irrs_reviewer_prompt": (
                "Verify that the operator correlates SST anomalies (and prior bloom history) with "
                "drum/band-screen ΔP trends, jellyfish nets / bubble curtains readiness, and power-reduction "
                "procedures for intake blockage, consistent with IAEA SSR-1 Req. 20 and SSG-18 Section 5."
            ),
        },
        {
            "standard": f"{IAEA_REFERENCES['NHSI-SMR']['code']} ({IAEA_REFERENCES['NHSI-SMR']['requirement']})",
            "status": "DATA PACKAGE READY",
            "finding": (
                f"Evaluated {hazard_metrics['total_revisit_observations']} cloud-screened and dark-pixel-guarded "
                f"Sentinel-2 L2A passes (mean water coverage: {hazard_metrics['mean_valid_water_coverage_pct']:.1f}%, "
                f"gate >= {MIN_VALID_WATER_PCT:.0f}%) with {hazard_metrics['high_risk_revisit_detections']} HIGH-risk "
                f"and {hazard_metrics['moderate_risk_revisit_detections']} MODERATE-risk Dual-Gate detections."
            ),
            "irrs_reviewer_prompt": (
                "Export standardized JSON/CSV site hazard envelope to the NHSI SMR regulatory "
                "pre-application dossier to benchmark passive/active cooling intake resilience."
            ),
        },
    ]
    return checklist


# TODO (Phase 2): Replace static regulatory notes with LLM/RAG retrieval over IAEA Safety Standards and historical IRRS mission reports
def run_rag_regulatory_analysis(report_dict: Dict[str, Any]) -> str:
    """
    Phase 2 Extension Hook for AI Knowledge Management & RAG Integration.

    In Phase 2, this function will query a vector database indexing:
      - IAEA Safety Requirements SSR-1 & Specific Safety Guide SSG-18 full clauses
      - Historical IAEA IRRS Mission Follow-up Reports & IRS (Incident Reporting System) lessons learned
      - NHSI SMR Design-Specific Ultimate Heat Sink (UHS) Envelope Specifications
    """
    site_name = report_dict.get("site_profile", {}).get("site_name", "Unknown Site")
    risk_level = report_dict.get("risk_classification", {}).get(
        "overall_scram_clogging_risk", "UNKNOWN"
    )
    primary_driver = report_dict.get("risk_classification", {}).get(
        "primary_hazard_driver", "N/A"
    )
    metrics = report_dict.get("hazard_metrics", {})
    max_ndci = metrics.get("max_ndci", 0.0)
    max_ndti = metrics.get("max_ndti", 0.0)
    max_z = metrics.get("max_relative_anomaly_zscore", 0.0)
    max_sst = metrics.get("max_sst", float("nan"))
    max_sst_z = metrics.get("max_sst_zscore", float("nan"))

    return (
        f"[Phase 2 RAG Hook Ready — Deterministic Preview for {site_name}] "
        f"Dual-Gate Intake Clogging Risk is classified as {risk_level} ({primary_driver}). "
        f"Peak coastal precursors (NDCI={max_ndci:+.3f}, NDTI={max_ndti:+.3f}, Peak Z={max_z:+.2f}σ, "
        f"SST={max_sst:.1f} °C / {max_sst_z:+.2f}σ) "
        f"are structured as trophic and physical water-mass indicators for semantic retrieval against "
        f"IAEA SSR-1 Requirement 20, SSG-18 Section 5 (Traveling Screen ΔP Monitoring), and IRS operating experience."
    )


def generate_site_baseline_report(
    site_name: str,
    site_meta: Dict[str, Any],
    df: pd.DataFrame,
    date_range: Tuple[Any, Any],
) -> Dict[str, Any]:
    """
    Compute outlier-immune non-parametric statistical hazard metrics (MAD + IQR Robust Z-Score)
    and Dual-Gate risk classifications from the Sentinel-2 water quality time series, returning
    a JSON-serializable baseline report dictionary for IRRS & NHSI SMR workflows.
    """
    if df is None or df.empty:
        raise ValueError("Cannot generate baseline report from an empty satellite DataFrame.")

    clean_df = df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    if "valid_water_coverage_pct" in clean_df.columns:
        clean_df = clean_df[
            pd.to_numeric(clean_df["valid_water_coverage_pct"], errors="coerce")
            >= MIN_VALID_WATER_PCT
        ].copy()

    if clean_df.empty:
        raise ValueError(
            f"No satellite passes met the minimum valid water coverage gate (>={MIN_VALID_WATER_PCT}%)."
        )

    clean_df = clean_df.sort_values("date").reset_index(drop=True)

    # Clamp to physical coastal range [-0.60, +0.60] to eliminate Sen2Cor dark-water spikes
    ndci_series = (
        pd.to_numeric(clean_df["ndci"], errors="coerce")
        .clip(lower=INDEX_CLAMP_RANGE[0], upper=INDEX_CLAMP_RANGE[1])
        .astype(float)
    )
    ndti_series = (
        pd.to_numeric(clean_df["ndti"], errors="coerce")
        .clip(lower=INDEX_CLAMP_RANGE[0], upper=INDEX_CLAMP_RANGE[1])
        .astype(float)
    )
    clean_df["ndci"] = ndci_series.round(4)
    clean_df["ndti"] = ndti_series.round(4)

    # Resolve site climatological reference baseline for short-window anchoring (<12 passes)
    clim_ref = CLIMATOLOGICAL_BASELINES.get(
        resolve_site_key(site_name), CLIMATOLOGICAL_BASELINES["default"]
    )

    # Compute outlier-immune non-parametric MAD + IQR Robust Z-scores
    ndci_z_series, baseline_median_ndci, sigma_robust_ndci = compute_robust_zscore_series(
        ndci_series,
        sigma_min=SIGMA_MIN["NDCI"],
        clim_median=clim_ref["ndci_median"],
        clim_sigma=clim_ref["ndci_sigma"],
    )
    ndti_z_series, baseline_median_ndti, sigma_robust_ndti = compute_robust_zscore_series(
        ndti_series,
        sigma_min=SIGMA_MIN["NDTI"],
        clim_median=clim_ref["ndti_median"],
        clim_sigma=clim_ref["ndti_sigma"],
    )
    clean_df["ndci_zscore"] = ndci_z_series
    clean_df["ndti_zscore"] = ndti_z_series
    df["ndci_zscore"] = ndci_z_series.values
    df["ndti_zscore"] = ndti_z_series.values

    # --- Thermal channel (SST °C) with the same MAD + IQR robust Z-score ---
    if "sst_celsius" not in clean_df.columns:
        clean_df["sst_celsius"] = np.nan
    sst_series = (
        pd.to_numeric(clean_df["sst_celsius"], errors="coerce")
        .clip(lower=SST_CLAMP_RANGE[0], upper=SST_CLAMP_RANGE[1])
        .astype(float)
    )
    has_sst = bool(sst_series.notna().any())
    if has_sst:
        sst_series = sst_series.interpolate(limit_direction="both")
        if (
            "sst_baseline_median" in df.attrs
            and "sst_baseline_sigma" in df.attrs
            and "sst_baseline_window" in df.attrs
        ):
            hist_sst_med = float(df.attrs["sst_baseline_median"])
            hist_sst_sig = float(df.attrs["sst_baseline_sigma"])
            sst_baseline_window = str(df.attrs["sst_baseline_window"])
        else:
            hist_sst_med, hist_sst_sig, sst_baseline_window = (
                compute_historical_window_sst_baseline(
                    start_date=date_range[0],
                    end_date=date_range[1],
                    site_name=site_name,
                    lat=float(site_meta.get("lat", 50.0)),
                    lon=float(site_meta.get("lon", 2.0)),
                    n_past_years=5,
                )
            )
        sst_z_series, baseline_median_sst, sigma_robust_sst = compute_robust_zscore_series(
            sst_series,
            sigma_min=SIGMA_MIN["SST"],
            clamp_range=SST_CLAMP_RANGE,
            clim_median=hist_sst_med,
            clim_sigma=hist_sst_sig,
            strict_external_baseline=True,
        )
    else:
        sst_z_series = pd.Series(np.nan, index=clean_df.index, dtype=float)
        baseline_median_sst, sigma_robust_sst = float("nan"), float("nan")
        sst_baseline_window = "n/a"
    clean_df["sst_celsius"] = sst_series.round(2)
    clean_df["sst_zscore"] = sst_z_series
    if len(df) == len(clean_df):
        df["sst_celsius"] = sst_series.round(2).values
        df["sst_zscore"] = sst_z_series.values

    # Trophic lag: max NDCI Z-score observed >= N days before each pass (within the window)
    lag_days = int(SST_COUPLED_GATES["TROPHIC_LAG_MIN_DAYS"])
    dates = pd.to_datetime(clean_df["date"])
    prior_max_ndci_z: List[float] = []
    prior_bloom_date: List[Optional[str]] = []
    for i in range(len(clean_df)):
        earlier = dates <= dates.iloc[i] - pd.Timedelta(days=lag_days)
        if earlier.any():
            sub = ndci_z_series[earlier]
            prior_max_ndci_z.append(float(sub.max()))
            prior_bloom_date.append(dates[sub.idxmax()].strftime("%Y-%m-%d"))
        else:
            prior_max_ndci_z.append(float("nan"))
            prior_bloom_date.append(None)
    clean_df["prior_max_ndci_zscore"] = prior_max_ndci_z

    latest_ndci = round(float(ndci_series.iloc[-1]), 4)
    max_ndci = round(float(ndci_series.max()), 4)
    p95_ndci = round(float(np.percentile(ndci_series, 95)), 4)

    latest_ndti = round(float(ndti_series.iloc[-1]), 4)
    max_ndti = round(float(ndti_series.max()), 4)
    p95_ndti = round(float(np.percentile(ndti_series, 95)), 4)

    latest_ndci_zscore = round(float(ndci_z_series.iloc[-1]), 2)
    max_ndci_zscore = round(float(ndci_z_series.max()), 2)
    latest_ndti_zscore = round(float(ndti_z_series.iloc[-1]), 2)
    max_ndti_zscore = round(float(ndti_z_series.max()), 2)

    max_relative_anomaly_zscore = round(max(max_ndci_zscore, max_ndti_zscore), 2)
    latest_relative_anomaly_zscore = round(max(latest_ndci_zscore, latest_ndti_zscore), 2)

    days_above_moderate_ndci = int(
        (
            (ndci_series >= NDCI_THRESHOLDS["MODERATE"])
            | (ndci_z_series >= ZSCORE_THRESHOLDS["MODERATE"])
        ).sum()
    )
    days_above_high_ndci = int(
        (
            (ndci_series >= NDCI_THRESHOLDS["HIGH"])
            | (ndci_z_series >= ZSCORE_THRESHOLDS["HIGH"])
        ).sum()
    )

    days_above_moderate_ndti = int(
        (
            (ndti_series >= NDTI_THRESHOLDS["MODERATE"])
            | (ndti_z_series >= ZSCORE_THRESHOLDS["MODERATE"])
        ).sum()
    )
    days_above_high_ndti = int(
        (
            (ndti_series >= NDTI_THRESHOLDS["HIGH"])
            | (ndti_z_series >= ZSCORE_THRESHOLDS["HIGH"])
        ).sum()
    )

    # Evaluate Dual-Gate risk level per pass (optical-only, thermal-only, combined)
    optical_risks: List[str] = []
    thermal_risks: List[str] = []
    pass_risks: List[str] = []
    for i in range(len(clean_df)):
        kw = dict(
            ndci=float(clean_df.loc[i, "ndci"]),
            ndti=float(clean_df.loc[i, "ndti"]),
            ndci_zscore=float(clean_df.loc[i, "ndci_zscore"]),
            ndti_zscore=float(clean_df.loc[i, "ndti_zscore"]),
        )
        sst_i = clean_df.loc[i, "sst_celsius"]
        sstz_i = clean_df.loc[i, "sst_zscore"]
        prior_i = prior_max_ndci_z[i]
        optical_risks.append(evaluate_pass_dual_gate(**kw))
        thermal_risks.append(evaluate_thermal_gate(sst_i, sstz_i, prior_i))
        pass_risks.append(
            evaluate_pass_dual_gate(
                **kw, sst_celsius=sst_i, sst_zscore=sstz_i, prior_max_ndci_zscore=prior_i
            )
        )
    high_mask = pd.Series([r == "HIGH" for r in pass_risks], index=clean_df.index)
    mod_mask = pd.Series([r == "MODERATE" for r in pass_risks], index=clean_df.index)
    clean_df["pass_risk"] = pass_risks
    if len(df) == len(clean_df):
        df["pass_risk"] = pass_risks

    optical_high = sum(r == "HIGH" for r in optical_risks)
    optical_mod = sum(r == "MODERATE" for r in optical_risks)
    thermal_high = sum(r == "HIGH" for r in thermal_risks)
    thermal_mod = sum(r == "MODERATE" for r in thermal_risks)

    # Trophic-lag jellyfish precursor (window level): an earlier NDCI bloom >= +1.50σ followed,
    # >= 20 days later, by an SST pass >= +1.50σ AND >= 18.5 °C
    g = SST_COUPLED_GATES
    trophic_lag_detected = False
    trophic_lag_bloom_date: Optional[str] = None
    trophic_lag_thermal_date: Optional[str] = None
    best_lag_sst: float = -999.0
    if has_sst:
        for i in range(len(clean_df)):
            prior_i = prior_max_ndci_z[i]
            cur_sst = float(clean_df.loc[i, "sst_celsius"])
            cur_sst_z = float(clean_df.loc[i, "sst_zscore"])
            if (
                _is_finite(prior_i)
                and prior_i >= g["PRIOR_BLOOM_NDCI_Z"]
                and cur_sst_z >= g["MODERATE_Z"]
                and cur_sst >= g["MODERATE_C"]
            ):
                trophic_lag_detected = True
                if cur_sst > best_lag_sst:
                    best_lag_sst = cur_sst
                    trophic_lag_bloom_date = prior_bloom_date[i]
                    trophic_lag_thermal_date = dates.iloc[i].strftime("%Y-%m-%d")

    high_risk_revisit_detections = int(high_mask.sum())
    moderate_risk_revisit_detections = int(mod_mask.sum())

    triggered_dates = [
        pd.to_datetime(d).strftime("%Y-%m-%d")
        for d in clean_df.loc[high_mask, "date"].tolist()
    ]
    moderate_triggered_dates = [
        pd.to_datetime(d).strftime("%Y-%m-%d")
        for d in clean_df.loc[mod_mask, "date"].tolist()
    ]

    if has_sst:
        latest_sst = round(float(sst_series.iloc[-1]), 2)
        max_sst = round(float(sst_series.max()), 2)
        p90_sst = round(float(np.percentile(sst_series, 90)), 2)
        max_sst_zscore = round(float(sst_z_series.max()), 2)
        latest_sst_zscore = round(float(sst_z_series.iloc[-1]), 2)
        max_sst_date = dates[sst_series.idxmax()].strftime("%Y-%m-%d")
        max_relative_anomaly_zscore = round(
            max(max_ndci_zscore, max_ndti_zscore, max_sst_zscore), 2
        )
        latest_relative_anomaly_zscore = round(
            max(latest_ndci_zscore, latest_ndti_zscore, latest_sst_zscore), 2
        )
        sst_sources = clean_df.get("sst_source")
        sst_data_source = (
            ", ".join(
                f"{k} ({v})" for k, v in sst_sources.value_counts().items()
            )
            if sst_sources is not None
            else "n/a"
        )
    else:
        latest_sst = max_sst = p90_sst = max_sst_zscore = latest_sst_zscore = float("nan")
        max_sst_date = None
        sst_data_source = "n/a"

    overall_risk, primary_driver = _classify_overall_risk(
        max_ndci=max_ndci,
        max_ndti=max_ndti,
        max_ndci_zscore=max_ndci_zscore,
        max_ndti_zscore=max_ndti_zscore,
        high_pass_count=high_risk_revisit_detections,
        moderate_pass_count=moderate_risk_revisit_detections,
        thermal_high_count=thermal_high,
        thermal_moderate_count=thermal_mod,
        optical_high_count=optical_high,
        optical_moderate_count=optical_mod,
        trophic_lag_detected=trophic_lag_detected,
        max_sst=max_sst,
        max_sst_zscore=max_sst_zscore,
    )

    lat = float(site_meta.get("lat", 0.0))
    lon = float(site_meta.get("lon", 0.0))
    buffer_deg = float(site_meta.get("buffer_deg", 0.015))
    bbox = compute_bounding_box(lat=lat, lon=lon, buffer_deg=buffer_deg)

    start_str = pd.to_datetime(date_range[0]).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(date_range[1]).strftime("%Y-%m-%d")
    data_source_mode = str(clean_df["data_source"].iloc[0])

    hazard_metrics: Dict[str, Any] = {
        "latest_ndci": latest_ndci,
        "max_ndci": max_ndci,
        "p95_ndci": p95_ndci,
        "baseline_median_ndci": baseline_median_ndci,
        "sigma_robust_ndci": sigma_robust_ndci,
        "latest_ndci_zscore": latest_ndci_zscore,
        "max_ndci_zscore": max_ndci_zscore,
        "latest_ndti": latest_ndti,
        "max_ndti": max_ndti,
        "p95_ndti": p95_ndti,
        "baseline_median_ndti": baseline_median_ndti,
        "sigma_robust_ndti": sigma_robust_ndti,
        "latest_ndti_zscore": latest_ndti_zscore,
        "max_ndti_zscore": max_ndti_zscore,
        "max_relative_anomaly_zscore": max_relative_anomaly_zscore,
        "latest_relative_anomaly_zscore": latest_relative_anomaly_zscore,
        "days_above_moderate_ndci": days_above_moderate_ndci,
        "days_above_high_ndci": days_above_high_ndci,
        "days_above_moderate_ndti": days_above_moderate_ndti,
        "days_above_high_ndti": days_above_high_ndti,
        "high_risk_revisit_detections": high_risk_revisit_detections,
        "moderate_risk_revisit_detections": moderate_risk_revisit_detections,
        "latest_sst": latest_sst,
        "max_sst": max_sst,
        "max_sst_date": max_sst_date,
        "p90_sst": p90_sst,
        "baseline_median_sst": baseline_median_sst,
        "sigma_robust_sst": sigma_robust_sst,
        "sst_baseline_window": sst_baseline_window,
        "latest_sst_zscore": latest_sst_zscore,
        "max_sst_zscore": max_sst_zscore,
        "sst_data_source": sst_data_source,
        "optical_high_pass_detections": int(optical_high),
        "optical_moderate_pass_detections": int(optical_mod),
        "thermal_high_pass_detections": int(thermal_high),
        "thermal_moderate_pass_detections": int(thermal_mod),
        "trophic_lag_jellyfish_precursor_detected": bool(trophic_lag_detected),
        "trophic_lag_bloom_date": trophic_lag_bloom_date,
        "trophic_lag_thermal_date": trophic_lag_thermal_date,
        "total_revisit_observations": int(len(clean_df)),
        "mean_valid_water_coverage_pct": round(
            float(clean_df["valid_water_coverage_pct"].mean()), 1
        ),
        "overall_scram_clogging_risk": overall_risk,
    }

    report: Dict[str, Any] = {
        "report_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool_name": TOOL_NAME,
            "tool_version": TOOL_VERSION,
            "data_source_mode": data_source_mode,
            "analysis_period": {
                "start_date": start_str,
                "end_date": end_str,
            },
            "thresholds_applied": {
                "NDCI_THRESHOLDS": NDCI_THRESHOLDS,
                "NDTI_THRESHOLDS": NDTI_THRESHOLDS,
                "ZSCORE_THRESHOLDS": ZSCORE_THRESHOLDS,
                "SIGMA_MIN": SIGMA_MIN,
                "SST_THRESHOLDS": SST_THRESHOLDS,
                "SST_COUPLED_GATES": SST_COUPLED_GATES,
                "MIN_VALID_WATER_PCT": MIN_VALID_WATER_PCT,
            },
        },
        "site_profile": {
            "site_name": site_name,
            "coordinates": {"lat": lat, "lon": lon},
            "bounding_box": bbox,
            "reactor_type": site_meta.get("type", "N/A"),
            "cooling_system": site_meta.get("cooling", "N/A"),
            "historical_note": site_meta.get("historical_note", ""),
        },
        "hazard_metrics": hazard_metrics,
        "risk_classification": {
            "overall_scram_clogging_risk": overall_risk,
            "primary_hazard_driver": primary_driver,
            "high_risk_triggered_dates": triggered_dates,
            "moderate_risk_triggered_dates": moderate_triggered_dates,
        },
        "iaea_regulatory_context": {
            "applicable_standards": [
                IAEA_REFERENCES["SSR-1"],
                IAEA_REFERENCES["SSG-18"],
                IAEA_REFERENCES["NHSI-SMR"],
            ],
            "preliminary_irrs_checklist": _build_irrs_checklist(
                risk_level=overall_risk,
                hazard_metrics=hazard_metrics,
                site_meta=site_meta,
            ),
        },
        "ai_knowledge_management_extension": {
            "module_status": "PHASE_2_HOOK_READY",
            "rag_target_corpus": [
                "IAEA SSR-1 (Requirement 20: Biological Phenomena)",
                "IAEA SSG-18 (Section 5: Intake Blockage & Traveling Screen ΔP Monitoring)",
                "IAEA IRS / IRRS Mission Historical Operating Experience Database",
                "NHSI SMR Standardized Site Evaluation Dossiers",
            ],
            "integration_note": (
                "Pass this report dictionary directly into `advisor_hook.run_rag_regulatory_analysis(report)` "
                "to invoke LLM/RAG synthesis in Phase 2."
            ),
        },
    }

    report["ai_knowledge_management_extension"]["rag_preview_summary"] = (
        run_rag_regulatory_analysis(report)
    )
    return report
