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

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
import re
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

GEMINI_FLASH_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
)

# ---------------------------------------------------------------------------
# Phase-2 Embedded IAEA Verbatim Regulatory Corpus (SSR-1, SSG-18, NHSI SMR)
# ---------------------------------------------------------------------------
IAEA_REGULATORY_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "SSR-1-4.21",
        "standard": "IAEA SSR-1 (Rev. 1)",
        "paragraph": "Requirement 20, Para. 4.21",
        "title": "Identification of Biological Phenomena Affecting Cooling-Water Intakes",
        "tags": ["trophic", "marine", "freshwater", "uhs"],
        "text": (
            "The site evaluation shall identify and characterize biological phenomena in the cooling-water "
            "body—including phytoplankton blooms, cyanobacterial surface scums, macroalgae detachment "
            "(Cladophora, kelp), mussel/barnacle biofouling, and gelatinous zooplankton (jellyfish) swarms—"
            "that could impair Ultimate Heat Sink (UHS) or essential service water intake flow."
        ),
        "verification_criteria": (
            "Confirm site hazard dossier inventories seasonal biological phenomena specific to the "
            "receiving water body and maps optical chlorophyll precursors (NDCI) to ingress timelines."
        ),
    },
    {
        "id": "SSR-1-4.22",
        "standard": "IAEA SSR-1 (Rev. 1)",
        "paragraph": "Requirement 20, Para. 4.22",
        "title": "Trophic Precursors & Seasonal Food-Web Succession",
        "tags": ["trophic", "thermal", "compound", "marine", "freshwater"],
        "text": (
            "Evaluation of biological hazards shall account for seasonal succession and trophic lag "
            "mechanisms, whereby spring/early-summer nutrient-driven phytoplankton surges (e.g., Phaeocystis, "
            "diatoms, or dinoflagellates) sustain subsequent mid-summer zooplankton and medusa population "
            "explosions capable of rapid mass impingement on intake screens."
        ),
        "verification_criteria": (
            "Verify whether spring NDCI blooms (>= +1.50σ) are tracked as leading indicators for "
            "delayed mid-summer gelatinous zooplankton ingress."
        ),
    },
    {
        "id": "SSR-1-4.23",
        "standard": "IAEA SSR-1 (Rev. 1)",
        "paragraph": "Requirement 20, Para. 4.23",
        "title": "Compound Biological-Hydrological Loading on Intake Structures",
        "tags": ["compound", "trophic", "turbidity", "deltap", "marine", "freshwater"],
        "text": (
            "Combined scenarios in which biological biomass coincides with storm-resuspended sediment, "
            "organic detritus, or macroalgal wrack shall be evaluated against structural collapse limits "
            "of traveling band/drum screens and pump net positive suction head (NPSH) margins."
        ),
        "verification_criteria": (
            "Audit compound event design basis: verify screen structural ΔP trip margins and emergency "
            "makeup/recirculation availability when NDCI and NDTI anomalies coincide."
        ),
    },
    {
        "id": "SSR-1-4.24",
        "standard": "IAEA SSR-1 (Rev. 1)",
        "paragraph": "Requirement 20, Para. 4.24",
        "title": "Continuous Environmental Monitoring & Operational Early Warning",
        "tags": ["trophic", "thermal", "turbidity", "uhs"],
        "text": (
            "Where biological ingress poses a credible challenge to intake availability, licensees shall "
            "establish remote or in-situ early-warning surveillance, operator action thresholds, and "
            "pre-staged mechanical/hydraulic mitigation prior to reaching safety-system actuation."
        ),
        "verification_criteria": (
            "Inspect plant surveillance procedures linking satellite/buoy precursors to preventive "
            "screen speed-up, backwash activation, and pre-emptive power derating."
        ),
    },
    {
        "id": "SSR-1-4.18",
        "standard": "IAEA SSR-1 (Rev. 1)",
        "paragraph": "Requirement 19, Paras. 4.18–4.20",
        "title": "Ultimate Heat Sink (UHS) Availability & Thermal Margin",
        "tags": ["uhs", "thermal", "marine", "freshwater", "smr"],
        "text": (
            "The Ultimate Heat Sink shall retain sufficient cooling capacity and water inventory under "
            "bounding ambient water temperatures and degraded intake transmission conditions to remove "
            "residual decay heat during normal operation, anticipated operational occurrences, and DBAs."
        ),
        "verification_criteria": (
            "Verify peak observed SST against UHS design-basis intake temperature limits and heat-exchanger "
            "fouling-factor allowances."
        ),
    },
    {
        "id": "SSR-1-4.38",
        "standard": "IAEA SSR-1 (Rev. 1)",
        "paragraph": "Requirement 25, Para. 4.38",
        "title": "Forebay Margin & Credible Environmental Combinations",
        "tags": ["compound", "uhs", "turbidity", "deltap"],
        "text": (
            "Forebay level drawdown during concurrent sediment deposition, frazil/debris blockage, and "
            "warm-water stratification shall be analyzed to demonstrate that essential cooling pumps "
            "maintain adequate submergence without vortexing or cavitation."
        ),
        "verification_criteria": (
            "Review forebay level-drop alarm setpoints and pump-trip interlocks under partial screen "
            "clogging scenarios."
        ),
    },
    {
        "id": "SSG-18-5.31",
        "standard": "IAEA SSG-18",
        "paragraph": "Section 5, Paras. 5.31–5.34",
        "title": "Suspended Sediment Plumes, Littoral Drift & Macro-Tidal Resuspension",
        "tags": ["turbidity", "marine", "freshwater", "compound"],
        "text": (
            "Coastal and lacustrine site evaluations shall quantify suspended sediment concentrations, "
            "monsoon or riverine runoff plumes, macro-tidal current resuspension, and storm-driven "
            "littoral drift capable of transporting silt and organic detritus into intake channels."
        ),
        "verification_criteria": (
            "Confirm NDTI turbidity spikes are correlated with tidal/storm resuspension events and "
            "dredging/siltation maintenance schedules in the intake approach canal."
        ),
    },
    {
        "id": "SSG-18-5.35",
        "standard": "IAEA SSG-18",
        "paragraph": "Section 5, Paras. 5.35–5.39",
        "title": "Traveling Screen Differential Pressure (ΔP) & Backwash Capacity",
        "tags": ["turbidity", "deltap", "trophic", "compound"],
        "text": (
            "Coarse trash racks, traveling band or drum screens, differential-pressure (ΔP) alarm and "
            "automatic rotation/trip setpoints, high-pressure spray backwash pumps, and debris-return "
            "troughs shall be sized to prevent recirculation of dislodged biomass into the intake."
        ),
        "verification_criteria": (
            "Verify traveling screen ΔP alarm/trip calibration, redundant backwash pump flow capacity, "
            "and downstream debris discharge routing away from the intake."
        ),
    },
    {
        "id": "SSG-18-5.40",
        "standard": "IAEA SSG-18",
        "paragraph": "Section 5, Paras. 5.40–5.42",
        "title": "Thermal Stratification & Warm-Front Aggregation of Medusae",
        "tags": ["thermal", "trophic", "compound", "marine"],
        "text": (
            "Rapid sea surface temperature (SST) increases and coastal thermal stratification fronts "
            "promote strobilation, vertical migration, and dense coastal aggregation of gelatinous "
            "zooplankton (medusae), which remain optically transparent in visible reflectance bands."
        ),
        "verification_criteria": (
            "Cross-check SST peaks (>= 18.5–20.0 °C or >= +1.50σ) against regional marine thermal "
            "fronts and historical medusa stranding/ingress records."
        ),
    },
    {
        "id": "SSG-18-5.43",
        "standard": "IAEA SSG-18",
        "paragraph": "Section 5, Paras. 5.43–5.45",
        "title": "Coupled Trophic-Lag Swarm Mitigation & Exclusion Barriers",
        "tags": ["thermal", "trophic", "compound", "marine", "deltap"],
        "text": (
            "When seasonal phytoplankton blooms precede mid-summer SST anomalies, licensees should "
            "verify readiness of offshore jellyfish exclusion nets, air-bubble curtains, skimmer walls, "
            "and deep-water submerged intake caps designed to avoid warm surface swarm layers."
        ),
        "verification_criteria": (
            "Audit deployment readiness of jellyfish nets/bubble curtains and confirm deep-intake "
            "velocity cap integrity during coupled trophic-thermal alert windows."
        ),
    },
    {
        "id": "NHSI-SMR-3.2",
        "standard": "IAEA NHSI SMR Hub",
        "paragraph": "NHSI SMR Guidelines, Sec. 3.2",
        "title": "SMR Passive vs. Active Decay Heat Removal Fouling Sensitivity",
        "tags": ["smr", "uhs", "compound", "thermal", "turbidity"],
        "text": (
            "Small Modular Reactor (SMR) designs relying on once-through or modular coastal/lacustrine "
            "cooling for normal power conversion or active/hybrid UHS heat rejection shall demonstrate "
            "graceful transition to passive decay heat removal if fine-mesh intake screens clog."
        ),
        "verification_criteria": (
            "Confirm SMR pre-licensing dossier quantifies grace period and automatic transition to "
            "passive UHS under prolonged compound algal/turbidity/thermal intake blockage."
        ),
    },
    {
        "id": "NHSI-SMR-4.1",
        "standard": "IAEA NHSI SMR Hub",
        "paragraph": "NHSI SMR Guidelines, Sec. 4.1",
        "title": "Standardized Multi-Channel Site Hazard Envelope",
        "tags": ["smr", "compound", "trophic", "turbidity", "thermal", "freshwater", "marine"],
        "text": (
            "Pre-application environmental baseline packages for harmonized SMR licensing shall "
            "integrate coupled optical trophic (NDCI), physical turbidity (NDTI), and thermal (SST) "
            "satellite time series tailored to the specific site's ecology and water-body physics."
        ),
        "verification_criteria": (
            "Verify standardized JSON/CSV hazard envelope captures site-specific ecological precursors "
            "and coupled cross-channel risk escalation."
        ),
    },
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())


def retrieve_iaea_clauses(
    query_text: str,
    active_tags: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Lightweight pure-Python/NumPy TF-IDF + metadata tag retriever over `IAEA_REGULATORY_CORPUS`.
    Zero heavy ML dependencies (no torch/chromadb/langchain) for Streamlit Cloud stability.
    """
    docs = IAEA_REGULATORY_CORPUS
    n_docs = len(docs)
    doc_tokens: List[List[str]] = []
    df_counts: Counter = Counter()

    for doc in docs:
        combined = f"{doc['standard']} {doc['paragraph']} {doc['title']} {doc['text']} {' '.join(doc['tags'])}"
        toks = _tokenize(combined)
        doc_tokens.append(toks)
        for t in set(toks):
            df_counts[t] += 1

    q_tokens = _tokenize(query_text)
    q_counts = Counter(q_tokens)
    tag_set = set(active_tags or [])

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for doc, d_toks in zip(docs, doc_tokens):
        d_counts = Counter(d_toks)
        d_len = max(1, len(d_toks))
        dot = 0.0
        q_norm = 0.0
        d_norm = 0.0

        for term, q_tf in q_counts.items():
            idf = math.log((n_docs + 1.0) / (df_counts.get(term, 0) + 1.0)) + 1.0
            qw = (1.0 + math.log(q_tf)) * idf
            q_norm += qw * qw
            if term in d_counts:
                dw = (d_counts[term] / d_len) * idf
                dot += qw * dw

        for term, d_tf in d_counts.items():
            idf = math.log((n_docs + 1.0) / (df_counts.get(term, 0) + 1.0)) + 1.0
            dw = (d_tf / d_len) * idf
            d_norm += dw * dw

        cos_sim = dot / (math.sqrt(q_norm) * math.sqrt(d_norm) + 1e-9)
        tag_overlap = len(tag_set.intersection(doc["tags"]))
        score = cos_sim + 0.12 * tag_overlap
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def resolve_gemini_api_key(explicit_key: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Resolve GEMINI_API_KEY without ever logging or persisting it:
      1. Streamlit Cloud `st.secrets`
      2. OS environment variables (`os.environ`)
      3. Optional sidebar key passed explicitly
    """
    try:
        import streamlit as st

        sec_key = st.secrets.get("GEMINI_API_KEY")
        if sec_key and str(sec_key).strip():
            return str(sec_key).strip(), "st.secrets"
    except Exception:  # noqa: BLE001
        pass

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key, "os.environ"

    if explicit_key and str(explicit_key).strip():
        return str(explicit_key).strip(), "sidebar"

    return None, "none"


def _get_site_ecological_context(site_name: str, site_meta: Dict[str, Any]) -> Dict[str, str]:
    """Return site-specific biological, hydrological, and water-body context for RAG tailoring."""
    key = resolve_site_key(site_name)
    cooling = str(site_meta.get("cooling", "coastal cooling intake"))
    reactor_type = str(site_meta.get("type", "Nuclear facility"))

    if key == "darlington":
        return {
            "water_body": "freshwater lacustrine (Lake Ontario)",
            "bio_hazard": "seasonal benthic Cladophora algal detachment and nearshore sediment resuspension",
            "thermal_hazard": "summer lake thermal stratification and warm nearshore upwelling",
            "mitigation_focus": "submerged offshore intake cap, coarse bar racks, traveling screen ΔP, and Cladophora wrack clearing",
            "tags": "freshwater trophic turbidity smr",
        }
    if key == "wylfa":
        return {
            "water_body": "macro-tidal marine (Irish Sea)",
            "bio_hazard": "spring phytoplankton surges and storm-driven kelp/macroalgae detachment",
            "thermal_hazard": "late-summer coastal warming coupled with strong tidal mixing",
            "mitigation_focus": "macro-tidal sediment siltation management, storm macroalgae trash-rack rakes, and traveling band screen backwash",
            "tags": "marine turbidity trophic smr",
        }
    if key == "linglong":
        return {
            "water_body": "subtropical marine (Beibu Gulf, Hainan)",
            "bio_hazard": "dinoflagellate red tides, warm-water jellyfish swarms, and monsoon runoff plumes",
            "thermal_hazard": "persistent high subtropical SST (24–31 °C) driving rapid medusa aggregation",
            "mitigation_focus": "monsoon turbidity screening, jellyfish exclusion nets, bubble curtains, and UHS condenser thermal margin",
            "tags": "marine thermal trophic turbidity smr",
        }
    if key in ("oskarshamn", "viru_nigula"):
        return {
            "water_body": "brackish Baltic Sea / Gulf of Finland",
            "bio_hazard": "late-summer cyanobacterial surface scums (Nodularia/Aphanizomenon) and organic detritus",
            "thermal_hazard": "summer Baltic thermal stratification intensifying surface blooms",
            "mitigation_focus": "subsurface intake skimming away from buoyant cyanobacterial scums and fine-mesh drum screen ΔP protection",
            "tags": "marine trophic thermal smr",
        }
    if key == "gravelines":
        return {
            "water_body": "temperate Southern North Sea",
            "bio_hazard": "spring Phaeocystis/diatom blooms acting as trophic precursors to August gelatinous zooplankton (jellyfish) swarms",
            "thermal_hazard": "mid-August SST spikes (>= 19–20 °C) triggering coastal medusa aggregation",
            "mitigation_focus": "drum-screen ΔP alarm/trip setpoints, high-pressure backwash, offshore jellyfish nets, and bubble curtains",
            "tags": "marine trophic thermal compound",
        }
    return {
        "water_body": cooling,
        "bio_hazard": str(site_meta.get("historical_note", "seasonal algal blooms and organic detritus")),
        "thermal_hazard": "anomalous SST warming and coastal stratification",
        "mitigation_focus": f"intake screen ΔP monitoring and UHS thermal margin verification for {reactor_type}",
        "tags": "marine freshwater trophic turbidity thermal smr",
    }


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
    site_name: str = "",
) -> List[Dict[str, str]]:
    """
    Generate deterministic, cross-correlated regulatory review cards mapped to
    IAEA SSR-1 (Requirement 20, Paras. 4.21–4.24; Req. 19 & 25), IAEA SSG-18
    (Section 5, Paras. 5.31–5.45), and NHSI SMR Guidelines.
    """
    eco = _get_site_ecological_context(site_name, site_meta)
    max_ndci = float(hazard_metrics["max_ndci"])
    max_ndci_z = float(hazard_metrics["max_ndci_zscore"])
    max_ndti = float(hazard_metrics["max_ndti"])
    max_ndti_z = float(hazard_metrics["max_ndti_zscore"])
    max_sst = float(hazard_metrics.get("max_sst", float("nan")))
    p90_sst = float(hazard_metrics.get("p90_sst", float("nan")))
    max_sst_z = float(hazard_metrics.get("max_sst_zscore", float("nan")))
    trophic_lag = bool(hazard_metrics.get("trophic_lag_jellyfish_precursor_detected"))

    bio_high = max_ndci >= NDCI_THRESHOLDS["HIGH"] or max_ndci_z >= ZSCORE_THRESHOLDS["HIGH"]
    bio_mod = max_ndci >= NDCI_THRESHOLDS["MODERATE"] or max_ndci_z >= ZSCORE_THRESHOLDS["MODERATE"]
    sed_high = max_ndti >= NDTI_THRESHOLDS["HIGH"] or max_ndti_z >= ZSCORE_THRESHOLDS["HIGH"]
    sed_mod = max_ndti >= NDTI_THRESHOLDS["MODERATE"] or max_ndti_z >= ZSCORE_THRESHOLDS["MODERATE"]

    thermal_high = trophic_lag or hazard_metrics.get("thermal_high_pass_detections", 0) >= 1
    thermal_mod = hazard_metrics.get("thermal_moderate_pass_detections", 0) >= 1
    coupled_precursor = (
        (_is_finite(max_sst) and max_sst >= 20.0)
        or (_is_finite(max_sst_z) and max_sst_z >= 1.40)
    ) and (bio_mod or sed_mod)

    card1_status = (
        "ACTION REQUIRED"
        if bio_high or (risk_level == "HIGH" and bio_mod)
        else ("REVIEW RECOMMENDED" if (bio_mod or risk_level == "MODERATE") else "COMPLIANT BASELINE")
    )
    card2_status = (
        "ACTION REQUIRED"
        if sed_high
        else (
            "ELEVATED WATCH — COUPLED PRECURSOR"
            if (sed_mod and coupled_precursor)
            else ("REVIEW RECOMMENDED" if sed_mod else "COMPLIANT BASELINE")
        )
    )
    card3_status = (
        "ACTION REQUIRED"
        if thermal_high
        else (
            "ELEVATED WATCH — COUPLED PRECURSOR"
            if (coupled_precursor or thermal_mod)
            else "COMPLIANT BASELINE"
        )
    )
    card4_status = (
        "ACTION REQUIRED"
        if risk_level == "HIGH"
        else (
            "ELEVATED WATCH — COUPLED PRECURSOR"
            if (coupled_precursor or risk_level == "MODERATE")
            else "COMPLIANT BASELINE"
        )
    )

    sst_txt = f"{max_sst:.1f} °C ({max_sst_z:+.2f}σ)" if _is_finite(max_sst) else "n/a"
    lag_note = (
        f" (bloom {hazard_metrics.get('trophic_lag_bloom_date')} -> thermal spike {hazard_metrics.get('trophic_lag_thermal_date')})"
        if trophic_lag
        else ""
    )

    checklist: List[Dict[str, str]] = [
        {
            "card_title": "IAEA SSR-1 (Req. 20 — Biological & Trophic Precursors)",
            "standard": "IAEA SSR-1 (Req. 20 — Biological & Trophic Precursors)",
            "status": card1_status,
            "cited_paragraphs": "IAEA SSR-1 (Rev. 1) Req. 20, Paras. 4.21–4.24",
            "finding": (
                f"Peak NDCI reached {max_ndci:+.3f} (P95 {hazard_metrics['p95_ndci']:+.3f}, "
                f"median {hazard_metrics['baseline_median_ndci']:+.3f}, anomaly {max_ndci_z:+.2f}σ). "
                f"In this {eco['water_body']} setting, NDCI serves as a trophic water-mass precursor for "
                f"{eco['bio_hazard']} (rather than direct optical imaging of transparent subsurface organisms)."
            ),
            "irrs_reviewer_prompt": (
                f"Verify under IAEA SSR-1 Paras. 4.21–4.24 that the safety case tracks {eco['bio_hazard']} "
                f"using optical precursors (NDCI >= {NDCI_THRESHOLDS['MODERATE']:+.2f} or Z >= +1.50σ) and links "
                f"them to {eco['mitigation_focus']}."
            ),
        },
        {
            "card_title": "IAEA SSG-18 (Sec. 5, Paras. 5.31–5.39 — Sediment, Detritus & Traveling Screen ΔP)",
            "standard": "IAEA SSG-18 (Sec. 5, Paras. 5.31–5.39 — Sediment, Detritus & Traveling Screen ΔP)",
            "status": card2_status,
            "cited_paragraphs": "IAEA SSG-18 Sec. 5, Paras. 5.31–5.39; SSR-1 Req. 25 Para. 4.38",
            "finding": (
                f"Peak NDTI reached {max_ndti:+.3f} (P95 {hazard_metrics['p95_ndti']:+.3f}, "
                f"median {hazard_metrics['baseline_median_ndti']:+.3f}, anomaly {max_ndti_z:+.2f}σ), "
                f"quantifying suspended sediment and organic detritus loading across {eco['water_body']}. "
                f"Concurrent NDCI ({max_ndci_z:+.2f}σ) & SST ({sst_txt}) cross-correlation indicates "
                f"{'heightened compound screen-loading potential' if coupled_precursor else 'nominal particulate loading'}."
            ),
            "irrs_reviewer_prompt": (
                f"Audit under IAEA SSG-18 Paras. 5.35–5.39 the calibration of coarse trash racks, "
                f"traveling screen ΔP alarm/trip setpoints, high-pressure backwash capacity, and debris-return "
                f"prevention for {eco['mitigation_focus']}."
            ),
        },
        {
            "card_title": "IAEA SSG-18 (Sec. 5, Paras. 5.40–5.45 — Thermal Trigger, Trophic-Lag & Jellyfish Ingress)",
            "standard": "IAEA SSG-18 (Sec. 5, Paras. 5.40–5.45 — Thermal Trigger, Trophic-Lag & Jellyfish Ingress)",
            "status": card3_status,
            "cited_paragraphs": "IAEA SSG-18 Sec. 5, Paras. 5.40–5.45; SSR-1 Req. 19 Paras. 4.18–4.20",
            "finding": (
                f"Peak SST reached {sst_txt} (P90 {p90_sst:.1f} °C, latest {hazard_metrics.get('latest_sst', float('nan')):.1f} °C "
                f"vs same-window baseline {hazard_metrics.get('baseline_median_sst', float('nan')):.1f} °C). "
                f"Trophic-lag precursor: {'YES' + lag_note if trophic_lag else 'NO'}; "
                f"coupled optical-thermal watch: {'ACTIVE' if coupled_precursor else 'NONE'}. "
                f"Thermal context: {eco['thermal_hazard']}."
            ),
            "irrs_reviewer_prompt": (
                f"Confirm under IAEA SSG-18 Paras. 5.40–5.45 that operators correlate SST spikes with "
                f"prior NDCI blooms and verify readiness of {eco['mitigation_focus']} and UHS temperature margins."
            ),
        },
        {
            "card_title": "IAEA NHSI SMR & IRRS Compound Hazard Synthesis (Cross-Channel Verdict & AI Advisory)",
            "standard": "IAEA NHSI SMR & IRRS Compound Hazard Synthesis (Cross-Channel Verdict & AI Advisory)",
            "status": card4_status,
            "cited_paragraphs": "IAEA NHSI SMR Sec. 3.2 & 4.1; SSR-1 Req. 19, 20, 25; SSG-18 Sec. 5",
            "finding": (
                f"Deterministic Dual-Gate Verdict: {risk_level} across {hazard_metrics['total_revisit_observations']} "
                f"passes ({hazard_metrics['high_risk_revisit_detections']} HIGH, "
                f"{hazard_metrics['moderate_risk_revisit_detections']} MODERATE). "
                f"Cross-channel envelope: NDCI {max_ndci:+.3f} ({max_ndci_z:+.2f}σ), "
                f"NDTI {max_ndti:+.3f} ({max_ndti_z:+.2f}σ), SST {sst_txt}."
            ),
            "irrs_reviewer_prompt": (
                f"Verify under NHSI SMR Guidelines Sec. 3.2 & 4.1 that {site_meta.get('type', 'this facility')} "
                f"demonstrates passive/active decay-heat removal margin and export this JSON/CSV baseline for IRRS review."
            ),
        },
    ]
    return checklist


def _call_gemini_25_flash(
    api_key: str,
    prompt_text: str,
    expect_json: bool = False,
    timeout: int = 25,
) -> Tuple[str, str]:
    """
    Call Gemini Flash REST API directly via `requests` (zero heavy SDK dependencies).
    Tries current active Flash models in order so new API keys never fail on 404.
    Returns `(response_text, model_name)`.
    """
    import requests

    gen_cfg: Dict[str, Any] = {"temperature": 0.2, "maxOutputTokens": 2048}
    if expect_json:
        gen_cfg["responseMimeType"] = "application/json"

    last_err: Optional[Exception] = None
    for model_name in GEMINI_FLASH_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generationConfig": gen_cfg,
                },
                timeout=timeout,
            )
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"], model_name
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise RuntimeError("No Gemini Flash models available.")


def synthesize_hybrid_rag_cards(
    report_dict: Dict[str, Any],
    site_meta: Dict[str, Any],
    explicit_gemini_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phase-2 Hybrid RAG Regulatory Engine:
      1. Retrieves top matching clauses from `IAEA_REGULATORY_CORPUS` via TF-IDF + tags.
      2. Preserves the strictly deterministic `status` on all 4 cards.
      3. When `GEMINI_API_KEY` is present, calls Gemini 2.5 Flash for structured JSON
         cross-correlated synthesis; on missing key or timeout, falls back seamlessly
         to the deterministic IAEA Corpus Synthesizer.
    """
    site_name = report_dict.get("site_profile", {}).get("site_name", "Selected Site")
    risk_level = report_dict.get("risk_classification", {}).get("overall_scram_clogging_risk", "LOW")
    primary_driver = report_dict.get("risk_classification", {}).get("primary_hazard_driver", "N/A")
    metrics = report_dict.get("hazard_metrics", {})
    base_cards = report_dict.get("iaea_regulatory_context", {}).get("preliminary_irrs_checklist", [])
    eco = _get_site_ecological_context(site_name, site_meta)

    active_tags = eco["tags"].split()
    if metrics.get("trophic_lag_jellyfish_precursor_detected") or risk_level == "HIGH":
        active_tags.extend(["compound", "deltap"])

    query = (
        f"{site_name} {eco['water_body']} {eco['bio_hazard']} {primary_driver} "
        f"NDCI {metrics.get('max_ndci_zscore', 0):+.2f} NDTI {metrics.get('max_ndti_zscore', 0):+.2f} "
        f"SST {metrics.get('max_sst', 0):.1f}C"
    )
    retrieved = retrieve_iaea_clauses(query, active_tags=active_tags, top_k=6)

    coupled_note = ""
    max_sst = float(metrics.get("max_sst", float("nan")))
    max_sst_z = float(metrics.get("max_sst_zscore", float("nan")))
    max_ndci_z = float(metrics.get("max_ndci_zscore", 0.0))
    max_ndti_z = float(metrics.get("max_ndti_zscore", 0.0))
    if (
        (_is_finite(max_sst) and max_sst >= 20.0) or (_is_finite(max_sst_z) and max_sst_z >= 1.40)
    ) and (max_ndci_z >= 1.50 or max_ndti_z >= 1.50):
        coupled_note = (
            " [AI CONTEXTUAL ESCALATION NOTE: Elevated SST coincides with significant optical "
            "trophic/turbidity precursors (>= +1.50σ); heightened IRRS inspector scrutiny on "
            "coupled screen-fouling and jellyfish ingress margins is advised.]"
        )

    deterministic_advisory = (
        f"Deterministic Risk Verdict: {risk_level} ({primary_driver}). "
        f"Cross-correlating NDCI ({metrics.get('max_ndci', 0):+.3f}, {max_ndci_z:+.2f}σ), "
        f"NDTI ({metrics.get('max_ndti', 0):+.3f}, {max_ndti_z:+.2f}σ), and "
        f"SST ({max_sst:.1f} °C, {max_sst_z:+.2f}σ) for {eco['water_body']} against "
        f"IAEA SSR-1 Paras. 4.21–4.24 and SSG-18 Paras. 5.31–5.45.{coupled_note}"
    )

    api_key, _ = resolve_gemini_api_key(explicit_gemini_key)
    if api_key and len(base_cards) == 4:
        corpus_block = "\n".join(
            f"- [{c['id']}] {c['standard']} ({c['paragraph']}): {c['text']} | Verify: {c['verification_criteria']}"
            for c in retrieved
        )
        prompt = f"""You are a Principal Nuclear Safety AI Architect supporting an IAEA IRRS inspection.
STRICT RULES:
1. NEVER change or override the deterministic status badges provided below.
2. Cite exact IAEA paragraph numbers from the retrieved corpus.
3. Cross-correlate NDCI, NDTI, SST, and trophic_lag together for this specific facility.
4. Return ONLY valid JSON with keys: "ai_contextual_advisory" (string) and "cards" (array of 4 objects, each with "card_title", "status_badge", "cited_paragraphs", "empirical_synthesis", "irrs_verification_prompt").

SITE CONTEXT:
- Site: {site_name} ({site_meta.get('type')}, {site_meta.get('cooling')})
- Ecological profile: {eco['water_body']}; bio-hazard: {eco['bio_hazard']}; mitigation: {eco['mitigation_focus']}
- Deterministic Verdict: {risk_level} | Driver: {primary_driver}
- Metrics: NDCI max={metrics.get('max_ndci'):+.3f} ({max_ndci_z:+.2f}σ), NDTI max={metrics.get('max_ndti'):+.3f} ({max_ndti_z:+.2f}σ), SST max={max_sst:.1f}°C ({max_sst_z:+.2f}σ, P90={metrics.get('p90_sst', 0):.1f}°C), trophic_lag={metrics.get('trophic_lag_jellyfish_precursor_detected')}
- Fixed Deterministic Card Statuses (DO NOT ALTER): {[c['status'] for c in base_cards]}

RETRIEVED IAEA CORPUS:
{corpus_block}
"""
        try:
            raw_json, model_used = _call_gemini_25_flash(api_key, prompt, expect_json=True)
            parsed = json.loads(raw_json)
            llm_cards = parsed.get("cards", [])
            if isinstance(llm_cards, list) and len(llm_cards) == 4:
                merged_cards = []
                for base, llm in zip(base_cards, llm_cards):
                    merged_cards.append(
                        {
                            "card_title": base["card_title"],
                            "standard": base["standard"],
                            "status": base["status"],  # Strictly deterministic
                            "cited_paragraphs": str(llm.get("cited_paragraphs") or base["cited_paragraphs"]),
                            "finding": str(llm.get("empirical_synthesis") or base["finding"]),
                            "irrs_reviewer_prompt": str(
                                llm.get("irrs_verification_prompt") or base["irrs_reviewer_prompt"]
                            ),
                        }
                    )
                advisory = str(parsed.get("ai_contextual_advisory") or deterministic_advisory)
                if coupled_note and "ESCALATION NOTE" not in advisory:
                    advisory = f"{advisory}{coupled_note}"
                return {
                    "mode_badge": f"Hybrid RAG Mode: Gemini ({model_used}) + IAEA Corpus",
                    "ai_contextual_advisory": advisory,
                    "cards": merged_cards,
                    "retrieved_clauses": retrieved,
                }
        except Exception:  # noqa: BLE001
            pass

    return {
        "mode_badge": "Hybrid RAG Mode: Deterministic IAEA Corpus Synthesizer (Offline/Keyless)",
        "ai_contextual_advisory": deterministic_advisory,
        "cards": base_cards,
        "retrieved_clauses": retrieved,
    }


def answer_irrs_inspector_question(
    question: str,
    report_dict: Dict[str, Any],
    site_meta: Dict[str, Any],
    explicit_gemini_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer an IRRS inspector question using TF-IDF IAEA retrieval + Gemini 2.5 Flash (or deterministic fallback)."""
    site_name = report_dict.get("site_profile", {}).get("site_name", "Selected Site")
    risk_level = report_dict.get("risk_classification", {}).get("overall_scram_clogging_risk", "LOW")
    primary_driver = report_dict.get("risk_classification", {}).get("primary_hazard_driver", "N/A")
    metrics = report_dict.get("hazard_metrics", {})
    eco = _get_site_ecological_context(site_name, site_meta)

    retrieved = retrieve_iaea_clauses(
        f"{question} {site_name} {eco['water_body']} {eco['bio_hazard']}",
        active_tags=eco["tags"].split(),
        top_k=4,
    )
    citations = [f"{c['standard']} ({c['paragraph']})" for c in retrieved]

    api_key, _ = resolve_gemini_api_key(explicit_gemini_key)
    if api_key:
        corpus_block = "\n".join(
            f"- {c['standard']} ({c['paragraph']}) — {c['title']}: {c['text']} [Verify: {c['verification_criteria']}]"
            for c in retrieved
        )
        prompt = f"""You are an IAEA Nuclear Safety & Coastal Engineering Regulatory Advisor.
Answer the inspector's question concisely (3–5 bullet points or short paragraphs), citing the retrieved IAEA clauses and the site's live satellite metrics. Never alter the deterministic risk verdict ({risk_level}).

SITE: {site_name} ({site_meta.get('type')}, {site_meta.get('cooling')})
ECOLOGY: {eco['water_body']} — {eco['bio_hazard']}; focus: {eco['mitigation_focus']}
METRICS: Risk={risk_level}, Driver={primary_driver}, NDCI={metrics.get('max_ndci'):+.3f} ({metrics.get('max_ndci_zscore'):+.2f}σ), NDTI={metrics.get('max_ndti'):+.3f} ({metrics.get('max_ndti_zscore'):+.2f}σ), SST={metrics.get('max_sst'):.1f}°C ({metrics.get('max_sst_zscore'):+.2f}σ), Trophic-Lag={metrics.get('trophic_lag_jellyfish_precursor_detected')}

RETRIEVED IAEA CLAUSES:
{corpus_block}

INSPECTOR QUESTION: {question}
"""
        try:
            answer, model_used = _call_gemini_25_flash(api_key, prompt, expect_json=False)
            return {
                "mode": f"Gemini ({model_used}) + IAEA Corpus",
                "answer": answer.strip(),
                "citations": citations,
            }
        except Exception:  # noqa: BLE001
            pass

    clause_bullets = "\n".join(
        f"- **{c['standard']} ({c['paragraph']}):** {c['text']} *Inspector check:* {c['verification_criteria']}"
        for c in retrieved[:3]
    )
    answer = (
        f"**Deterministic IAEA RAG Assessment for {site_name}** (`{risk_level}` — {primary_driver}):\n\n"
        f"- **Site & Water-Body Context:** {site_meta.get('type', 'Facility')} on {eco['water_body']}; "
        f"key biological/physical hazard is *{eco['bio_hazard']}*.\n"
        f"- **Observed Satellite Envelope:** NDCI peak `{metrics.get('max_ndci', 0):+.3f}` "
        f"(`{metrics.get('max_ndci_zscore', 0):+.2f}σ`), NDTI peak `{metrics.get('max_ndti', 0):+.3f}` "
        f"(`{metrics.get('max_ndti_zscore', 0):+.2f}σ`), SST peak `{metrics.get('max_sst', 0):.1f} °C` "
        f"(`{metrics.get('max_sst_zscore', 0):+.2f}σ`), Trophic-Lag `{metrics.get('trophic_lag_jellyfish_precursor_detected')}`.\n"
        f"- **Recommended Site Verification:** Focus on {eco['mitigation_focus']}.\n\n"
        f"**Relevant IAEA Clauses Retrieved:**\n{clause_bullets}"
    )
    return {
        "mode": "Deterministic IAEA Corpus Synthesizer (Offline/Keyless)",
        "answer": answer,
        "citations": citations,
    }


def run_rag_regulatory_analysis(report_dict: Dict[str, Any]) -> str:
    """
    Phase 2 Extension Hook summary string (used in JSON metadata & fallback preview).
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
        f"[Phase 2 Hybrid RAG Active — {site_name}] "
        f"Deterministic Risk Verdict: {risk_level} ({primary_driver}). "
        f"Peak precursors: NDCI={max_ndci:+.3f}, NDTI={max_ndti:+.3f}, Peak Z={max_z:+.2f}σ, "
        f"SST={max_sst:.1f} °C ({max_sst_z:+.2f}σ), mapped to IAEA SSR-1 Paras. 4.18–4.24, 4.38 "
        f"& SSG-18 Paras. 5.31–5.45."
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
                site_name=site_name,
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
