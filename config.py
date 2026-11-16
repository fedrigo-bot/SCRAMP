"""
Configuration constants, preset nuclear facility intake coordinates,
spectral hazard thresholds, and IAEA regulatory metadata for SCRAM-Predictor.
"""

from typing import Dict, Any
import os

TOOL_NAME = "SCRAM-Predictor: IRRS & SMR Cooling-Water Hazard Screening Tool"
TOOL_VERSION = "1.2.0-Thermal"
OPENEO_BACKEND_URL = "https://openeo.dataspace.copernicus.eu"

# Copernicus Data Space Ecosystem (CDSE) OAuth2 Client Credentials
# (Injected via Streamlit Cloud `st.secrets` or environment variables; never hardcode secrets)
DEFAULT_CDSE_CLIENT_ID = os.environ.get("CDSE_CLIENT_ID", "")
DEFAULT_CDSE_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")

# Offshore coordinates shifted slightly further seaward with a tightened 0.015° buffer
# to eliminate harbor jetty contamination, shoreline foam, and land-adjacency effects
# (ensuring >80% pure seawater pixel coverage for SCL == 6).
PRESET_SITES: Dict[str, Dict[str, Any]] = {
    "Gravelines NPP (France - North Sea Intake)": {
        "lat": 51.0320,
        "lon": 2.1250,
        "buffer_deg": 0.015,
        "type": "Operating NPP (6x 900 MWe PWR)",
        "cooling": "Once-through North Sea intake",
        "historical_note": (
            "Reference coastal NPP; experienced August 2025/2026 gelatinous zooplankton (jellyfish) "
            "intake drum clogging events."
        ),
    },
    "Oskarshamn NPP (Sweden - Baltic Sea Intake)": {
        "lat": 57.4180,
        "lon": 16.7000,
        "buffer_deg": 0.015,
        "type": "Operating NPP (1400 MWe BWR)",
        "cooling": "Once-through Baltic seawater intake",
        "historical_note": (
            "Reference brackish-water NPP; exposed to late-summer Baltic cyanobacterial blooms "
            "and thermal stratification."
        ),
    },
    "Darlington SMR (Canada - BWRX-300 Lake Ontario)": {
        "lat": 43.8550,
        "lon": -78.7200,
        "buffer_deg": 0.015,
        "type": "Construction SMR (4x 300 MWe GEH BWRX-300)",
        "cooling": "Once-through freshwater lake intake (Lake Ontario)",
        "historical_note": (
            "Flagship G7 SMR site; primary biological hazard is seasonal benthic Cladophora algal "
            "detachment and nearshore sediment resuspension."
        ),
    },
    "Linglong One / Changjiang SMR (China - ACP100 Hainan)": {
        "lat": 19.4780,
        "lon": 108.8850,
        "buffer_deg": 0.015,
        "type": "Demonstration SMR (125 MWe CNNC ACP100)",
        "cooling": "Once-through subtropical coastal intake (Beibu Gulf)",
        "historical_note": (
            "World's first commercial onshore SMR; warm subtropical waters prone to dinoflagellate "
            "red tides, jellyfish swarms, and monsoon turbidity plumes."
        ),
    },
    "Wylfa SMR (UK - Rolls-Royce SMR Irish Sea)": {
        "lat": 53.4320,
        "lon": -4.4900,
        "buffer_deg": 0.015,
        "type": "Planned SMR (470 MWe Rolls-Royce PWR)",
        "cooling": "Once-through macro-tidal coastal intake (Irish Sea)",
        "historical_note": (
            "Lead UK SMR site; subject to intense tidal current sediment resuspension, storm "
            "macroalgae detachment, and spring phytoplankton surges."
        ),
    },
    "Viru-Nigula / Letea SMR (Estonia - BWRX-300 Gulf of Finland)": {
        "lat": 59.5550,
        "lon": 26.5400,
        "buffer_deg": 0.015,
        "type": "Candidate SMR (2x 300 MWe BWRX-300)",
        "cooling": "Once-through brackish intake (Gulf of Finland)",
        "historical_note": (
            "Baltic Newcomer SMR candidate site; key IAEA SSR-1 screening focus on summer "
            "cyanobacterial surface scums and coastal runoff."
        ),
    },
    "Candidate SMR Site (Custom User Coordinates)": {
        "lat": 45.2500,
        "lon": 12.3200,
        "buffer_deg": 0.015,
        "type": "Custom Candidate SMR Deployment Site",
        "cooling": "User-defined coastal/lacustrine intake",
        "historical_note": (
            "Custom coordinate screening mode for IAEA NHSI SMR Regulation Hub baseline assessments."
        ),
    },
}

# Marine-Calibrated Sentinel-2 L2A (Sen2Cor surface reflectance) thresholds over Case-2
# coastal waters (where pure seawater absorption B03 > B04 > B05 yields negative baselines):
# NDCI (Normalized Difference Chlorophyll Index) = (B05 - B04) / (B05 + B04)
NDCI_THRESHOLDS: Dict[str, float] = {
    "MODERATE": -0.02,
    "HIGH": 0.03,
}

# NDTI (Normalized Difference Turbidity Index) = (B04 - B03) / (B04 + B03)
NDTI_THRESHOLDS: Dict[str, float] = {
    "MODERATE": -0.18,
    "HIGH": -0.10,
}

# Relative Site Anomaly (Outlier-immune Robust MAD/IQR Z-Score, in +σ above site median)
ZSCORE_THRESHOLDS: Dict[str, float] = {
    "MODERATE": 1.50,
    "HIGH": 2.20,
}

# Radiometric noise-floor constants for non-parametric robust scale estimator
SIGMA_MIN: Dict[str, float] = {
    "NDCI": 0.020,
    "NDTI": 0.035,
    "SST": 0.60,  # °C — minor thermal fluctuations must not inflate Z-scores
}

# Sea Surface Temperature (°C) thresholds for temperate coastal intakes: scyphozoan
# strobilation / medusa aggregation and thermal stratification onset. Always used together
# with the site-relative SST Z-score, so subtropical sites (e.g. Adriatic) are judged
# primarily on local thermal anomalies rather than on these absolute values alone.
SST_THRESHOLDS: Dict[str, float] = {
    "MODERATE": 18.5,
    "HIGH": 20.0,
    "EXTREME": 22.0,
}

# Coupled "Trophic-Lag + Thermal Trigger" jellyfish gates (pass level)
SST_COUPLED_GATES: Dict[str, float] = {
    "MODERATE_Z": 1.50,          # sst_zscore >= 1.50 AND sst >= 18.5 °C -> Pass-MODERATE
    "MODERATE_C": 18.5,
    "TROPHIC_LAG_Z": 1.65,       # sst_zscore >= 1.65 AND sst >= 19.0 °C AND prior bloom -> Pass-HIGH
    "TROPHIC_LAG_C": 19.0,
    "SEVERE_Z": 2.20,            # sst_zscore >= 2.20 AND sst >= 20.0 °C -> Pass-HIGH
    "SEVERE_C": 20.0,
    "PRIOR_BLOOM_NDCI_Z": 1.50,  # earlier phytoplankton bloom (NDCI Z) in the screening window
    "TROPHIC_LAG_MIN_DAYS": 20,  # bloom must precede the thermal spike by >= 20 days
}

# Physical seawater clamp (°C) for thermal retrievals
SST_CLAMP_RANGE = (0.0, 35.0)

# Seasonal SST climatology used for demo generation and as last-resort gap filler:
# Combines a temperate coastal baseline with summer warming and a mid-August thermal
# stratification / marine heatwave peak around DOY 222 (Aug 10).
SST_CLIMATOLOGY: Dict[str, Dict[str, float]] = {
    "gravelines": {"mean": 13.6, "amp": 6.6, "peak_doy": 222, "baseline": 13.2, "seasonal_amp": 4.2, "surge_amp": 2.8},    # peak ~20.2 °C ~Aug 10
    "oskarshamn": {"mean": 10.4, "amp": 9.0, "peak_doy": 218, "baseline": 10.5, "seasonal_amp": 5.8, "surge_amp": 3.1},    # peak ~19.4 °C
    "darlington": {"mean": 11.2, "amp": 10.5, "peak_doy": 220, "baseline": 11.0, "seasonal_amp": 7.5, "surge_amp": 3.2},   # Lake Ontario peak ~21.7 °C
    "linglong": {"mean": 25.6, "amp": 4.9, "peak_doy": 205, "baseline": 25.5, "seasonal_amp": 3.2, "surge_amp": 1.8},      # Subtropical Hainan ~24-30.5 °C
    "wylfa": {"mean": 12.0, "amp": 5.2, "peak_doy": 228, "baseline": 11.8, "seasonal_amp": 3.6, "surge_amp": 1.8},         # Macro-tidal Irish Sea peak ~17.2 °C
    "viru_nigula": {"mean": 10.0, "amp": 9.0, "peak_doy": 215, "baseline": 10.0, "seasonal_amp": 6.2, "surge_amp": 2.8},   # Gulf of Finland peak ~19.0 °C
    "adriatic": {"mean": 18.0, "amp": 8.5, "peak_doy": 222, "baseline": 18.2, "seasonal_amp": 5.3, "surge_amp": 3.0},      # peak ~26.5 °C
    "default": {"mean": 15.0, "amp": 6.0, "peak_doy": 220, "baseline": 14.5, "seasonal_amp": 4.0, "surge_amp": 2.5},
}

# Empirical multi-month Sentinel-2 L2A & Sentinel-3/Marine SST climatological baselines
# (median & robust sigma) used as reference anchors when screening short (<12 pass) time
# windows or when summer cloud-free sampling bias skews raw multi-month SST medians.
CLIMATOLOGICAL_BASELINES: Dict[str, Dict[str, float]] = {
    "gravelines": {
        "ndci_median": -0.0720,
        "ndci_sigma": 0.0240,
        "ndti_median": -0.2850,
        "ndti_sigma": 0.0480,
        # Coastal North Sea SST baseline (°C) & synoptic/seasonal scale
        "sst_median": 14.8,
        "sst_sigma": 2.35,
    },
    "oskarshamn": {
        "ndci_median": -0.0180,
        "ndci_sigma": 0.0320,
        "ndti_median": -0.2450,
        "ndti_sigma": 0.0550,
        "sst_median": 12.2,
        "sst_sigma": 2.60,
    },
    "darlington": {
        "ndci_median": -0.0550,
        "ndci_sigma": 0.0260,
        "ndti_median": -0.2950,
        "ndti_sigma": 0.0450,
        "sst_median": 14.0,
        "sst_sigma": 3.20,
    },
    "linglong": {
        "ndci_median": -0.0420,
        "ndci_sigma": 0.0280,
        "ndti_median": -0.2500,
        "ndti_sigma": 0.0520,
        "sst_median": 27.2,
        "sst_sigma": 1.60,
    },
    "wylfa": {
        "ndci_median": -0.0680,
        "ndci_sigma": 0.0250,
        "ndti_median": -0.2150,
        "ndti_sigma": 0.0580,
        "sst_median": 13.0,
        "sst_sigma": 1.90,
    },
    "viru_nigula": {
        "ndci_median": -0.0250,
        "ndci_sigma": 0.0340,
        "ndti_median": -0.2650,
        "ndti_sigma": 0.0500,
        "sst_median": 11.8,
        "sst_sigma": 2.80,
    },
    "default": {
        "ndci_median": -0.0500,
        "ndci_sigma": 0.0300,
        "ndti_median": -0.3400,
        "ndti_sigma": 0.0500,
        "sst_median": 15.5,
        "sst_sigma": 2.50,
    },
    "adriatic": {
        "ndci_median": -0.0500,
        "ndci_sigma": 0.0300,
        "ndti_median": -0.3400,
        "ndti_sigma": 0.0500,
        "sst_median": 20.5,
        "sst_sigma": 2.75,
    },
}


def resolve_site_key(site_name: str) -> str:
    """Map a site display name to its climatology key."""
    s = (site_name or "").lower()
    if "gravelines" in s:
        return "gravelines"
    if "oskarshamn" in s:
        return "oskarshamn"
    if "darlington" in s:
        return "darlington"
    if "linglong" in s or "changjiang" in s:
        return "linglong"
    if "wylfa" in s:
        return "wylfa"
    if "viru" in s or "letea" in s:
        return "viru_nigula"
    if "adriatic" in s:
        return "adriatic"
    return "default"

# Minimum required cloud-free water pixel coverage (%) per pass to reject cloud-edge artifacts
MIN_VALID_WATER_PCT: float = 15.0

# Physical coastal clamping bounds to guard against Sen2Cor dark-pixel aerosol over-subtraction
INDEX_CLAMP_RANGE = (-0.60, 0.60)

# Standard IAEA Safety Standards references for IRRS & NHSI SMR Hub tagging
IAEA_REFERENCES: Dict[str, Dict[str, str]] = {
    "SSR-1": {
        "code": "IAEA SSR-1",
        "title": "Site Evaluation for Nuclear Installations",
        "requirement": "Requirement 20: Biological Phenomena",
        "scope": (
            "Evaluation of coastal trophic precursors (phytoplankton surges, algal blooms) and "
            "correlated biological ingress (gelatinous zooplankton/jellyfish, macro-algae) "
            "capable of degrading ultimate heat sink (UHS) and cooling-water intake availability."
        ),
    },
    "SSG-18": {
        "code": "IAEA SSG-18",
        "title": "Meteorological and Hydrological Hazards in Site Evaluation for Nuclear Installations",
        "requirement": "Section 5: Intake Blockage & Traveling Screen ΔP Monitoring",
        "scope": (
            "Assessment of suspended sediment plumes, organic detritus, fine-screen biofouling, "
            "and traveling band screen differential-pressure (ΔP) protection setpoints."
        ),
    },
    "NHSI-SMR": {
        "code": "IAEA NHSI SMR Hub",
        "title": "Nuclear Harmonization and Standardization Initiative - Site Evaluation",
        "requirement": "Standardized Environmental Baseline Data Package",
        "scope": (
            "Harmonized site hazard characterization metrics for pre-licensing and "
            "regulatory screening of Small Modular Reactor (SMR) coastal deployments."
        ),
    },
}


def compute_bounding_box(lat: float, lon: float, buffer_deg: float = 0.015) -> Dict[str, float]:
    """
    Build a geographic bounding box [west, south, east, north] around an intake point.
    """
    return {
        "west": round(lon - buffer_deg, 6),
        "south": round(lat - buffer_deg, 6),
        "east": round(lon + buffer_deg, 6),
        "north": round(lat + buffer_deg, 6),
    }
