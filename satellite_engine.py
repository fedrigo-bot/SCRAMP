"""
Satellite data ingestion engine for SCRAM-Predictor.
Supports both Live Copernicus Data Space Ecosystem (CDSE) API queries
(Statistical API fast-path + openEO synchronous graph) and a marine-calibrated
synthetic/historical Sentinel-2 L2A simulation engine.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import struct
import warnings
import zlib
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    CLIMATOLOGICAL_BASELINES,
    INDEX_CLAMP_RANGE,
    MIN_VALID_WATER_PCT,
    OPENEO_BACKEND_URL,
    SIGMA_MIN,
    SST_CLAMP_RANGE,
    SST_CLIMATOLOGY,
    compute_bounding_box,
    resolve_site_key,
)

logger = logging.getLogger(__name__)

CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def climatological_sst(
    dates: pd.DatetimeIndex,
    site_name: str = "",
    lat: float = 50.0,
    historical_normal: bool = False,
) -> np.ndarray:
    """
    Seasonal SST climatology (°C) combining annual solar warming with a mid-August
    coastal thermal stratification surge peaking around DOY 222 (Aug 10):
      - Gravelines peak ~20.2 °C around Aug 10 (historical normal years ~18.4 °C)
      - Oskarshamn peak ~19.4 °C (historical normal years ~17.5 °C)
      - Northern Adriatic peak ~26.5 °C (historical normal years ~24.6 °C)
    When `historical_normal=True`, models the 5-year historical reference ensemble
    (normal summer stratification without an extreme SCRAM-level marine heatwave anomaly).
    """
    key = resolve_site_key(site_name)
    clim = dict(SST_CLIMATOLOGY.get(key, SST_CLIMATOLOGY["default"]))
    if key == "default":
        base_lat = float(np.clip(26.5 - 0.28 * abs(lat), 6.0, 24.0))
        clim["baseline"] = base_lat
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=float)
    peak_doy = clim.get("peak_doy", 222)
    baseline = clim.get("baseline", clim["mean"] - 0.4)
    seasonal_amp = clim.get("seasonal_amp", 0.65 * clim["amp"])
    surge_amp = clim.get("surge_amp", 0.35 * clim["amp"])
    if historical_normal:
        surge_amp = 0.32 * surge_amp
    seasonal_wave = seasonal_amp * np.cos(2.0 * np.pi * (doy - peak_doy) / 365.25)
    summer_surge = surge_amp * np.exp(-0.5 * ((doy - peak_doy) / 16.0) ** 2)
    vals = baseline + seasonal_wave + summer_surge
    return np.clip(vals, SST_CLAMP_RANGE[0], SST_CLAMP_RANGE[1])


def compute_historical_window_sst_baseline(
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    site_name: str = "",
    lat: float = 50.0,
    lon: float = 2.0,
    n_past_years: int = 5,
    live_fetcher: Optional[Any] = None,
) -> Tuple[float, float, str]:
    """
    Compute the SST baseline median (`0.0σ`) and non-parametric robust scale (`sigma_robust`)
    from the EXACT SAME calendar window (e.g., May–June) across the previous `n_past_years`
    (e.g., 2021–2025 when analyzing May–June 2026).

    In Live Mode (`live_fetcher` provided), queries the historical marine/archive SST record
    for each of the past `n_past_years` over that exact calendar window; falls back seamlessly
    to the calibrated multi-year historical ensemble when offline.
    """
    start_dt = pd.to_datetime(start_date).tz_localize(None).normalize()
    end_dt = pd.to_datetime(end_date).tz_localize(None).normalize()
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=30)

    window_days = max(1, int((end_dt - start_dt).days))
    ref_years = [end_dt.year - k for k in range(n_past_years, 0, -1)]
    year_slices = []
    for k in range(n_past_years, 0, -1):
        s_k = (start_dt - pd.DateOffset(years=k)).normalize()
        e_k = (end_dt - pd.DateOffset(years=k)).normalize()
        year_slices.append((s_k, e_k))

    hist_values: list = []

    if live_fetcher is not None:
        def _fetch_one_year(bounds: Tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
            s_k, e_k = bounds
            try:
                ser = live_fetcher(
                    lat,
                    lon,
                    s_k.strftime("%Y-%m-%d"),
                    e_k.strftime("%Y-%m-%d"),
                )
                if ser is not None and not ser.empty:
                    smoothed = ser.rolling(5, center=True, min_periods=1).mean().dropna()
                    if not smoothed.empty:
                        return smoothed.to_numpy(dtype=float)
            except Exception:  # noqa: BLE001
                pass
            return np.array([], dtype=float)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, n_past_years)) as pool:
            for arr in pool.map(_fetch_one_year, year_slices):
                if arr.size > 0:
                    hist_values.append(arr)

    if not hist_values:
        # Deterministic multi-year historical ensemble for the exact same calendar window
        base_seed = int(abs(round(lat * 10000) * 31 + round(lon * 10000) * 17)) % (2**31 - 1)
        for idx, (s_k, e_k) in enumerate(year_slices):
            d_range = pd.date_range(start=s_k, end=e_k, freq="1D")
            if len(d_range) == 0:
                d_range = pd.DatetimeIndex([s_k])
            yr_rng = np.random.default_rng(base_seed + 101 * (idx + 1))
            interannual_offset = yr_rng.normal(0.0, 0.35)
            synoptic_noise = pd.Series(yr_rng.normal(0.0, 0.65, size=len(d_range))).rolling(
                5, center=True, min_periods=1
            ).mean().to_numpy()
            yr_curve = (
                climatological_sst(d_range, site_name=site_name, lat=lat, historical_normal=True)
                + interannual_offset
                + synoptic_noise
            )
            hist_values.append(np.clip(yr_curve, SST_CLAMP_RANGE[0], SST_CLAMP_RANGE[1]))

    pooled = np.concatenate(hist_values)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        clim_ref = CLIMATOLOGICAL_BASELINES.get(
            resolve_site_key(site_name), CLIMATOLOGICAL_BASELINES["default"]
        )
        return (
            float(clim_ref.get("sst_median", 15.0)),
            float(clim_ref.get("sst_sigma", 2.35)),
            "Climatological Fallback Baseline",
        )

    hist_median = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - hist_median)))
    q75, q25 = np.percentile(pooled, [75, 25])
    iqr = float(q75 - q25)
    raw_hist_sigma = max(1.4826 * mad, 0.7413 * iqr, float(SIGMA_MIN["SST"]))

    # For wide multi-season windows (> 90 days, e.g. Apr-Sep), cap the winter-to-summer
    # solar ramp inflation at the site's synoptic seasonal scale while keeping the exact
    # historical window median; for 1-3 month windows (e.g. May-June), use raw_hist_sigma directly.
    clim_ref = CLIMATOLOGICAL_BASELINES.get(
        resolve_site_key(site_name), CLIMATOLOGICAL_BASELINES["default"]
    )
    clim_sigma_cap = float(clim_ref.get("sst_sigma", 2.35))
    if window_days > 90:
        hist_sigma = max(min(raw_hist_sigma, clim_sigma_cap), float(SIGMA_MIN["SST"]))
    else:
        hist_sigma = raw_hist_sigma

    label = (
        f"{ref_years[0]}–{ref_years[-1]} Same-Window Baseline "
        f"({start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d')})"
    )
    return round(hist_median, 4), round(hist_sigma, 4), label


def resolve_cdse_credentials(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Resolve Copernicus CDSE OAuth2 credentials in priority order:
      1. Streamlit Cloud `st.secrets` (`CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET`)
      2. Environment variables (`os.environ`)
      3. Explicitly provided UI/function arguments
    Returns `(client_id, client_secret, source_label)`.
    """
    # 1. Check Streamlit Cloud secrets safely (never fails if secrets.toml is absent)
    try:
        import streamlit as st

        sec_id = st.secrets.get("CDSE_CLIENT_ID")
        sec_secret = st.secrets.get("CDSE_CLIENT_SECRET")
        if sec_id and sec_secret:
            return str(sec_id).strip(), str(sec_secret).strip(), "st.secrets"
    except Exception:  # noqa: BLE001
        pass

    # 2. Check OS environment variables
    env_id = os.environ.get("CDSE_CLIENT_ID", "").strip()
    env_secret = os.environ.get("CDSE_CLIENT_SECRET", "").strip()
    if env_id and env_secret:
        return env_id, env_secret, "os.environ"

    # 3. Fall back to explicitly passed credentials (e.g. sidebar inputs)
    arg_id = (client_id or "").strip()
    arg_secret = (client_secret or "").strip()
    if arg_id and arg_secret:
        return arg_id, arg_secret, "manual"

    return None, None, "none"


def _get_cdse_access_token(clean_id: str, clean_secret: str, timeout: int = 15) -> str:
    """OAuth2 token for CDSE (client credentials, or e-mail/password via cdse-public)."""
    import requests

    if "@" in clean_id:
        data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": clean_id,
            "password": clean_secret,
        }
    else:
        data = {
            "grant_type": "client_credentials",
            "client_id": clean_id,
            "client_secret": clean_secret,
        }
    resp = requests.post(CDSE_TOKEN_URL, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise ValueError(f"CDSE OAuth2 authentication failed ({resp.status_code}: {resp.text})")
    return resp.json()["access_token"]


def compute_robust_zscore_series(
    series: pd.Series,
    sigma_min: float,
    clamp_range: Tuple[float, float] = INDEX_CLAMP_RANGE,
    clim_median: Optional[float] = None,
    clim_sigma: Optional[float] = None,
    strict_external_baseline: bool = False,
) -> Tuple[pd.Series, float, float]:
    """
    Outlier-immune non-parametric Robust Z-Score estimator using Median, MAD, and IQR:
        x_tilde = median(X)
        MAD = median(|X - x_tilde|)
        IQR = P75(X) - P25(X)
        sigma_robust = max(1.4826 * MAD, 0.7413 * IQR, sigma_min)
        Z_t = (X_t - x_tilde) / sigma_robust

    When `strict_external_baseline=True` (used for SST with a Same-Window Multi-Year
    Historical Baseline), `clim_median` and `clim_sigma` from the historical reference
    years for that exact calendar window are used directly.
    """
    clamped = pd.to_numeric(series, errors="coerce").clip(
        lower=clamp_range[0], upper=clamp_range[1]
    )
    valid_vals = clamped.dropna().to_numpy(dtype=float)
    if len(valid_vals) == 0:
        zeros = pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        return zeros, 0.0, float(sigma_min)

    sample_median = float(np.median(valid_vals))
    mad = float(np.median(np.abs(valid_vals - sample_median)))
    q75, q25 = np.percentile(valid_vals, [75, 25])
    iqr = float(q75 - q25)
    sample_sigma = max(1.4826 * mad, 0.7413 * iqr, float(sigma_min))

    if strict_external_baseline and clim_median is not None and clim_sigma is not None:
        median_val = float(clim_median)
        sigma_robust = max(float(clim_sigma), float(sigma_min))
    elif len(valid_vals) <= 10 and clim_median is not None and clim_sigma is not None:
        # For short windows (<= 10 passes, ~1-2 months), evaluate anomalies directly
        # against the site's multi-month climatological baseline so a 2-week or 1-month
        # window does not self-normalize against its own local mean.
        median_val = float(clim_median)
        sigma_robust = max(float(clim_sigma), float(sigma_min))
    elif len(valid_vals) < 20 and clim_median is not None and clim_sigma is not None:
        w = (len(valid_vals) - 10) / 10.0
        median_val = w * sample_median + (1.0 - w) * float(clim_median)
        sigma_robust = max(w * sample_sigma + (1.0 - w) * float(clim_sigma), float(sigma_min))
    else:
        median_val = sample_median
        sigma_robust = sample_sigma

    z_scores = ((clamped - median_val) / sigma_robust).round(2)
    return z_scores, round(median_val, 4), round(sigma_robust, 4)


class CopernicusWaterMonitor:
    """
    Retrieves and processes Sentinel-2 L2A surface reflectance data over nuclear
    cooling-water intake channels to compute Chlorophyll (NDCI), Turbidity (NDTI),
    and site-relative Robust Z-Score hazard proxies.
    """

    REQUIRED_COLUMNS = [
        "date",
        "ndci",
        "ndti",
        "ndci_zscore",
        "ndti_zscore",
        "sst_celsius",
        "sst_zscore",
        "valid_water_coverage_pct",
        "data_source",
        "sst_source",
    ]
    OPENEO_TIMEOUT_SECONDS = 45

    def __init__(
        self,
        backend_url: str = OPENEO_BACKEND_URL,
        timeout: int = OPENEO_TIMEOUT_SECONDS,
    ) -> None:
        self.backend_url = backend_url
        self.timeout = timeout
        self.last_warning: Optional[str] = None
        self.sst_notes: list = []
        self._site_name: str = ""
        self._lat: float = 50.0
        self._lon: float = 2.0
        self._start_date: Optional[str] = None
        self._end_date: Optional[str] = None
        self._use_live_sst_baseline: bool = False
        self._last_access_token: Optional[str] = None
        self.last_sst_baseline_info: Optional[Tuple[float, float, str]] = None

    @staticmethod
    def _build_bbox_polygon(bbox: dict) -> dict:
        """
        Convert a [west, south, east, north] bounding box dictionary into a
        GeoJSON Polygon geometry required by openEO `aggregate_spatial`.
        """
        w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [w, s],
                    [e, s],
                    [e, n],
                    [w, n],
                    [w, s],
                ]
            ],
        }

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Cloud-Edge Coverage Gate (`valid_water_coverage_pct >= MIN_VALID_WATER_PCT`),
        Dark-Pixel physical range clamping (`[-0.60, +0.60]`), and compute non-parametric
        `ndci_zscore` and `ndti_zscore` columns.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ndci"] = pd.to_numeric(df["ndci"], errors="coerce")
        df["ndti"] = pd.to_numeric(df["ndti"], errors="coerce")
        df["valid_water_coverage_pct"] = pd.to_numeric(
            df["valid_water_coverage_pct"], errors="coerce"
        )

        # Drop NaN rows and enforce Cloud-Edge Coverage Gate (>= 15.0% valid water pixels)
        df = df.dropna(subset=["ndci", "ndti", "valid_water_coverage_pct"])
        df = df[df["valid_water_coverage_pct"] >= MIN_VALID_WATER_PCT].copy()

        if df.empty:
            raise ValueError(
                f"No Sentinel-2 passes met the minimum valid water coverage gate "
                f"(>={MIN_VALID_WATER_PCT:.1f}%) after cloud and dark-pixel screening."
            )

        # Clamp pass-level NDCI and NDTI to physical coastal bounds [-0.60, +0.60]
        df["ndci"] = df["ndci"].clip(
            lower=INDEX_CLAMP_RANGE[0], upper=INDEX_CLAMP_RANGE[1]
        ).round(4)
        df["ndti"] = df["ndti"].clip(
            lower=INDEX_CLAMP_RANGE[0], upper=INDEX_CLAMP_RANGE[1]
        ).round(4)

        df = df.sort_values("date").reset_index(drop=True)

        clim_ref = CLIMATOLOGICAL_BASELINES.get(
            resolve_site_key(self._site_name), CLIMATOLOGICAL_BASELINES["default"]
        )

        df["ndci_zscore"], _, _ = compute_robust_zscore_series(
            df["ndci"],
            sigma_min=SIGMA_MIN["NDCI"],
            clim_median=clim_ref.get("ndci_median"),
            clim_sigma=clim_ref.get("ndci_sigma"),
        )
        df["ndti_zscore"], _, _ = compute_robust_zscore_series(
            df["ndti"],
            sigma_min=SIGMA_MIN["NDTI"],
            clim_median=clim_ref.get("ndti_median"),
            clim_sigma=clim_ref.get("ndti_sigma"),
        )

        # Thermal channel: guarantee a gap-free sst_celsius column
        if "sst_celsius" not in df.columns:
            df["sst_celsius"] = np.nan
        if "sst_source" not in df.columns:
            df["sst_source"] = None
        df["sst_celsius"] = pd.to_numeric(df["sst_celsius"], errors="coerce")
        missing = df["sst_celsius"].isna()
        if missing.any():
            df.loc[missing, "sst_celsius"] = climatological_sst(
                pd.DatetimeIndex(df.loc[missing, "date"]), self._site_name, self._lat
            )
            df.loc[missing, "sst_source"] = "Seasonal SST Climatology (gap fill)"
        df["sst_source"] = df["sst_source"].fillna("Seasonal SST Climatology (gap fill)")
        df["sst_celsius"] = df["sst_celsius"].clip(*SST_CLAMP_RANGE).round(2)

        win_start = self._start_date or df["date"].min().strftime("%Y-%m-%d")
        win_end = self._end_date or df["date"].max().strftime("%Y-%m-%d")
        if self.last_sst_baseline_info is None:
            self.last_sst_baseline_info = compute_historical_window_sst_baseline(
                start_date=win_start,
                end_date=win_end,
                site_name=self._site_name,
                lat=self._lat,
                lon=self._lon,
                n_past_years=5,
                live_fetcher=(
                    self._fetch_open_meteo_marine_sst if self._use_live_sst_baseline else None
                ),
            )
        hist_sst_med, hist_sst_sig, hist_sst_label = self.last_sst_baseline_info

        df["sst_zscore"], _, _ = compute_robust_zscore_series(
            df["sst_celsius"],
            sigma_min=SIGMA_MIN["SST"],
            clamp_range=SST_CLAMP_RANGE,
            clim_median=hist_sst_med,
            clim_sigma=hist_sst_sig,
            strict_external_baseline=True,
        )

        out_df = df[self.REQUIRED_COLUMNS].copy()
        out_df.attrs["sst_baseline_median"] = hist_sst_med
        out_df.attrs["sst_baseline_sigma"] = hist_sst_sig
        out_df.attrs["sst_baseline_window"] = hist_sst_label
        return out_df

    # ------------------------------------------------------------------
    # Thermal channel (Sentinel-3 SLSTR S8 + Open-Meteo fusion fallback)
    # ------------------------------------------------------------------
    def _fetch_sentinel3_slstr_sst(
        self, bbox: dict, start_str: str, end_str: str, access_token: str
    ) -> pd.Series:
        """
        Daily Sentinel-3 SLSTR band S8 (10.85 µm) brightness temperature over the ROI via the
        CDSE Statistical API, converted to °C (S8 - 273.15). Cloud screening: pixels outside
        the physical seawater range [0, 35] °C (cold cloud tops) are masked.
        NOTE: S8 is a top-of-atmosphere brightness temperature (single channel, no split-window
        correction), typically 1-3 K colder than true SST; it is bias-corrected against the
        marine SST reanalysis in `_build_daily_sst_series` when overlap exists.
        """
        import requests

        evalscript = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["S8", "dataMask"] }],
    output: [
      { id: "bt", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  let ok = (s.dataMask === 1 && s.S8 >= 273.15 && s.S8 <= 308.15) ? 1 : 0;
  return { bt: [s.S8], dataMask: [ok] };
}
"""
        payload = {
            "input": {
                "bounds": {"bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]]},
                "data": [{"type": "sentinel-3-slstr", "dataFilter": {}}],
            },
            "aggregation": {
                "timeRange": {"from": f"{start_str}T00:00:00Z", "to": f"{end_str}T23:59:59Z"},
                "aggregationInterval": {"of": "P5D"},
                "evalscript": evalscript,
                "width": 12,
                "height": 12,
            },
            "calculations": {"default": {}},
        }
        resp = requests.post(
            "https://sh.dataspace.copernicus.eu/api/v1/statistics",
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
            timeout=min(12, self.timeout),
        )
        if resp.status_code != 200:
            raise RuntimeError(f"S3 SLSTR Statistical API HTTP {resp.status_code}: {resp.text[:200]}")

        vals = {}
        for item in resp.json().get("data", []):
            stats = item.get("outputs", {}).get("bt", {}).get("bands", {}).get("B0", {}).get("stats", {})
            n = stats.get("sampleCount", 0) - stats.get("noDataCount", 0)
            mean_k = stats.get("mean")
            if n <= 0 or mean_k is None:
                continue
            try:
                c = float(mean_k) - 273.15
            except (TypeError, ValueError):
                continue
            if SST_CLAMP_RANGE[0] <= c <= SST_CLAMP_RANGE[1]:
                day = pd.to_datetime(item["interval"]["from"]).tz_localize(None).normalize()
                vals[day] = c
        return pd.Series(vals, dtype=float).sort_index()

    def _fetch_open_meteo_marine_sst(
        self, lat: float, lon: float, start_str: str, end_str: str
    ) -> pd.Series:
        """Daily mean marine SST (°C) from the credential-free Open-Meteo Marine API."""
        import requests

        resp = requests.get(
            OPEN_METEO_MARINE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "sea_surface_temperature",
                "start_date": start_str,
                "end_date": end_str,
                "timezone": "UTC",
            },
            timeout=20,
        )
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})
        s = pd.Series(
            pd.to_numeric(hourly.get("sea_surface_temperature", []), errors="coerce"),
            index=pd.to_datetime(hourly.get("time", [])),
            dtype=float,
        ).dropna()
        return s.resample("1D").mean().dropna()

    def _fetch_open_meteo_air_temperature(
        self, lat: float, lon: float, start_str: str, end_str: str
    ) -> pd.Series:
        """
        Daily mean 2 m AIR temperature from Open-Meteo Archive (ERA5). This is NOT SST — it is
        only used as a labelled proxy when neither SLSTR nor marine SST are available.
        """
        import requests

        resp = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_mean",
                "start_date": start_str,
                "end_date": end_str,
                "timezone": "UTC",
            },
            timeout=20,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        return pd.Series(
            pd.to_numeric(daily.get("temperature_2m_mean", []), errors="coerce"),
            index=pd.to_datetime(daily.get("time", [])),
            dtype=float,
        ).dropna()

    def _build_daily_sst_series(
        self,
        lat: float,
        lon: float,
        bbox: dict,
        start_str: str,
        end_str: str,
        access_token: Optional[str],
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Gap-free daily SST fusion. Priority per day:
          1. Sentinel-3 SLSTR S8 (bias-corrected vs marine SST when >= 3 overlapping days,
             and rejected if > 3 °C colder than marine SST -> residual cloud contamination)
          2. Open-Meteo Marine SST
          3. Open-Meteo Archive 2 m air temperature (labelled proxy)
          4. Seasonal SST climatology
        Then interpolated and smoothed with a centred 5-day rolling mean (coastal thermal inertia).
        Returns (daily_sst_celsius, daily_source_label). Never raises.
        """
        # Pad the window so the 5-day smoothing is not truncated at the edges
        pad_start = (pd.to_datetime(start_str) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        pad_end = (pd.to_datetime(end_str) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        days = pd.date_range(pad_start, pad_end, freq="1D")
        sst = pd.Series(np.nan, index=days, dtype=float)
        src = pd.Series(None, index=days, dtype=object)
        notes = []

        s3 = pd.Series(dtype=float)
        if access_token:
            try:
                s3 = self._fetch_sentinel3_slstr_sst(bbox, pad_start, pad_end, access_token)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"S3 SLSTR unavailable ({exc.__class__.__name__})")

        marine = pd.Series(dtype=float)
        try:
            marine = self._fetch_open_meteo_marine_sst(lat, lon, pad_start, pad_end)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Open-Meteo Marine unavailable ({exc.__class__.__name__})")

        if not s3.empty:
            s3 = s3.reindex(days)
            if not marine.empty:
                m = marine.reindex(days)
                overlap = s3.notna() & m.notna()
                if overlap.sum() >= 3:
                    # reject cloud-contaminated (too cold) S8 retrievals, then bias-correct
                    cold = overlap & ((m - s3) > 3.0)
                    s3[cold] = np.nan
                    overlap = s3.notna() & m.notna()
                    if overlap.sum() >= 3:
                        s3 = s3 + float(np.median((m - s3)[overlap]))
            ok = s3.notna()
            sst[ok] = s3[ok]
            src[ok] = "Sentinel-3 SLSTR S8 (bias-corrected)"

        if not marine.empty:
            m = marine.reindex(days)
            fill = sst.isna() & m.notna()
            sst[fill] = m[fill]
            src[fill] = "Open-Meteo Marine SST"

        if sst.isna().mean() > 0.5:
            try:
                air = self._fetch_open_meteo_air_temperature(lat, lon, pad_start, pad_end).reindex(days)
                fill = sst.isna() & air.notna()
                sst[fill] = air[fill]
                src[fill] = "Open-Meteo ERA5 2m AIR temp (proxy, not SST)"
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Open-Meteo Archive unavailable ({exc.__class__.__name__})")

        # Short gaps: interpolate; long gaps: climatology
        sst = sst.interpolate(limit=4, limit_direction="both")
        src = src.ffill(limit=4).bfill(limit=4)
        still = sst.isna()
        if still.any():
            sst[still] = climatological_sst(pd.DatetimeIndex(days[still]), self._site_name, lat)
            src[still] = "Seasonal SST Climatology (gap fill)"

        sst = sst.rolling(5, center=True, min_periods=1).mean().clip(*SST_CLAMP_RANGE)
        if notes:
            logger.info("SST fusion notes: %s", "; ".join(notes))
            self.sst_notes = notes
        return sst, src

    def _attach_sst(
        self,
        df: pd.DataFrame,
        lat: float,
        lon: float,
        bbox: dict,
        start_str: str,
        end_str: str,
        access_token: Optional[str],
    ) -> pd.DataFrame:
        """Attach fused daily SST to every Sentinel-2 revisit date and recompute Z-scores."""
        sst_daily, src_daily = self._build_daily_sst_series(
            lat, lon, bbox, start_str, end_str, access_token
        )
        self._use_live_sst_baseline = True
        self.last_sst_baseline_info = None
        df = df.copy()
        days = pd.to_datetime(df["date"]).dt.normalize()
        df["sst_celsius"] = sst_daily.reindex(days, method="nearest").to_numpy()
        df["sst_source"] = src_daily.reindex(days, method="nearest").to_numpy()
        return self._finalize_dataframe(df)

    def _fetch_cdse_fast_statistical_api(
        self,
        bbox: dict,
        start_str: str,
        end_str: str,
        clean_id: str,
        clean_secret: str,
    ) -> pd.DataFrame:
        """
        High-speed Copernicus Data Space Ecosystem (CDSE) Statistical API execution
        (`https://sh.dataspace.copernicus.eu/api/v1/statistics`).

        Implements:
        - Pixel-level water mask (`SCL == 6`)
        - Dark-pixel radiometric sanity guard (`B03 >= 0.002`, `B04 >= 0.001`, `B05 >= 0.001`)
          to reject Sen2Cor aerosol over-subtraction artifacts over CDOM-rich water
        - Spatial Ratio-of-Means (`NDCI` and `NDTI` evaluated from spatial mean reflectances)
        """
        import requests

        access_token = _get_cdse_access_token(clean_id, clean_secret)
        self._last_access_token = access_token

        # Output both spatial band reflectances (B03, B04, B05) for Ratio-of-Means
        # and clamped pixel-level indices with Dark-Pixel Radiometric Guard
        evalscript = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B03", "B04", "B05", "SCL", "dataMask"]
    }],
    output: [
      { id: "bands_refl", bands: 3, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(samples) {
  let b03 = samples.B03;
  let b04 = samples.B04;
  let b05 = samples.B05;
  // Dark-pixel radiometric guard against Sen2Cor negative/near-zero aerosol over-correction
  let radiometricValid = (b03 >= 0.002 && b04 >= 0.001 && b05 >= 0.001) ? 1 : 0;
  let isWater = (samples.dataMask === 1 && samples.SCL === 6 && radiometricValid === 1) ? 1 : 0;
  return {
    bands_refl: [b03, b04, b05],
    dataMask: [isWater]
  };
}
"""
        payload = {
            "input": {
                "bounds": {
                    "bbox": [bbox["west"], bbox["south"], bbox["east"], bbox["north"]],
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "maxCloudCoverage": 40,
                            "mosaickingOrder": "leastCC",
                        },
                    }
                ],
            },
            "aggregation": {
                "timeRange": {
                    "from": f"{start_str}T00:00:00Z",
                    "to": f"{end_str}T23:59:59Z",
                },
                "aggregationInterval": {"of": "P5D"},
                "evalscript": evalscript,
                "width": 100,
                "height": 100,
            },
            "calculations": {"default": {}},
        }

        stat_resp = requests.post(
            "https://sh.dataspace.copernicus.eu/api/v1/statistics",
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
            timeout=self.timeout,
        )
        if stat_resp.status_code != 200:
            raise RuntimeError(
                f"CDSE Statistical API HTTP {stat_resp.status_code}: {stat_resp.text[:300]}"
            )

        body = stat_resp.json()
        records = []
        for item in body.get("data", []):
            interval_from = item.get("interval", {}).get("from", "")
            outputs = item.get("outputs", {})
            refl_bands = outputs.get("bands_refl", {}).get("bands", {})
            b03_stats = refl_bands.get("B0", {}).get("stats", {})
            b04_stats = refl_bands.get("B1", {}).get("stats", {})
            b05_stats = refl_bands.get("B2", {}).get("stats", {})

            sample_count = b03_stats.get("sampleCount", 0)
            no_data_count = b03_stats.get("noDataCount", 0)
            valid_water_pixels = max(0, sample_count - no_data_count)
            if sample_count <= 0 or valid_water_pixels <= 0:
                continue

            water_pct = (valid_water_pixels / float(sample_count)) * 100.0
            if water_pct < MIN_VALID_WATER_PCT:
                continue

            mean_b03 = b03_stats.get("mean")
            mean_b04 = b04_stats.get("mean")
            mean_b05 = b05_stats.get("mean")
            if mean_b03 is None or mean_b04 is None or mean_b05 is None:
                continue

            mean_b03 = float(mean_b03)
            mean_b04 = float(mean_b04)
            mean_b05 = float(mean_b05)
            if (mean_b05 + mean_b04) <= 0 or (mean_b04 + mean_b03) <= 0:
                continue

            # Ratio of spatial band means (eliminates single-pixel denominator blowups)
            ndci_val = (mean_b05 - mean_b04) / (mean_b05 + mean_b04)
            ndti_val = (mean_b04 - mean_b03) / (mean_b04 + mean_b03)

            records.append(
                {
                    "date": pd.to_datetime(interval_from).strftime("%Y-%m-%d"),
                    "ndci": round(
                        float(np.clip(ndci_val, INDEX_CLAMP_RANGE[0], INDEX_CLAMP_RANGE[1])),
                        4,
                    ),
                    "ndti": round(
                        float(np.clip(ndti_val, INDEX_CLAMP_RANGE[0], INDEX_CLAMP_RANGE[1])),
                        4,
                    ),
                    "valid_water_coverage_pct": round(float(np.clip(water_pct, 0.0, 100.0)), 1),
                    "data_source": "Copernicus CDSE Live (SENTINEL2_L2A)",
                }
            )

        if not records:
            raise ValueError(
                f"Zero cloud-free water passes (>={MIN_VALID_WATER_PCT}% coverage) returned by CDSE Statistical API."
            )

        return self._finalize_dataframe(pd.DataFrame(records))

    def fetch_live_openeo_data(
        self,
        lat: float,
        lon: float,
        buffer_deg: float,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        site_name: str = "Selected Intake Site",
    ) -> pd.DataFrame:
        """
        Mode A: Connect to Copernicus Data Space Ecosystem (CDSE), authenticate via
        OIDC Client Credentials, load Sentinel-2 L2A bands [B03, B04, B05, SCL] with
        server-side cloud filtering (eo:cloud_cover <= 40), apply water & dark-pixel
        masking, compute Ratio-of-Means NDCI/NDTI, and filter out passes with
        valid_water_coverage_pct < 15%.

        Gracefully falls back to `generate_demo_timeseries` with a clear warning
        if credentials are missing/invalid or the remote service times out.
        """
        self.last_warning = None
        self.sst_notes = []
        self._site_name = site_name
        self._lat = float(lat)
        self._lon = float(lon)
        self._last_access_token = None
        self.last_sst_baseline_info = None
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        self._start_date = start_str
        self._end_date = end_str
        self._use_live_sst_baseline = False

        resolved_id, resolved_secret, _ = resolve_cdse_credentials(client_id, client_secret)
        if not resolved_id or not resolved_secret:
            msg = (
                "Live Copernicus CDSE mode was selected without OIDC Client ID / Secret. "
                "Automatically falling back to Calibrated Historical Simulation (Demo Mode)."
            )
            self.last_warning = msg
            warnings.warn(msg, RuntimeWarning)
            return self.generate_demo_timeseries(
                lat=lat,
                lon=lon,
                start_date=start_str,
                end_date=end_str,
                site_name=site_name,
            )

        try:
            import openeo
            import requests

            bbox = compute_bounding_box(lat=lat, lon=lon, buffer_deg=buffer_deg)
            polygon = self._build_bbox_polygon(bbox)
            clean_id = resolved_id
            clean_secret = resolved_secret

            # Fast Path: Copernicus CDSE Sentinel Hub Statistical API
            try:
                s2_df = self._fetch_cdse_fast_statistical_api(
                    bbox=bbox,
                    start_str=start_str,
                    end_str=end_str,
                    clean_id=clean_id,
                    clean_secret=clean_secret,
                )
                return self._attach_sst(
                    s2_df, lat, lon, bbox, start_str, end_str, self._last_access_token
                )
            except Exception as fast_exc:
                logger.info(
                    "CDSE Statistical fast-path unavailable (%s); attempting openEO synchronous graph.",
                    fast_exc,
                )

            def _run_openeo_sync_query() -> dict:
                connection = openeo.connect(
                    self.backend_url,
                    default_timeout=self.timeout,
                    auto_validate=False,
                    retry=False,
                )
                connection.default_timeout = self.timeout

                if "@" in clean_id:
                    token_url = (
                        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
                    )
                    resp = requests.post(
                        token_url,
                        data={
                            "client_id": "cdse-public",
                            "grant_type": "password",
                            "username": clean_id,
                            "password": clean_secret,
                        },
                        timeout=min(15, self.timeout),
                    )
                    if resp.status_code != 200:
                        raise ValueError(
                            f"CDSE Email/Password login rejected ({resp.status_code}: {resp.text})."
                        )
                    access_token = resp.json().get("access_token")
                    connection.authenticate_oidc_access_token(access_token=access_token)
                else:
                    connection.authenticate_oidc_client_credentials(
                        client_id=clean_id,
                        client_secret=clean_secret,
                    )

                datacube = connection.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=bbox,
                    temporal_extent=[start_str, end_str],
                    bands=["B03", "B04", "B05", "SCL"],
                    properties={"eo:cloud_cover": lambda v: v <= 40},
                )

                scl_band = datacube.band("SCL")
                b03_raw = datacube.band("B03")
                b04_raw = datacube.band("B04")
                b05_raw = datacube.band("B05")

                # Mask out non-water (SCL != 6) and dark-pixel Sen2Cor artifacts
                # (Sentinel-2 L2A DN scale = 10000 -> 0.002 = 20 DN, 0.001 = 10 DN)
                invalid_mask = (
                    (scl_band != 6)
                    | (b03_raw < 20)
                    | (b04_raw < 10)
                    | (b05_raw < 10)
                )

                b03 = b03_raw.mask(invalid_mask)
                b04 = b04_raw.mask(invalid_mask)
                b05 = b05_raw.mask(invalid_mask)

                ndci_cube = (b05 - b04) / (b05 + b04)
                ndti_cube = (b04 - b03) / (b04 + b03)
                valid_water_cube = (scl_band == 6) * 1.0

                merged_cube = (
                    ndci_cube.add_dimension("bands", "ndci", type="bands")
                    .merge_cubes(ndti_cube.add_dimension("bands", "ndti", type="bands"))
                    .merge_cubes(
                        valid_water_cube.add_dimension("bands", "water_frac", type="bands")
                    )
                )

                spatial_cube = merged_cube.aggregate_spatial(
                    geometries=polygon,
                    reducer="mean",
                )

                return connection.execute(
                    spatial_cube,
                    timeout=self.timeout,
                    validate=False,
                )

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_run_openeo_sync_query)
            try:
                timeseries_json = future.result(timeout=self.timeout + 5)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            records = []
            for iso_ts, payload in sorted(timeseries_json.items()):
                if not payload or not payload[0]:
                    continue
                vals = payload[0]
                if len(vals) >= 2 and vals[0] is not None and vals[1] is not None:
                    water_pct = (
                        float(vals[2]) * 100.0
                        if len(vals) >= 3 and vals[2] is not None
                        else 85.0
                    )
                    if water_pct < MIN_VALID_WATER_PCT:
                        continue
                    records.append(
                        {
                            "date": pd.to_datetime(iso_ts).strftime("%Y-%m-%d"),
                            "ndci": round(
                                float(
                                    np.clip(
                                        vals[0],
                                        INDEX_CLAMP_RANGE[0],
                                        INDEX_CLAMP_RANGE[1],
                                    )
                                ),
                                4,
                            ),
                            "ndti": round(
                                float(
                                    np.clip(
                                        vals[1],
                                        INDEX_CLAMP_RANGE[0],
                                        INDEX_CLAMP_RANGE[1],
                                    )
                                ),
                                4,
                            ),
                            "valid_water_coverage_pct": round(
                                float(np.clip(water_pct, 0.0, 100.0)), 1
                            ),
                            "data_source": "Copernicus CDSE Live (SENTINEL2_L2A)",
                        }
                    )

            if not records:
                raise ValueError(
                    "openEO query returned zero valid cloud-free water observations "
                    f"meeting the >={MIN_VALID_WATER_PCT}% coverage threshold."
                )

            s2_df = self._finalize_dataframe(pd.DataFrame(records))
            token = self._last_access_token
            if token is None:
                try:
                    token = _get_cdse_access_token(clean_id, clean_secret)
                except Exception:  # noqa: BLE001
                    token = None
            return self._attach_sst(s2_df, lat, lon, bbox, start_str, end_str, token)

        except Exception as exc:
            is_timeout = "timeout" in exc.__class__.__name__.lower() or "timed out" in str(exc).lower()
            if is_timeout:
                msg = (
                    f"Copernicus CDSE openEO synchronous request exceeded the {self.timeout}s timeout "
                    f"({exc.__class__.__name__}). Gracefully falling back to Calibrated Historical Simulation (Demo Mode)."
                )
            else:
                msg = (
                    f"Copernicus openEO API connection or execution failed ({exc.__class__.__name__}: {exc}). "
                    "Gracefully falling back to Calibrated Historical Simulation (Demo Mode)."
                )
            self.last_warning = msg
            logger.warning(msg)
            warnings.warn(msg, RuntimeWarning)
            return self.generate_demo_timeseries(
                lat=lat,
                lon=lon,
                start_date=start_str,
                end_date=end_str,
                site_name=site_name,
            )

    def generate_demo_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        site_name: str = "Selected Intake Site",
    ) -> pd.DataFrame:
        """
        Mode B: Deterministic, realistic 5-day revisit Sentinel-2 L2A time-series
        generator calibrated to coastal nuclear intake seasonal dynamics:
        - Baseline winter/spring NDCI ~ -0.08 to -0.04 and NDTI ~ -0.35 to -0.25
        - Summer biological/turbidity surge peaks reaching -0.015 to +0.045 for NDCI
          and -0.17 to -0.08 for NDTI
        - Realistic offshore valid_water_coverage_pct (75%–95% on clear passes)
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if end_dt <= start_dt:
            end_dt = start_dt + pd.Timedelta(days=30)
        self._site_name = site_name
        self._lat = float(lat)
        self._lon = float(lon)
        self._start_date = start_dt.strftime("%Y-%m-%d")
        self._end_date = end_dt.strftime("%Y-%m-%d")
        self._use_live_sst_baseline = False
        self.last_sst_baseline_info = None

        revisit_dates = pd.date_range(start=start_dt, end=end_dt, freq="5D")
        if len(revisit_dates) == 0:
            revisit_dates = pd.DatetimeIndex([start_dt])

        seed_val = int(abs(round(lat * 10000) * 31 + round(lon * 10000) * 17)) % (2**31 - 1)
        rng = np.random.default_rng(seed_val)
        sst_rng = np.random.default_rng(seed_val + 7)  # independent stream: keeps NDCI/NDTI demo unchanged

        site_key = resolve_site_key(site_name)
        site_lower = site_name.lower()
        if site_key == "gravelines":
            base_ndci = -0.082
            bloom_amp = 0.072   # Summer peaks around -0.015 to +0.005 (Z > +2.2σ)
            peak_doy = 222      # Mid-August (Aug 10-14 trophic precursor & jellyfish ingress window)
            bloom_width = 18.0
            base_ndti = -0.325  # Southern North Sea baseline
            turbidity_amp = 0.165  # Peaks around -0.165 to -0.145
            mean_water_cov = 86.0
        elif site_key == "oskarshamn":
            base_ndci = -0.052
            bloom_amp = 0.088   # Baltic cyanobacterial bloom peaking around +0.034 to +0.042
            peak_doy = 218      # Early-to-mid August
            bloom_width = 21.0
            base_ndti = -0.275  # Baltic coastal baseline
            turbidity_amp = 0.175  # Peaks around -0.105 to -0.085
            mean_water_cov = 88.0
        elif site_key == "darlington":
            # Lake Ontario (freshwater): summer Cladophora algal detachment + nearshore resuspension
            base_ndci = -0.064
            bloom_amp = 0.082   # Summer benthic Cladophora & phytoplankton peak (~ +0.015 to +0.030)
            peak_doy = 212      # Late July / early August
            bloom_width = 19.0
            base_ndti = -0.305
            turbidity_amp = 0.165  # Nearshore resuspension events
            mean_water_cov = 87.0
        elif site_key == "linglong":
            # Beibu Gulf (Hainan subtropical): dinoflagellate red tides & monsoon turbidity plumes
            base_ndci = -0.048
            bloom_amp = 0.092   # Subtropical red-tide peaks (~ +0.035 to +0.050)
            peak_doy = 205      # Mid-late July / monsoon transition
            bloom_width = 22.0
            base_ndti = -0.260
            turbidity_amp = 0.185  # Monsoon runoff & coastal sediment plumes (~ -0.09)
            mean_water_cov = 84.0
        elif site_key == "wylfa":
            # Irish Sea (macro-tidal): high background tidal turbidity + spring phytoplankton surge
            base_ndci = -0.074
            bloom_amp = 0.068   # Moderate summer bloom + strong spring bloom
            peak_doy = 228      # Mid-to-late August
            bloom_width = 20.0
            base_ndti = -0.220  # Elevated macro-tidal suspended sediment baseline
            turbidity_amp = 0.180  # Peaks around -0.07 to -0.09 during equinoctial tides/storms
            mean_water_cov = 83.0
        elif site_key == "viru_nigula":
            # Gulf of Finland (brackish Baltic): intense late-summer cyanobacterial scums
            base_ndci = -0.045
            bloom_amp = 0.098   # Cyanobacterial surface scum peaks (~ +0.045 to +0.060)
            peak_doy = 214      # Early August
            bloom_width = 18.0
            base_ndti = -0.270
            turbidity_amp = 0.160  # Coastal runoff / organic detritus
            mean_water_cov = 86.0
        else:
            # Custom Candidate SMR Coordinates
            lat_factor = np.clip((60.0 - abs(lat)) / 30.0, 0.7, 1.2)
            base_ndci = -0.058 * lat_factor
            bloom_amp = 0.095 * lat_factor  # Peaks around +0.035 to +0.045
            peak_doy = 225
            bloom_width = 20.0
            base_ndti = -0.350
            turbidity_amp = 0.170  # Peaks around -0.18 to -0.15
            mean_water_cov = 89.0

        records = []
        for ts in revisit_dates:
            doy = ts.dayofyear

            # Simulate occasional cloud-cover dropout (~14% of passes rejected by cloud filter)
            cloud_roll = rng.uniform(0.0, 1.0)
            if cloud_roll < 0.14 and len(revisit_dates) > 6:
                continue

            # Primary late-summer biological bloom curve (Gaussian in day-of-year)
            primary_bloom = bloom_amp * np.exp(-0.5 * ((doy - peak_doy) / bloom_width) ** 2)
            # Spring phytoplankton / diatom pulse around mid-April / early May (DOY ~ 112)
            if site_key == "gravelines":
                spring_scale = 1.15
            elif site_key == "wylfa":
                spring_scale = 0.95  # Strong Irish Sea spring phytoplankton surge
            else:
                spring_scale = 0.45
            spring_bloom = (spring_scale * bloom_amp) * np.exp(-0.5 * ((doy - 112) / 15.0) ** 2)

            ndci_noise = rng.normal(0.0, 0.011)
            ndci_val = base_ndci + primary_bloom + spring_bloom + ndci_noise

            # Turbidity (NDTI) responds to both biological organic detritus and autumn storm mixing
            autumn_storm_pulse = (0.78 * turbidity_amp) * np.exp(
                -0.5 * ((doy - 270) / 24.0) ** 2
            )
            bio_turbidity_coupling = 0.62 * primary_bloom
            ndti_noise = rng.normal(0.0, 0.018)
            ndti_val = (
                base_ndti
                + bio_turbidity_coupling
                + autumn_storm_pulse
                + ndti_noise
            )

            # Realistic offshore valid cloud-free water coverage (75% - 95%)
            valid_water_pct = float(
                np.clip(rng.normal(mean_water_cov, 5.0), 75.0, 96.5)
            )

            # Seasonal SST (cosine climatology peaking mid-August) + small synoptic noise
            sst_val = float(climatological_sst(pd.DatetimeIndex([ts]), site_name, lat)[0])
            sst_val += sst_rng.normal(0.0, 0.25)

            records.append(
                {
                    "date": ts,
                    "ndci": round(
                        float(np.clip(ndci_val, INDEX_CLAMP_RANGE[0], INDEX_CLAMP_RANGE[1])),
                        4,
                    ),
                    "ndti": round(
                        float(np.clip(ndti_val, INDEX_CLAMP_RANGE[0], INDEX_CLAMP_RANGE[1])),
                        4,
                    ),
                    "sst_celsius": round(float(np.clip(sst_val, *SST_CLAMP_RANGE)), 2),
                    "valid_water_coverage_pct": round(valid_water_pct, 1),
                    "data_source": "Calibrated Sentinel-2 L2A Historical Simulation",
                    "sst_source": "Simulated SST Climatology (Demo)",
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(
                [
                    {
                        "date": start_dt,
                        "ndci": round(base_ndci, 4),
                        "ndti": round(base_ndti, 4),
                        "valid_water_coverage_pct": 88.0,
                        "data_source": "Calibrated Sentinel-2 L2A Historical Simulation",
                    }
                ]
            )

        return self._finalize_dataframe(df)


def compute_site_regime_profile(
    df: pd.DataFrame,
    end_date: Union[str, date, datetime],
    trophic_lag_detected: bool = False,
) -> Dict[str, Any]:
    """
    Separate **Historical Regime Characterization** from **Present-Day Operational Surveillance**:
      - `is_present_day_window`: True if `(date.today() - end_date).days <= 2`
      - `window_regime_profile`:
          * STABLE BASELINE REGIME        (N_exceed == 0)
          * EPISODIC / IMPULSE REGIME     (1 <= N_exceed <= 2 and max_streak == 1 and exceedance_pct < 18%)
          * SEASONAL / MULTI-PHASE REGIME (18% <= exceedance_pct <= 35%, or max_streak == 2, or coupled trophic-lag)
          * CHRONIC / SUSTAINED STRESS REGIME (exceedance_pct > 35% or max_streak >= 3)
      - `current_operational_risk`: evaluated strictly on the Recent Operational Tail
        (last 21 days up to `end_date`, or latest 2 passes if within 21 days), guarded by
        `days_since_last_pass > 18` -> `CLOUD_OBSCURED`.
    """
    work = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(work["date"]).dt.tz_localize(None)
    end_dt = pd.to_datetime(end_date).tz_localize(None).normalize()
    is_present = (date.today() - end_dt.date()).days <= 2

    risks = work["pass_risk"].astype(str).tolist() if "pass_risk" in work.columns else ["LOW"] * len(work)
    n_total = len(work)
    is_exceed = [r in ("MODERATE", "HIGH") for r in risks]
    n_exceed = sum(is_exceed)
    exceedance_pct = round((n_exceed / max(1, n_total)) * 100.0, 1)

    max_streak = 0
    cur_streak = 0
    for flag in is_exceed:
        if flag:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            cur_streak = 0

    z_cols = [c for c in ("ndci_zscore", "ndti_zscore", "sst_zscore") if c in work.columns]
    if z_cols:
        peak_idx = work[z_cols].max(axis=1).idxmax()
        peak_event_date = pd.to_datetime(work.loc[peak_idx, "date"]).strftime("%Y-%m-%d")
    else:
        peak_event_date = dates.iloc[-1].strftime("%Y-%m-%d")

    if n_exceed == 0:
        regime = "STABLE BASELINE REGIME"
    elif exceedance_pct > 35.0 or max_streak >= 3:
        regime = "CHRONIC / SUSTAINED STRESS REGIME"
    elif 1 <= n_exceed <= 2 and max_streak == 1 and exceedance_pct < 18.0 and not trophic_lag_detected:
        regime = "EPISODIC / IMPULSE REGIME"
    else:
        regime = "SEASONAL / MULTI-PHASE REGIME"

    latest_pass_dt = dates.iloc[-1].normalize()
    days_since_last = max(0, int((end_dt - latest_pass_dt).days))
    latest_valid_pass_date = latest_pass_dt.strftime("%Y-%m-%d")

    # Recent Operational Tail: only passes within the last 21 days up to end_date
    tail_cutoff = end_dt - pd.Timedelta(days=21)
    tail_mask = dates >= tail_cutoff
    tail_df = work.loc[tail_mask]
    if len(tail_df) > 2 and (end_dt - dates.loc[tail_df.index[-2]]).days <= 21:
        # keep all passes within the 21-day tail
        pass
    tail_risks = tail_df["pass_risk"].astype(str).tolist() if ("pass_risk" in tail_df.columns and not tail_df.empty) else []

    if days_since_last > 18 or not tail_risks:
        current_op_risk = "CLOUD_OBSCURED"
    elif "HIGH" in tail_risks:
        current_op_risk = "HIGH"
    elif "MODERATE" in tail_risks:
        current_op_risk = "MODERATE"
    else:
        current_op_risk = "LOW"

    return {
        "is_present_day_window": bool(is_present),
        "window_regime_profile": regime,
        "anomalous_passes_count": int(n_exceed),
        "total_valid_passes": int(n_total),
        "exceedance_pct": exceedance_pct,
        "max_consecutive_streak": int(max_streak),
        "peak_event_date": peak_event_date,
        "days_since_last_pass": days_since_last,
        "latest_valid_pass_date": latest_valid_pass_date,
        "current_operational_risk": current_op_risk,
        "recent_tail_count": int(len(tail_risks)),
    }


def _encode_rgb_png(rgb_uint8: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 numpy array into a PNG bytes buffer using stdlib zlib/struct."""
    h, w, _ = rgb_uint8.shape
    raw_rows = b"".join(b"\x00" + rgb_uint8[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + _chunk(b"IEND", b"")
    )


def _generate_synthetic_cutout_pair(
    lat: float,
    lon: float,
    target_date_str: str,
    ndci_val: float = 0.0,
    ndti_val: float = 0.0,
) -> Dict[str, Any]:
    """
    Generate deterministic 512x512 True Color RGB and Spatial NDCI Heatmap PNGs
    calibrated to the pass's `ndci_val` and `ndti_val`, showing the coastal intake
    shoreline (bottom edge) and offshore ±0.015° ROI bounding box.
    """
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / w
    yn = yy / h

    # Deterministic plume & coastal shoreline mask (land at bottom-right/bottom)
    seed_val = int(abs(round(lat * 1000) * 17 + round(lon * 1000) * 31)) % (2**31 - 1)
    rng = np.random.default_rng(seed_val)
    shore_y = 0.80 + 0.06 * np.sin(xn * 5.0 + 0.4) + 0.03 * np.cos(xn * 11.0)
    is_land = yn >= shore_y

    # Plume intensity driven by ndci_val (-0.15..+0.25) and ndti_val (-0.40..0.0)
    bloom_strength = float(np.clip((ndci_val + 0.10) / 0.20, 0.05, 1.35))
    turb_strength = float(np.clip((ndti_val + 0.35) / 0.25, 0.05, 1.25))
    plume = np.exp(-(((xn - 0.48) ** 2) / 0.14 + ((yn - 0.56) ** 2) / 0.09))
    filament = 0.5 + 0.5 * np.sin(xn * 9.0 + yn * 6.0 + rng.uniform(0, 3.14))
    spatial_ndci = (
        ndci_val
        - 0.04
        + 0.11 * bloom_strength * plume * (0.65 + 0.35 * filament)
        + 0.03 * np.exp(-((yn - 0.72) ** 2) / 0.015)
    )

    # --- 1) True Color RGB ---
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = 18.0 + 38.0 * turb_strength * plume + 28.0 * bloom_strength * plume * 0.4
    rgb[..., 1] = 52.0 + 68.0 * bloom_strength * plume + 32.0 * turb_strength * plume
    rgb[..., 2] = 92.0 + 35.0 * (1.0 - plume) + 20.0 * turb_strength * plume
    # Coastal land / breakwaters
    rgb[is_land, 0] = 68.0
    rgb[is_land, 1] = 74.0
    rgb[is_land, 2] = 62.0

    # --- 2) Spatial NDCI Heatmap (-0.15..+0.25 -> Dark Blue -> Cyan -> Yellow -> Red) ---
    t = np.clip((spatial_ndci + 0.15) / 0.40, 0.0, 1.0)
    ndci_rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Piecewise 4-stop gradient: (0.0: 10,25,90), (0.33: 0,210,235), (0.67: 250,230,45), (1.0: 225,35,45)
    m1 = t < 0.333
    m2 = (t >= 0.333) & (t < 0.667)
    m3 = t >= 0.667
    s1 = t[m1] / 0.333
    ndci_rgb[m1, 0] = 10.0 * (1.0 - s1)
    ndci_rgb[m1, 1] = 25.0 + 185.0 * s1
    ndci_rgb[m1, 2] = 90.0 + 145.0 * s1
    s2 = (t[m2] - 0.333) / 0.334
    ndci_rgb[m2, 0] = 250.0 * s2
    ndci_rgb[m2, 1] = 210.0 + 20.0 * s2
    ndci_rgb[m2, 2] = 235.0 - 190.0 * s2
    s3 = (t[m3] - 0.667) / 0.333
    ndci_rgb[m3, 0] = 250.0 - 25.0 * s3
    ndci_rgb[m3, 1] = 230.0 - 195.0 * s3
    ndci_rgb[m3, 2] = 45.0
    ndci_rgb[is_land] = [30.0, 41.0, 59.0]  # Muted dark slate for land

    # Draw offshore ±0.015° ROI box inside [lon±0.035, lat±0.028]
    x0, x1 = int(w * (0.035 - 0.015) / 0.070), int(w * (0.035 + 0.015) / 0.070)
    y0, y1 = int(h * (0.028 - 0.015) / 0.056), int(h * (0.028 + 0.015) / 0.056)
    for img in (rgb, ndci_rgb):
        img[y0 : y0 + 3, x0:x1] = [0.0, 240.0, 255.0]
        img[y1 - 3 : y1, x0:x1] = [0.0, 240.0, 255.0]
        img[y0:y1, x0 : x0 + 3] = [0.0, 240.0, 255.0]
        img[y0:y1, x1 - 3 : x1] = [0.0, 240.0, 255.0]

    return {
        "rgb_png": _encode_rgb_png(np.clip(rgb, 0, 255).astype(np.uint8)),
        "ndci_png": _encode_rgb_png(np.clip(ndci_rgb, 0, 255).astype(np.uint8)),
        "source_label": "Simulated Spectral Raster (Demo/Offline Mode)",
        "is_live": False,
        "bbox": [round(lon - 0.035, 6), round(lat - 0.028, 6), round(lon + 0.035, 6), round(lat + 0.028, 6)],
        "date": target_date_str,
    }


def _optional_st_cache_data(func):
    try:
        import streamlit as st

        return st.cache_data(ttl=86400, show_spinner=False)(func)
    except Exception:  # noqa: BLE001
        return func


@_optional_st_cache_data
def fetch_sentinel2_pass_imagery(
    lat: float,
    lon: float,
    target_date_str: str,
    client_id: str = "",
    client_secret: str = "",
    ndci_val: float = 0.0,
    ndti_val: float = 0.0,
    allow_cloud_secrets: bool = True,
) -> Dict[str, Any]:
    """
    Fetch 10m-native Sentinel-2 L2A 512x512 PNG cutouts (True Color RGB + Spatial NDCI Heatmap)
    from the Copernicus CDSE Sentinel Hub Process API within ±2 days of `target_date_str`.
    Falls back cleanly to a calibrated synthetic 512x512 raster if credentials are absent or HTTP fails.
    """
    if allow_cloud_secrets:
        resolved_id, resolved_secret, _ = resolve_cdse_credentials(client_id, client_secret)
    else:
        resolved_id = (client_id or "").strip() or None
        resolved_secret = (client_secret or "").strip() or None
    if not resolved_id or not resolved_secret:
        return _generate_synthetic_cutout_pair(lat, lon, target_date_str, ndci_val, ndti_val)

    bbox = [round(lon - 0.035, 6), round(lat - 0.028, 6), round(lon + 0.035, 6), round(lat + 0.028, 6)]
    dt = pd.to_datetime(target_date_str)
    from_iso = (dt - pd.Timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    to_iso = (dt + pd.Timedelta(days=2)).strftime("%Y-%m-%dT23:59:59Z")

    evalscript_rgb = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B03", "B02", "dataMask"] }],
    output: { bands: 4, sampleType: "UINT8" }
  };
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return [15, 23, 42, 255];
  let r = Math.round(Math.pow(Math.min(1.0, Math.max(0.0, s.B04 * 2.5)), 0.45) * 255);
  let g = Math.round(Math.pow(Math.min(1.0, Math.max(0.0, s.B03 * 2.5)), 0.45) * 255);
  let b = Math.round(Math.pow(Math.min(1.0, Math.max(0.0, s.B02 * 2.5)), 0.45) * 255);
  return [r, g, b, 255];
}
"""

    evalscript_ndci = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B04", "B05", "B08", "SCL", "dataMask"] }],
    output: { bands: 4, sampleType: "UINT8" }
  };
}
function evaluatePixel(s) {
  if (s.dataMask === 0) return [30, 41, 59, 255];
  let ndwi = (s.B03 + s.B08) > 0 ? (s.B03 - s.B08) / (s.B03 + s.B08) : -1.0;
  let isWater = (s.SCL === 6 || ndwi > -0.05);
  if (!isWater) return [30, 41, 59, 255];
  let denom = s.B05 + s.B04;
  if (denom <= 0.001) return [10, 25, 90, 255];
  let ndci = (s.B05 - s.B04) / denom;
  let t = Math.min(1.0, Math.max(0.0, (ndci + 0.15) / 0.40));
  let r, g, b;
  if (t < 0.333) {
    let u = t / 0.333;
    r = Math.round(10 * (1 - u));
    g = Math.round(25 + 185 * u);
    b = Math.round(90 + 145 * u);
  } else if (t < 0.667) {
    let u = (t - 0.333) / 0.334;
    r = Math.round(250 * u);
    g = Math.round(210 + 20 * u);
    b = Math.round(235 - 190 * u);
  } else {
    let u = (t - 0.667) / 0.333;
    r = Math.round(250 - 25 * u);
    g = Math.round(230 - 195 * u);
    b = 45;
  }
  return [r, g, b, 255];
}
"""

    try:
        import requests

        token = _get_cdse_access_token(resolved_id, resolved_secret, timeout=15)
        process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "image/png",
        }

        def _req(evalscript: str) -> bytes:
            payload = {
                "input": {
                    "bounds": {"bbox": bbox},
                    "data": [
                        {
                            "type": "sentinel-2-l2a",
                            "dataFilter": {
                                "timeRange": {"from": from_iso, "to": to_iso},
                                "mosaickingOrder": "leastCC",
                            },
                        }
                    ],
                },
                "output": {
                    "width": 512,
                    "height": 512,
                    "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
                },
                "evalscript": evalscript,
            }
            r = requests.post(process_url, headers=headers, json=payload, timeout=25)
            r.raise_for_status()
            return r.content

        rgb_png = _req(evalscript_rgb)
        ndci_png = _req(evalscript_ndci)
        return {
            "rgb_png": rgb_png,
            "ndci_png": ndci_png,
            "source_label": "Live Copernicus Sentinel-2 L2A (10m Cutout)",
            "is_live": True,
            "bbox": bbox,
            "date": target_date_str,
        }
    except Exception as exc:  # noqa: BLE001
        logger.info("Process API cutout fallback for %s (%s)", target_date_str, exc)
        return _generate_synthetic_cutout_pair(lat, lon, target_date_str, ndci_val, ndti_val)
