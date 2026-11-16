# SCRAM-Predictor: IRRS & SMR Cooling-Water Hazard Screening Tool

[![IAEA SSR-1](https://img.shields.io/badge/IAEA-SSR--1%20Req.%2020-0A66C2)](#regulatory-alignment)
[![IAEA SSG-18](https://img.shields.io/badge/IAEA-SSG--18%20Sec.%205-0A66C2)](#regulatory-alignment)
[![Copernicus CDSE](https://img.shields.io/badge/Copernicus-Sentinel--2%20%2B%20Sentinel--3-2A9D8F)](#three-channel-copernicus-architecture)

**SCRAM-Predictor** is an open-source, cloud-ready Earth Observation screening tool built for the **IAEA Department of Nuclear Safety and Security (NSNI)** to support **Integrated Regulatory Review Service (IRRS)** mission preparation and **Nuclear Harmonization and Standardization Initiative (NHSI) SMR Regulation Hub** site evaluations.

It screens coastal and lacustrine nuclear cooling-water intakes (Ultimate Heat Sink / once-through condensers) for biological fouling, algal proliferation, suspended sediment plumes, and thermal jellyfish-swarm triggers using live Copernicus satellite observations.

---

## Three-Channel Copernicus Architecture

1. **Sentinel-2 L2A MSI — Trophic & Biological Precursor (`NDCI`):**
   $$\text{NDCI} = \frac{B05 - B04}{B05 + B04}$$
   Tracks phytoplankton biomass surges and coastal bloom precursors using dark-pixel radiometric guards (`B03 >= 0.002`, `B04 >= 0.001`, `B05 >= 0.001`), spatial Ratio-of-Means aggregation, and an `SCL == 6` water-coverage gate (`>= 15%`).

2. **Sentinel-2 L2A MSI — Physical Turbidity & Detritus (`NDTI`):**
   $$\text{NDTI} = \frac{B04 - B03}{B04 + B03}$$
   Monitors suspended sediment plumes, storm resuspension, and organic detritus loading capable of accelerating traveling band/drum screen differential-pressure ($\Delta P$) buildup.

3. **Sentinel-3 SLSTR (`S8`) + Copernicus Marine SST — Thermal Swarm Trigger (`SST °C`):**
   Retrieves 10.85 µm thermal infrared brightness temperature (`S8 - 273.15`), fuses gaps seamlessly with marine SST reanalysis, and applies 5-day coastal thermal inertia smoothing (`rolling(5).mean()`). Evaluates anomalies against a **5-year same-calendar-window historical baseline** (e.g., May–June 2026 vs. May–June 2021–2025) so seasonal solar warming never masks mid-summer marine heatwaves.

---

## Statistical & Decision Methodology

### Outlier-Immune Robust Non-Parametric Z-Score (`MAD + IQR`)
Coastal satellite distributions are skewed by episodic blooms and Sen2Cor dark-water artifacts. Instead of parametric mean/standard deviation, each channel computes:

$$\tilde{x} = \text{median}(X), \quad \text{MAD} = \text{median}(|X - \tilde{x}|), \quad \text{IQR} = P_{75}(X) - P_{25}(X)$$
$$\sigma_{\text{robust}} = \max\left(1.4826 \cdot \text{MAD},\; 0.7413 \cdot \text{IQR},\; \sigma_{\min}\right), \quad Z_t = \frac{x_t - \tilde{x}}{\sigma_{\text{robust}}}$$

with radiometric/thermal noise floors $\sigma_{\min} = \{0.020\text{ NDCI},\; 0.035\text{ NDTI},\; 0.60^\circ\text{C SST}\}$.

### Dual-Gate Hybrid Matrix & Trophic-Lag Jellyfish Rule
* **Optical Dual-Gate (Relative $Z$-Score + Marine-Calibrated Absolute):**
  - **`HIGH`:** `ndci_zscore >= +2.20` OR `ndti_zscore >= +2.20` OR (`ndci_zscore >= +1.50` **AND** `ndti_zscore >= +1.50` compound surge) OR `NDCI >= +0.03` OR `NDTI >= -0.10`.
  - **`MODERATE`:** `ndci_zscore >= +1.50` OR `ndti_zscore >= +1.50` OR `NDCI >= -0.02` OR `NDTI >= -0.18`.
* **Coupled Trophic-Lag + Thermal Trigger (`SST`):**
  - **`HIGH`:** (`sst_zscore >= +1.65` **AND** `SST >= 19.0°C` **AND** an earlier seasonal phytoplankton bloom `NDCI Z >= +1.50` occurred $\ge 20$ days earlier) OR (`sst_zscore >= +2.20` **AND** `SST >= 20.0°C`).
  - **`MODERATE`:** `sst_zscore >= +1.50` **AND** `SST >= 18.5°C`.

---

## Regulatory Alignment

* **IAEA SSR-1 (*Requirement 20: Biological Phenomena*):** Screens trophic water-mass precursors and thermal stratification regimes supporting gelatinous zooplankton (jellyfish) ingress and macro-algal detachment.
* **IAEA SSG-18 (*Section 5: Intake Blockage & Traveling Screen $\Delta P$ Monitoring*):** Maps physical turbidity, organic detritus loading, and coupled trophic-thermal triggers to traveling screen backwash and $\Delta P$ protection verification.
* **NHSI SMR Hub:** Includes pre-configured offshore ROIs for **Gravelines NPP**, **Oskarshamn NPP**, **Darlington SMR**, **Linglong One (ACP100)**, **Wylfa SMR**, **Viru-Nigula SMR**, and **Custom User Coordinates**, with standardized JSON/CSV export and a Phase 2 LLM/RAG hook (`advisor_hook.py`).

---

## Local Setup & Streamlit Cloud Deployment

```bash
pip install -r requirements.txt
streamlit run app.py
```

For **Live Copernicus CDSE** access without manual login, add `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` to `.streamlit/secrets.toml` (locally, git-ignored) or **Advanced settings → Secrets** on Streamlit Community Cloud. All live queries are cached for 24 hours (`ttl=86400`) and rate-limited to 5 uncached calls per session.
