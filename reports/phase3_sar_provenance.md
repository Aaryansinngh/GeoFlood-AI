# GeoFlood AI — Phase 3: Sentinel-1 SAR Validation & Provenance Report

**Document Status:** Satellite Remote Sensing Provenance & Algorithmic Audit  
**Date:** October 2026  
**Subject:** Synthetic Aperture Radar (SAR) Flood Detection Module  
**Evaluated Artifacts:** `src/geo/sar_detection.py`, `outputs/sar_demo_data.json`

---

## 1. Executive Summary

This report establishes the technical provenance, physical foundation, and empirical boundaries of the Sentinel-1 SAR flood detection module in GeoFlood AI. 

In initial releases of the system, a headline metric of **0.991 Intersection-over-Union (IoU)** was cited for satellite SAR flood mapping. Following our rigorous scientific integrity audit, we report that this 0.991 IoU metric represents an **internal algorithmic consistency benchmark on a calibrated bimodal synthetic backscatter distribution**, rather than an independent validation against raw Sentinel-1 Level-1 Ground Range Detected (GRD) satellite scenes.

This document details the exact mathematical process that produced this result, explains the physical mechanisms that limit real-world C-band SAR accuracy to $0.70–0.85$ IoU, and outlines the production pipeline required for live Copernicus satellite ingest.

---

## 2. Satellite Sensor Specifications

The module is mathematically parameterized to model the European Space Agency's (ESA) **Copernicus Sentinel-1** satellite constellation:

* **Sensor Type:** Active Synthetic Aperture Radar (C-band SAR)
* **Central Frequency:** $5.405\text{ GHz}$ (Wavelength $\lambda \approx 5.55\text{ cm}$)
* **Operational Mode:** Interferometric Wide Swath (IW)
* **Standard Polarization:** Dual-polarisation ($\text{VV} + \text{VH}$)
* **Pixel Spacing (GRD):** $10\text{ m} \times 10\text{ m}$ (downsampled to the project's $250\text{ m}$ grid)
* **Incidence Angle Range:** $29.1^\circ$ to $46.0^\circ$ across the swath
* **Orbital Revisit:** 6 to 12 days (depending on single- or dual-satellite constellation operation)
* **Key Physical Property:** Microwave radar signals penetrate cloud cover, monsoonal overcast, and nighttime conditions, overcoming the primary limitation of optical sensors (MODIS, Sentinel-2).

---

## 3. Dataset Provenance in Repository

The SAR demonstration data resides in `outputs/sar_demo_data.json`:

* **Geographic Extent:** $100 \times 100$ grid cells ($N = 10,000$ pixels)
* **Bounding Box:**
  * Latitude: $26.6397^\circ\text{N}$ to $26.8620^\circ\text{N}$
  * Longitude: $93.3626^\circ\text{E}$ to $93.5849^\circ\text{E}$
* **Target Feature Extracted:** $\Delta \sigma^0$ (calibrated decibel difference between post-flood and pre-flood co-polarized backscatter).
* **Reference Mask:** Derived from the Dartmouth Flood Observatory (DFO_4259) MODIS $250\text{ m}$ surface water raster.
* **Sample Breakdown:**
  * Total evaluated cells: 10,000
  * Ground truth flooded cells: 2,116 ($21.16\%$ prevalence)
  * Ground truth dry cells: 7,884 ($78.84\%$)
  * Permanent water cells: Excluded via JRC perennial water mask.

---

## 4. Mathematical Formulation of Otsu SAR Detection

The detection pipeline implements an automated thresholding algorithm on the backscatter difference image $\Delta \sigma^0 = \sigma^0_{\text{post}} - \sigma^0_{\text{pre}}$ (expressed in decibels, dB):

$$\Delta \sigma^0_{i,j} = 10 \log_{10}\left(\frac{\sigma^0_{\text{post}, i, j}}{\sigma^0_{\text{pre}, i, j}}\right)$$

Under smooth open-water flooding, microwave radar energy undergoes specular reflection away from the satellite antenna, resulting in a severe drop in backscatter ($\Delta \sigma^0 \ll 0\text{ dB}$).

Otsu's method determines the optimal decision threshold $t^*$ by maximizing the between-class variance $\sigma_B^2(t)$:

$$\sigma_B^2(t) = \omega_0(t) \, \omega_1(t) \, \big[\mu_0(t) - \mu_1(t)\big]^2$$

Where:
* $\omega_0(t) = \sum_{i=1}^{t} p_i$ and $\omega_1(t) = \sum_{i=t+1}^{L} p_i$ are the cumulative class probabilities for flooded and non-flooded pixels.
* $\mu_0(t)$ and $\mu_1(t)$ are the mean backscatter differences of the respective classes.
* $t^* = \arg\max_{t} \sigma_B^2(t)$.

### Optimal Empirical Threshold in GeoFlood AI:
* Computed threshold: $t^* = -4.782\text{ dB}$
* Pixels with $\Delta \sigma^0 \le -4.782\text{ dB}$ are classified as **inundated**.

---

## 5. Deconstruction of the 0.991 IoU Metric

The evaluated contingency table for the $100 \times 100$ demonstration grid is:

| Classification | Reference Flooded | Reference Dry | Total Predicted |
| :--- | :--- | :--- | :--- |
| **Predicted Flooded** | $\text{TP} = 2,097$ | $\text{FP} = 0$ | $2,097$ |
| **Predicted Dry** | $\text{FN} = 19$ | $\text{TN} = 7,884$ | $7,903$ |
| **Total Reference** | $2,116$ | $7,884$ | $N = 10,000$ |

### Calculated Verification Metrics:
* **Intersection-over-Union (IoU / Jaccard Index):**
  $$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{2,097}{2,097 + 0 + 19} = \frac{2,097}{2,116} = \mathbf{0.9910}$$
* **Precision:** $\frac{2,097}{2,097 + 0} = \mathbf{1.0000}$
* **Recall (Sensitivity):** $\frac{2,097}{2,116} = \mathbf{0.9910}$
* **F1-Score:** $\frac{2 \times 1.0000 \times 0.9910}{1.0000 + 0.9910} = \mathbf{0.9955}$

### Root Cause of the Exceptional Score:
The input `sar_demo_data.json` was generated using a simulation where flooded pixels were assigned a Gaussian backscatter drop of $\mathcal{N}(-10.5\text{ dB}, 1.5^2)$ while dry pixels were assigned $\mathcal{N}(-0.5\text{ dB}, 1.2^2)$. Because the distributions have virtually zero overlap (the classes are separated by $> 6$ standard deviations), Otsu's variance maximization achieves near-total separation.

**Scientific Clarification:**  
This score validates that the **Otsu algorithm and array processing pipeline execute correctly without numerical bugs**. It does **not** represent empirical sensor accuracy against independently surveyed terrain.

---

## 6. Real-World Radar Physics & Operational Confounders

In real-world deployment on real Sentinel-1 GRD imagery, empirical IoUs typically range between **$0.70$ and $0.85$**. Four environmental and physical phenomena account for the divergence:

### 1. Wind-Induced Bragg Scattering (False Negatives)
While stagnant floodwater acts as a specular reflector, surface winds exceeding $3–5\text{ m/s}$ generate short surface capillary waves matching the C-band radar wavelength ($\approx 5.6\text{ cm}$). This produces Bragg scattering, reflecting microwave energy directly back to the sensor. The backscatter rises from $-22\text{ dB}$ to $-14\text{ dB}$, indistinguishable from unflooded soils.

### 2. Emergent Vegetation and "Double-Bounce" Scattering (False Negatives)
In the riparian grasslands of Kaziranga National Park and agricultural rice paddies, floodwaters submerge the ground beneath standing vegetation. Radar pulses strike the water surface, reflect horizontally into vertical plant stems, and bounce straight back to the receiver. This "double-bounce" mechanism **increases** radar backscatter rather than decreasing it, rendering threshold-based flood detectors blind to sub-canopy inundation.

### 3. Radar Layover, Foreshortening, and Shadow (False Positives)
Due to Sentinel-1's side-looking active geometry, steep terrain—such as the Karbi Anglong plateau bordering the southern boundary of the study area—causes radar shadowing. Slopes facing away from the radar pulse receive no illumination, returning backscatter values $\le -26\text{ dB}$. Unchecked thresholding misclassifies dry mountain shadows as standing water bodies.

### 4. Temporal Revisit Latency
With a 6-to-12 day orbital pass frequency over Assam, rapid monsoonal flood pulses frequently peak and partially drain before an overpass occurs, recording only remnant ponds.

---

## 7. Operational Transition Roadmap

To upgrade GeoFlood AI from an algorithmic benchmark to an operational satellite Earth Observation pipeline:

1. **Google Earth Engine (GEE) Integration:** Connect to `COPERNICUS/S1_GRD_FLOAT` to stream radiometrically terrain-corrected (RTC) scenes.
2. **Terrain Masking:** Enforce a strict topographic shadow mask derived from NASADEM/SRTM with a local incidence angle filter ($\theta_{\text{inc}} \ge 25^\circ$).
3. **Dual-Polarization Fusion:** Combine $\text{VV}$ (sensitive to surface roughness) and $\text{VH}$ (sensitive to volume scattering) using a ratio index $\gamma = \sigma^0_{\text{VH}} / \sigma^0_{\text{VV}}$ to disambiguate flooded vegetation.
4. **Adaptive Tiling:** Replace global Otsu with sub-region split-based thresholding (e.g., $10\text{ km} \times 10\text{ km}$ tiles) to account for regional moisture variations across the Brahmaputra valley.

---

## 8. Summary & Viva Defense Guidance

When presenting GeoFlood AI:
* Always state: *"The Sentinel-1 SAR module demonstrates the automated physics-based extraction of flood footprints via Otsu backscatter thresholding."*
* Never claim: *"Our satellite detector has a proven 99.1% accuracy on real Earth Observation imagery."*
* Emphasize: *"The 0.991 IoU is an internal verification benchmark on a calibrated bimodal distribution; operational SAR pipelines achieve 70–85% IoU due to Bragg wind scattering and vegetation double-bounce."*
