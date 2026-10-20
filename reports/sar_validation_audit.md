# GeoFlood AI — Sentinel-1 SAR Validation & Integrity Audit

**Document Status:** Formal Integrity & Methodological Audit  
**Date:** October 20, 2026  
**Subject:** Sentinel-1 Synthetic Aperture Radar (SAR) Flood Detection (Stretch Goal A)  
**Evaluated Artifacts:** `src/geo/sar_detection.py`, `outputs/sar_demo_data.json`

---

## 1. Audit Objective & Background

The GeoFlood AI prototype includes a module for satellite-based flood detection using Synthetic Aperture Radar (SAR) imagery from the European Space Agency's Sentinel-1 constellation. In prior application releases, the system displayed an **Intersection over Union (IoU) of 0.991**, with **Precision = 1.000** and **Recall = 0.991** across 10,000 grid cells ($100 \times 100$ subregion).

In accordance with scientific integrity guidelines, this audit investigates:
1. The exact provenance of the reference flood mask and SAR backscatter data.
2. Whether the SAR detector and reference labels used independent imagery.
3. Why the measured IoU reached 0.991, and whether that metric represents an independent empirical accuracy score or an internal synthetic agreement.
4. Physical mechanisms in radar remote sensing that prevent 0.99+ IoU in real-world operational deployments.

---

## 2. Investigation & Key Audit Findings

### Finding 1: Reference Mask & SAR Input Provenance
- **Data Source in Repository:** `outputs/sar_demo_data.json`
- **Subregion Analyzed:** Latitude `[26.6397°N, 26.8620°N]`, Longitude `[93.3626°E, 93.5849°E]`, grid shape `100 x 100` ($10,000$ pixels).
- **Reported Counts:** Total flooded SAR cells = $2,097$; Total reference flooded cells = $2,116$.
- **Finding:**
  The repository does **not** contain raw Sentinel-1 Level-1 Ground Range Detected (GRD) `.SAFE` archives or calibrated $\sigma_0$ backscatter rasters for the 2015 event. Instead, `sar_demo_data.json` was generated using a **calibrated synthetic backscatter distribution** modeled after published C-band radar physics (pre-flood mean $\approx -10.70\text{ dB}$, post-flood mean $\approx -12.19\text{ dB}$, with inundated water pixels experiencing an average drop of $-9.5\text{ dB}$ to $-20.5\text{ dB}$).
  
### Finding 2: Circularity and Synthetic Alignment
- **Root Cause of 0.991 IoU:**
  The synthetic radar backscatter drop was simulated directly conditional on the optical MODIS flood mask from DFO_4259. The Otsu thresholding algorithm was then applied to this clean, bimodal Gaussian difference distribution ($\Delta \sigma_0$).
- Under an idealized bimodal Gaussian distribution with low variance ($\sigma \approx 1.5\text{ dB}$), Otsu's intra-class variance minimization cleanly separates water from land with near-zero misclassification, yielding an IoU of $0.991$.
- **Scientific Verdict:**
  **This is an internal consistency / algorithmic proof-of-concept benchmark, NOT an independently validated empirical accuracy score.** Retaining this figure as proof of real-world operational SAR performance would be scientifically misleading.

---

## 3. Real-World Radar Physics & Operational Limitations

In real-world operational remote sensing, Sentinel-1 SAR flood mapping rarely achieves IoUs exceeding $0.75 - 0.85$ when evaluated against independent aerial or survey ground truth. The primary physical and environmental blockers include:

1. **Wind-Induced Bragg Scattering:**
   - Smooth open water acts as a specular reflector, bouncing radar energy away from the sensor and appearing dark (low backscatter, $\sim -22\text{ dB}$).
   - However, sustained monsoonal surface winds ($> 5\text{ m/s}$) create centimeter-scale surface capillary waves (Bragg scattering), increasing backscatter to levels indistinguishable from rough dry land ($\sim -13\text{ dB}$), causing severe false negatives.

2. **Emergent Vegetation & Double-Bounce Scattering:**
   - In floodplains with tall reeds, crops, or flooded forests (such as the riparian grasslands of Kaziranga), radar pulses undergo corner reflection (water surface $\rightarrow$ vertical stalk $\rightarrow$ sensor).
   - This "double-bounce" effect actually **increases** backscatter rather than decreasing it, causing standard Otsu thresholding to miss flooded vegetation entirely.

3. **Geometric Layover and Radar Shadow:**
   - Side-looking radar geometry produces foreshortening, layover, and shadow in hilly terrain (such as the Karbi Anglong hills bordering the southern edge of the study region).
   - Radar shadows appear completely dark ($\le -25\text{ dB}$), generating false-positive water classifications on dry mountain slopes unless rigorously masked with a high-resolution DEM.

4. **Temporal Revisit Latency:**
   - Sentinel-1 has a 6-to-12 day orbital repeat cycle. A flash flood peak occurring between satellite overpasses will be missed, capturing only receding waters.

---

## 4. Reclassification & Corrective Actions Implemented

| Item | Prior Description | Audited Reclassification |
| :--- | :--- | :--- |
| **Performance Metric** | "SAR vs MODIS IoU: 0.991 (Empirical Accuracy)" | **"Algorithmic Agreement on Calibrated Bimodal Distribution: 0.991 (Synthetic Benchmark)"** |
| **Data Provenance** | Implied live Sentinel-1 radar scene | **Documented as synthetic C-band backscatter simulation based on literature parameters** |
| **Scientific Claim** | Standalone operational flood validation | **Educational demonstration of Otsu thresholding physics; disclaimer added across UI and docs** |
| **Permanent Water Handling** | Stated JRC water mask applied | **Verified that perennial river channel (Brahmaputra mainstem) is excluded from flood calculation** |

### Verified Audit Metrics Table

- **Evaluated Subregion:** $100 \times 100$ cells ($10,000$ pixels)
- **Positive Prevalence:** $21.16\%$ ($2,116$ reference flooded pixels)
- **Optimal Otsu Threshold:** $-4.78\text{ dB}$ backscatter reduction
- **True Positives (TP):** $2,097$
- **False Positives (FP):** $0$
- **False Negatives (FN):** $19$
- **True Negatives (TN):** $7,884$
- **Intersection over Union (IoU):** $0.9910$
- **Precision:** $1.0000$
- **Recall:** $0.9910$
- **F1-Score:** $0.9955$

---

## 5. Summary Statement

The Sentinel-1 SAR module successfully illustrates the fundamental microwave backscatter physics of flood detection (specular reflection reduction). However, its reported IoU of $0.991$ reflects the mathematical convergence of Otsu thresholding on an ideal bimodal simulation, not empirical satellite ground truth.

All documentation, viva notes, and application interfaces have been updated to explicitly label this module as a **methodological demonstration**, preventing unwarranted claims of operational validation.
