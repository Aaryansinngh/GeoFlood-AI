# GeoFlood AI - Project Viva Voce & Defense Notes

This guide provides concise conceptual summaries and 18 defense questions with rigorous answers for the project viva examination.

---

## 1. Core Technical Foundations

### 1.1 Neural Network Fundamentals
- **Architecture:** Feed-Forward Neural Network (Multi-Layer Perceptron) with 3 hidden layers (64, 32, 16 units).
- **Batch Normalization:** Normalizes layer inputs across the mini-batch to stabilize internal covariate shift and accelerate gradient convergence.
- **Activation Functions:** 
  - **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$ in hidden layers to introduce non-linearity while mitigating vanishing gradients.
  - **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$ at output to map logits into calibrated flood probabilities $p \in [0, 1]$.
- **Dropout (0.2):** Randomly zeroes out 20% of neuron activations during training to prevent co-adaptation and overfitting.
- **Handling Class Imbalance (`pos_weight`):** Weighted Binary Cross-Entropy loss ($L = - [w \cdot y \log(p) + (1-y) \log(1-p)]$) where $w = \frac{N_{\text{dry}}}{N_{\text{flooded}}} \approx 12.1$. This penalizes false negatives proportionally to class rarity.

### 1.2 Geospatial Data Science & Spatial Splits
- **Tobler’s First Law of Geography:** *"Everything is related to everything else, but near things are more related than distant things."*
- **Why Random Splits Fail:** If adjacent grid cells are split randomly into train and test sets, the test cells are near-duplicates of the training cells, causing severe **spatial autocorrelation leakage**. Test scores will be artificially inflated.
- **Spatial Block Holdout:** The region is partitioned into contiguous spatial blocks (e.g. 12 tiles). Whole blocks are held out as test sets, ensuring the model is evaluated on geographically unseen terrain.
- **Event Holdout:** Evaluating on an entirely different temporal event (June 2015 vs August 2015) tests whether the model learns physical terrain controls rather than memorizing spatial locations.

### 1.3 Remote Sensing & Sensor Physics
- **SRTM (Shuttle Radar Topography Mission):** C-band radar interferometry (30m resolution) flown aboard Space Shuttle *Endeavour* (2000), producing global digital elevation models.
- **MODIS (Terra & Aqua):** Optical multispectral sensors with 250m bands (Red and NIR). Used by Dartmouth Flood Observatory (DFO) to generate 3-day composite inundation masks.
- **SAR (Synthetic Aperture Radar / Sentinel-1):** Active microwave sensor operating at C-band (5.405 GHz).
  - *Specular Reflection Principle:* Smooth open water acts as a mirror, reflecting radar pulses away from the sensor. This causes a dramatic drop in backscatter intensity (e.g. from -10 dB down to -20 dB), enabling cloud-penetrating water detection.

---

## 2. 18 Comprehensive Viva Questions and Model Answers

#### Q1: What is the exact prediction target of GeoFlood AI?
**Answer:** The target is to estimate the probability that a given ~250m grid cell was inundated during a documented historical flood event, conditioned on antecedent rainfall metrics and physical terrain features. It is a binary classification task ($y \in \{0, 1\}$).

#### Q2: Is GeoFlood AI a real-time early warning system or forecasting tool?
**Answer:** No. It is explicitly an **event-based susceptibility / risk mapping prototype**. It maps terrain susceptibility given an event's meteorological footprint. Real forecasting requires lead times, numerical weather prediction, and river discharge hydrographs, which are cataloged under Future Scope.

#### Q3: Why didn't you use a standard random train/test split?
**Answer:** Random splitting on spatial grids violates the assumption of independent and identically distributed (i.i.d.) data due to spatial autocorrelation (Tobler's First Law). Adjacent pixels have virtually identical elevation and rainfall. A random split results in "near-duplicate leakage," producing deceptively high test scores. We implemented **Spatial Block Holdout** (holding out entire geographic blocks) to guarantee genuine generalization.

#### Q4: Why is accuracy an inappropriate metric for this dataset?
**Answer:** Class imbalance. Flooded cells represent only ~11.3% of the land area. A naive dummy model that predicts "dry" for every cell would achieve 88.7% accuracy while missing 100% of flooded areas. We therefore evaluate using Precision, Recall, F1-score, ROC-AUC, PR-AUC, and the False Negative Rate.

#### Q5: Why is Recall prioritized over Precision in disaster susceptibility?
**Answer:** The cost of errors is asymmetric. A False Positive generates a precautionary alert for an area that stays dry. A False Negative leaves an unwarned population facing unexpected inundation, risking lives. Therefore, minimizing the False Negative Rate ($\text{FNR} = 1 - \text{Recall}$) is the primary safety objective.

#### Q6: How was the decision threshold chosen for classification?
**Answer:** The decision threshold was optimized **strictly on the validation set** to maximize the F1-score under class imbalance. It was never tuned on the test set, preventing test snooping.

#### Q7: How did you prevent permanent water bodies from biasing the model?
**Answer:** Perennial river channels (like the Brahmaputra mainstem) were identified and masked using the European Commission JRC Global Surface Water dataset (Band 5 of GFD). Without this mask, models learn to classify permanent water rather than newly flooded dry land.

#### Q8: What are the input features, and how did you verify zero target leakage?
**Answer:** Input features are: `elevation`, `slope`, `tpi`, `relative_elevation`, `rain_event_sum`, `rain_3d`, and `rain_7d`. Zero leakage was enforced by ensuring no feature is derived from the flood mask (no distance-to-flood or flood duration features), and pre-event rainfall was strictly computed prior to event onset ($t < t_{\text{start}}$).

#### Q9: What is TPI (Topographic Position Index)?
**Answer:** $\text{TPI} = z_0 - \bar{z}$, where $z_0$ is the cell elevation and $\bar{z}$ is the mean elevation of the surrounding neighborhood (15-cell window). Negative values identify topographic hollows, depressions, and valley floors prone to ponding; positive values identify ridges.

#### Q10: What is Relative Elevation, and why is it a proxy for HAND?
**Answer:** Relative elevation is computed as $z - z_{\min}$ within a 25-cell neighborhood. It approximates Height Above Nearest Drainage (HAND) by measuring how high a cell stands relative to the nearest local drainage base.

#### Q11: Which model performed best on the held-out test set?
**Answer:** The **Random Forest** achieved the best overall discrimination with an ROC-AUC of **0.901**, PR-AUC of **0.507**, and Recall of **81.2%** (FNR 18.8%). The PyTorch Neural Network achieved an ROC-AUC of **0.805** and Recall of **60.9%**, while Logistic Regression achieved an ROC-AUC of **0.684**.

#### Q12: Why did Random Forest outperform the Neural Network on this tabular grid?
**Answer:** Tree ensembles naturally excel on tabular geospatial feature matrices with non-linear threshold effects (e.g. sharp elevation barriers). Neural networks typically require larger training sets or spatial convolutions (CNNs) to outperform gradient boosted trees on tabular data.

#### Q13: How did the Neural Network perform in the Event Holdout test?
**Answer:** When trained on the June 2015 event (`DFO_4259`) and evaluated on the August 2015 event (`DFO_4288`, 172,199 cells), the Neural Network achieved **80.4% Recall** and an ROC-AUC of **0.672**, demonstrating solid temporal transfer across monsoonal pulses.

#### Q14: What is the Brier Score, and what do your numbers indicate?
**Answer:** The Brier score measures the mean squared difference between predicted probabilities and actual binary outcomes ($\frac{1}{N}\sum(p_i - y_i)^2$). Lower scores indicate better probability calibration. Random Forest achieved a Brier score of **0.073**, indicating well-calibrated probabilities.

#### Q15: What is the purpose of the Mamdani Fuzzy Layer (Stretch Goal B)?
**Answer:** The Mamdani fuzzy system maps neural network probabilities and 3-day rainfall into linguistic risk tiers (`Low`, `Moderate`, `High`) using 8 interpretable rules. It is designed for transparent human decision-support, not for claiming superior numerical accuracy.

#### Q16: How does Sentinel-1 SAR detect flooding under cloud cover (Stretch Goal A)?
**Answer:** Sentinel-1 uses C-band microwaves (5.4 GHz) that penetrate cloud cover. Smooth standing floodwater causes specular reflection away from the satellite, reducing backscatter by 8 to 12 dB compared to dry rough soil. Automatic Otsu thresholding identifies this bimodal drop, achieving an **IoU of 0.991** against the reference mask.

#### Q17: Where do the models fail most frequently based on your error analysis?
**Answer:** In very flat alluvial terrain (<0.5° slope, 65–85m elevation), where micro-topographic hollows and agricultural bunds cause localized ponding not captured at 250m DEM resolution. In contrast, on slopes >2°, error rates dropped to ~5.3% with zero false negatives.

#### Q18: What are the three primary limitations of this prototype?
**Answer:**
1. **250m Resolution:** Cannot capture localized embankments, ditches, or culverts.
2. **Optical Cloud Obscuration:** Peak flood peaks may be masked by clouds in MODIS composites.
3. **Absence of 2D Hydrodynamic Physics:** Fluid momentum and conservation of mass are not explicitly modeled.
