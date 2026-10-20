# GeoFlood AI — Generalisation Evaluation & Boundary Analysis

**Document Status:** Research-Grade Evaluation Report  
**Date:** October 20, 2026  
**Focus:** Spatial Independence, Boundary Autocorrelation & Cross-Event Temporal Transferability

---

## 1. Overview of Generalisation Protocols

Evaluating machine learning models on geospatial data requires special care because nearby geographic cells exhibit **spatial autocorrelation** (Tobler's First Law of Geography). Standard random K-fold cross-validation randomly assigns adjacent cells into training and testing sets, resulting in severe data leakage and artificially inflated performance.

To establish defensible scientific boundaries, GeoFlood AI was evaluated across three distinct protocols:

1. **Protocol 1: Spatial Block Holdout (Primary Evaluation)**
   - Partitions the study area into a $3 \times 4$ grid of 12 contiguous spatial blocks ($\sim 33\text{ km} \times 35\text{ km}$ per block).
   - Training: Blocks `[0, 1, 3, 4, 6, 8, 9]` ($92,471$ valid land cells).
   - Validation: Blocks `[5, 11]` ($39,979$ valid land cells).
   - Testing: Blocks `[2, 7, 10]` ($39,749$ valid land cells).
   - Target Event: Primary June 2015 Assam Monsoon (DFO_4259).

2. **Protocol 2: Spatially Buffered Holdout (Boundary Autocorrelation Audit)**
   - Eliminates potential label bleeding at block boundaries by applying a $k = 4$ cell boundary buffer ($\sim 1\text{ km}$ spatial exclusion) around the interior perimeter of the test blocks.
   - Retains only core interior cells of Blocks 2, 7, and 10 ($35,815$ cells; $3,934$ border cells excluded).

3. **Protocol 3: Cross-Event Temporal Holdout (Inter-Event Transferability)**
   - Trains models on the June 2015 flood event (DFO_4259) and evaluates without modification on an independent, future monsoonal inundation event in August 2015 (DFO_4288).
   - Test Size: $172,199$ valid land cells across the entire study region ($21,322$ flooded cells; prevalence = $12.38\%$).

---

## 2. Empirical Benchmark Matrix

The following table reports the exact empirical metrics measured across all three protocols from the reproducible validation run (`reports/phase3_metrics.csv`):

| Evaluation Protocol | Model Architecture | Sample Count | Positive Prevalence | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Specificity | False Negative Rate | Brier Score | Climatology Baseline Brier | Brier Skill Score (BSS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spatial Block** | **Random Forest** | 39,749 | 9.35% | **0.901** | **0.507** | **81.2%** | **34.1%** | **0.481** | 83.8% | **18.8%** | **0.0728** | 0.0848 | **+0.141** |
| **Spatial Block** | **Neural Network (Uncal.)** | 39,749 | 9.35% | 0.805 | 0.448 | 60.9% | 22.0% | 0.323 | 77.7% | 39.1% | 0.1651 | 0.0848 | -0.947 |
| **Spatial Block** | **Neural Network (Platt Cal.)** | 39,749 | 9.35% | 0.805 | 0.448 | 32.4% | 57.5% | 0.415 | 97.5% | 67.6% | **0.0695** | 0.0848 | **+0.180** |
| **Spatial Block** | **Logistic Regression** | 39,749 | 9.35% | 0.684 | 0.320 | 53.5% | 31.7% | 0.398 | 88.1% | 46.5% | 0.1313 | 0.0848 | -0.549 |
| **Buffered (1km)** | **Random Forest** | 35,815 | 8.84% | **0.908** | **0.515** | **82.7%** | **32.1%** | **0.462** | 83.0% | **17.3%** | **0.0684** | 0.0806 | **+0.151** |
| **Buffered (1km)** | **Neural Network** | 35,815 | 8.84% | 0.816 | 0.451 | 62.3% | 20.8% | 0.312 | 77.0% | 37.7% | 0.1654 | 0.0806 | -1.053 |
| **Buffered (1km)** | **Logistic Regression** | 35,815 | 8.84% | 0.690 | 0.310 | 53.9% | 31.8% | 0.400 | 88.8% | 46.1% | 0.1239 | 0.0806 | -0.538 |
| **Cross-Event** | **Logistic Regression** | 172,199 | 12.38% | **0.713** | **0.233** | **83.7%** | 15.9% | 0.268 | 37.6% | **16.3%** | 0.4815 | 0.1085 | -3.438 |
| **Cross-Event** | **Neural Network** | 172,199 | 12.38% | 0.672 | 0.172 | 80.4% | 18.3% | 0.298 | 49.1% | 19.6% | 0.2043 | 0.1085 | -0.883 |
| **Cross-Event** | **Random Forest** | 172,199 | 12.38% | 0.485 | 0.131 | 26.0% | 13.3% | 0.176 | 76.1% | 74.0% | 0.1181 | 0.1085 | -0.089 |

---

## 3. Key Scientific Findings & Analyses

### Finding A: Spatial Autocorrelation Does Not Inflate Block Holdout Metrics
A common criticism of spatial block partitioning is that cells adjacent to block boundaries might leak information due to continuous topography or river networks.
- In our audit, excluding all cells within $1\text{ km}$ ($4$ grid cells) of any training or validation boundary reduced the test set from $39,749$ to $35,815$ cells.
- **Random Forest ROC-AUC increased from 0.901 to 0.908.**
- **Neural Network ROC-AUC increased from 0.805 to 0.816.**
- **Logistic Regression ROC-AUC increased from 0.684 to 0.690.**
- *Conclusion:* Performance was **not** artificially inflated by boundary bleed. The core interior cells of the test blocks demonstrate genuine geographical generalisation across the Brahmaputra floodplain.

### Finding B: Random Forest Overfitting vs. Continuous Hyperplane Generalisation
A critical discovery emerged from the Cross-Event Holdout (Protocol 3):
- While Random Forest achieved the highest ROC-AUC on the spatial block holdout (**0.901**), its performance **collapsed to 0.485** on the unseen future event (DFO_4288, August 2015).
- In contrast, the **PyTorch Neural Network maintained an ROC-AUC of 0.672 with 80.4% Recall**, and **Logistic Regression achieved an ROC-AUC of 0.713 with 83.7% Recall**.
- *Mechanistic Explanation:*
  - Random Forest splits space using orthogonal, piecewise-constant decision cuts (e.g., $\text{rain\_event\_sum} \ge 850\text{ mm}$ AND $\text{elevation} \le 72.5\text{ m}$). These cuts fit the specific spatial rainfall distribution of the June event very tightly. When the August monsoon hit with a different spatial rainfall pattern, those rigid split points failed to generalize.
  - In contrast, the continuous linear and neural hyperplanes learned smooth monotonic trade-offs between gravitational drainage ($\text{elevation}$, $\text{slope}$) and precipitation head, preserving meaningful discrimination across independent flood episodes.

### Finding C: Probability Calibration is Critical for Minority Class Inundation
- In imbalanced spatial problems ($9.35\%$ test prevalence), uncalibrated neural networks trained with standard binary cross-entropy output overconfident probabilities for the minority class.
- The uncalibrated Neural Network had a Brier score of **0.1651**, substantially worse than the trivial baseline of predicting the historical prevalence everywhere ($0.0848$).
- Fitting **Platt Scaling** (logistic sigmoid calibration) strictly on the validation partition (Blocks 5 and 11) reduced the test Brier score to **0.0695**, achieving a **Brier Skill Score of +0.180** over climatology.

---

## 4. Scope and Limitations of the Evaluation

### What These Protocols Establish
1. **Spatial Generalisation within the Same Basin:** Models evaluated on unseen spatial blocks can predict flood propensity for unmonitored communities along the Brahmaputra floodplain during a monsoon event.
2. **Boundary Autocorrelation Resistance:** Removing boundary cells proves that the spatial holdout metrics are robust against spatial bleed.
3. **Temporal Sensitivity:** Testing on an independent future flood event exposes how different model families handle temporal distribution shifts.

### What These Protocols CANNOT Establish
1. **Inter-Basin Transferability:** These models were trained and tested exclusively on the Brahmaputra River Basin in Assam, India. They cannot be transferred to other river systems (e.g., the Ganges, Godavari, or Mississippi) without retraining, because river bathymetry, sediment dynamics, and soil infiltration capacities differ fundamentally.
2. **Real-Time Dynamic Wave Fronts:** Static susceptibility evaluates where inundation occurs over an event composite; it does not predict the hour-by-hour velocity or depth of the flood wave.
3. **Multi-Year Climate Stationarity:** Both evaluated events occurred during the 2015 monsoonal season. Long-term climate shifts or significant embankment infrastructure modifications will alter the hydrological baseline.
