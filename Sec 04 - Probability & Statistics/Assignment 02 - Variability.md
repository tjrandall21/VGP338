# Assignment 02: Measures of Variability & Outlier Detection

**Course:** AI & Machine Learning Foundations  
**Section:** 04 - Probability & Statistics  
**Reference Notebook:** `02 - Variability.ipynb`  
**Estimated Time to Complete:** 20 – 30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_02_variability_<your_name>.ipynb`) or Python script (`assignment_02_variability_<your_name>.py`).

---

## Overview & Objectives

Central tendency tells only half the story. Two machine learning models might have the exact same average response time of 50 ms, but one model might fluctuate between 48 ms and 52 ms, while the other oscillates between 5 ms and 200 ms. 

**Variability** measures the degree of dispersion, spread, or scattering of data points around their center. Understanding variance, standard deviation, and the interquartile range (IQR) is vital for data preprocessing, feature normalization, and anomaly detection.

By completing this assignment, you will demonstrate your ability to:
1. Compute the **Range**, **Variance**, and **Standard Deviation** using NumPy and Python.
2. Differentiate mathematically and programmatically between **Population Variance** ($N$) and **Sample Variance** ($n-1$, Bessel's correction with `ddof=1`).
3. Calculate quartiles ($Q_1, Q_2, Q_3$) and the **Interquartile Range (IQR)** to detect anomalies using Tukey’s $1.5 \times \text{IQR}$ rule.
4. Compare the robustness of standard deviation vs. IQR against outliers and visualize the findings using box plots.

---

## Setup & Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent aesthetics
plt.style.use('default')
sns.set_theme(style='whitegrid', palette='husl')
np.random.seed(42)
```

---

## Task 1: Range & Step-by-Step Variance (Bessel's Correction) (~8 mins)
*Reference: Part 2 & Part 3 in `02 - Variability.ipynb`*

Recall the formulas:
- **Range:** $\text{Max} - \text{Min}$
- **Population Variance ($\sigma^2$):** $\sigma^2 = \frac{\sum_{i=1}^{N}(x_i - \mu)^2}{N}$ (`ddof=0`)
- **Sample Variance ($s^2$):** $s^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n - 1}$ (`ddof=1`)

### Scenario:
A robotic arm in an automated assembly line has its positioning accuracy tested across 10 trials. The positional errors (in micrometers) are:
```python
errors_um = np.array([12, 15, 14, 11, 16, 13, 15, 12, 14, 18])
```

### Instructions:
1. Calculate the **Range** using `np.max() - np.min()` or `np.ptp(errors_um)`.
2. Compute the **Sample Variance ($s^2$) step-by-step** without calling `np.var`:
   - Step a: Compute the sample mean $\bar{x}$.
   - Step b: Compute the deviations $(x_i - \bar{x})$.
   - Step c: Square each deviation $(x_i - \bar{x})^2$.
   - Step d: Sum the squared deviations $\sum (x_i - \bar{x})^2$.
   - Step e: Divide by $n - 1$ where $n$ is the length of the array.
3. Verify your manual result by calling `np.var(errors_um, ddof=1)`.
4. Also compute the **Population Variance** using `np.var(errors_um, ddof=0)`.
5. **Conceptual Question:** *Why does sample variance use $n - 1$ in the denominator instead of $n$? (What does Bessel's correction compensate for?)*

---

## Task 2: Standard Deviation & The Empirical Spread (~6 mins)
*Reference: Part 4 in `02 - Variability.ipynb`*

The standard deviation is the square root of the variance ($s = \sqrt{s^2}$). Its primary advantage over variance is that it is expressed in the **same physical units** as the original data.

### Instructions:
1. Calculate the sample **Standard Deviation** ($s$) of `errors_um` using `np.std(errors_um, ddof=1)`.
2. Calculate the sample mean $\bar{x}$.
3. Define the 1-standard-deviation boundary: $[\bar{x} - s, \; \bar{x} + s]$.
4. Write a brief NumPy boolean condition to calculate what percentage of values in `errors_um` fall within 1 standard deviation of the mean:
   ```python
   within_1_sd = np.mean((errors_um >= (mean - std)) & (errors_um <= (mean + std))) * 100
   print(f"Percentage of points within 1 SD: {within_1_sd:.1f}%")
   ```
5. Compare this observed percentage to the theoretical ~68% expectation for mound-shaped distributions.

---

## Task 3: Interquartile Range (IQR) & Anomaly Detection (~8 mins)
*Reference: Part 5 in `02 - Variability.ipynb`*

Quartiles divide an ordered dataset into four quarters:
- $Q_1$ (25th percentile)
- $Q_2$ (50th percentile / Median)
- $Q_3$ (75th percentile)
- $\text{IQR} = Q_3 - Q_1$
- **Tukey's Fences for Outliers:**
  - $\text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}$
  - $\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$

### Instructions:
1. Consider the following dataset representing page load times (in milliseconds) from 20 website visitors:
   ```python
   load_times = np.array([
       180, 195, 205, 210, 215, 220, 222, 225, 228, 230,
       235, 238, 240, 245, 250, 255, 260, 275, 450, 520
   ])
   ```
2. Using `np.percentile()`, calculate $Q_1$, $Q_2$, and $Q_3$.
3. Compute the $\text{IQR} = Q_3 - Q_1$.
4. Calculate the **Lower Fence** and **Upper Fence**.
5. Using NumPy boolean masking, identify and print:
   - Any values in `load_times` that are less than the Lower Fence.
   - Any values in `load_times` that are greater than the Upper Fence.
6. Print the detected anomalies and their corresponding array indices.

---

## Task 4: Robustness Comparison & Visualizing with Box Plots (~8 mins)
*Reference: IQR vs Standard Deviation in `02 - Variability.ipynb`*

### Instructions:
1. Create two versions of a synthetic student scores dataset:
   ```python
   clean_scores = np.array([72, 75, 78, 80, 82, 83, 85, 87, 88, 90, 92, 95])
   # Contaminated with an input entry typo: a score of 10 and a score of 250
   corrupted_scores = np.append(clean_scores, [10, 250])
   ```
2. Write a helper function `summarize_spread(data)` that returns a dictionary containing:
   - Range (`np.ptp(data)`)
   - Sample Standard Deviation (`np.std(data, ddof=1)`)
   - IQR (`np.percentile(data, 75) - np.percentile(data, 25)`)
3. Compute `summarize_spread()` on both `clean_scores` and `corrupted_scores`. Print the percentage increase in each metric caused by the two outliers.
4. Using Matplotlib and Seaborn, construct side-by-side boxplots:
   - Create a figure with 1 row and 2 columns: `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))`
   - In `ax1`, plot `sns.boxplot(y=clean_scores, ax=ax1, color='lightblue')` with title `'Clean Scores'`.
   - In `ax2`, plot `sns.boxplot(y=corrupted_scores, ax=ax2, color='lightcoral')` with title `'Corrupted Scores (with Outliers)'`.
   - Set y-limits on both to `(0, 270)` for fair comparison.
5. In 1–2 sentences, explain why the box height (IQR) remains stable while the whiskers and standard deviation stretch drastically.

---

## Submission Checklist

- [ ] Task 1: Calculated range and manually computed sample variance with $n-1$, confirming `np.var(..., ddof=1)`.
- [ ] Task 2: Computed standard deviation and measured the fraction of data within 1 standard deviation.
- [ ] Task 3: Calculated $Q_1$, $Q_3$, IQR, and successfully identified outliers using Tukey's fences.
- [ ] Task 4: Compared the outlier sensitivity of Range, SD, and IQR, and generated side-by-side boxplots.
- [ ] Code runs cleanly without errors.
