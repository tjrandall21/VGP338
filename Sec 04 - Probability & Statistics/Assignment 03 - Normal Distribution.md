# Assignment 03: The Normal Distribution, Z-Scores & Probabilistic Modeling

**Course:** AI & Machine Learning Foundations  
**Section:** 04 - Probability & Statistics  
**Reference Notebook:** `03 - Normal Distribution.ipynb`  
**Estimated Time to Complete:** 20 – 30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_03_normal_distribution_<your_name>.ipynb`) or Python script (`assignment_03_normal_distribution_<your_name>.py`).

---

## Overview & Objectives

The **Normal (Gaussian) Distribution** is the cornerstone of probability and statistics in machine learning. From the Central Limit Theorem and weight initialization in deep neural networks to standardizing input features ($Z$-score scaling) and anomaly detection, the bell curve appears almost everywhere in applied AI.

By completing this assignment, you will demonstrate your ability to:
1. Validate the **Empirical Rule (68-95-99.7% Rule)** using simulated data in NumPy.
2. Calculate and interpret **Z-scores** to standardize variables onto the standard normal distribution $\mathcal{N}(0, 1)$.
3. Use `scipy.stats.norm` functions—specifically the **Probability Density Function (PDF)**, **Cumulative Distribution Function (CDF)**, and **Percent Point Function (PPF)**—to answer real-world probabilistic engineering questions.
4. Visualize normal probability distributions and shade specific rejection / warranty regions.

---

## Setup & Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Consistent styling
plt.style.use('default')
np.random.seed(42)
```

---

## Task 1: Empirical Rule Simulation (68–95–99.7% Rule) (~7 mins)
*Reference: Part 2 & Part 4 in `03 - Normal Distribution.ipynb`*

For any normally distributed variable with mean $\mu$ and standard deviation $\sigma$:
- Approximately **68.27%** of data lies within $[\mu - \sigma, \; \mu + \sigma]$
- Approximately **95.45%** of data lies within $[\mu - 2\sigma, \; \mu + 2\sigma]$
- Approximately **99.73%** of data lies within $[\mu - 3\sigma, \; \mu + 3\sigma]$

### Instructions:
1. Simulate $N = 20,000$ battery lifespans (in hours) for commercial camera drones from a normal distribution with:
   $$\mu = 120 \text{ hours}, \quad \sigma = 12 \text{ hours}$$
   ```python
   mu = 120
   sigma = 12
   battery_lifespans = np.random.normal(loc=mu, scale=sigma, size=20000)
   ```
2. Using boolean indexing in NumPy, compute the empirical percentage of samples that fall within:
   - 1 Standard Deviation: $[\mu - \sigma, \mu + \sigma]$
   - 2 Standard Deviations: $[\mu - 2\sigma, \mu + 2\sigma]$
   - 3 Standard Deviations: $[\mu - 3\sigma, \mu + 3\sigma]$
3. Print a formatted comparison between your **Empirical Percentages** and the theoretical **68.27% / 95.45% / 99.73%** values.

---

## Task 2: Standardizing Features with Z-Scores (~7 mins)
*Reference: Part 3 in `03 - Normal Distribution.ipynb`*

The **Z-score** measures how many standard deviations an observation $x$ is from the mean:
$$Z = \frac{x - \mu}{\sigma}$$
Transforming raw features to Z-scores scales any $\mathcal{N}(\mu, \sigma)$ distribution to the **Standard Normal Distribution** $\mathcal{N}(0, 1)$.

### Scenario:
Two candidate drone motor models from different manufacturers were tested under different conditions:
- **Motor A:** Mean speed $\mu_A = 8500\text{ RPM}$, standard deviation $\sigma_A = 250\text{ RPM}$. A test unit clocked **8950 RPM**.
- **Motor B:** Mean speed $\mu_B = 12000\text{ RPM}$, standard deviation $\sigma_B = 600\text{ RPM}$. A test unit clocked **12900 RPM**.

### Instructions:
1. Calculate the Z-scores for both test units using the formula $Z = \frac{x - \mu}{\sigma}$.
2. Print both Z-scores formatted to 2 decimal places.
3. State which motor performed relatively better compared to its peer group, and explain why comparing raw RPM differences directly would be misleading.
4. Using NumPy, take your `battery_lifespans` array from Task 1, standardize all 20,000 values into an array `z_scores`, and print `np.mean(z_scores)` and `np.std(z_scores)` to confirm they equal approximately `0.0` and `1.0`.

---

## Task 3: Probabilistic Queries with `scipy.stats.norm` (~8 mins)
*Reference: Part 4 & Part 5 in `03 - Normal Distribution.ipynb`*

Using drone battery distribution $\mathcal{N}(\mu = 120, \sigma = 12)$:

```python
dist = stats.norm(loc=mu, scale=sigma)
```

Answer the following three engineering queries using `dist.cdf()` and `dist.ppf()`:

1. **Warranty Failure Risk:**  
   The manufacturer offers a free replacement warranty if a battery lasts **less than 95 hours**. What percentage of batteries will the company have to replace under warranty?  
   $$\text{Formula: } P(X < 95) = \text{dist.cdf}(95)$$
2. **Acceptable Range:**  
   What is the probability that a randomly chosen battery lasts **between 110 and 130 hours**?  
   $$\text{Formula: } P(110 \le X \le 130) = \text{dist.cdf}(130) - \text{dist.cdf}(110)$$
3. **Premium Fleet Cutoff (Percent Point Function):**  
   The enterprise division wants to select the **top 5% longest-lasting batteries** (the 95th percentile) for emergency medical delivery drones. What is the minimum lifespan (in hours) a battery must achieve to qualify?  
   $$\text{Formula: } \text{cutoff} = \text{dist.ppf}(0.95)$$
4. Print all answers clearly with percentage or hour units.

---

## Task 4: Visualizing the PDF & Shading Probability Regions (~8 mins)
*Reference: Visualizing PDF and CDF in `03 - Normal Distribution.ipynb`*

### Instructions:
1. Generate an array of 1,000 evenly spaced x-values spanning $\mu \pm 4\sigma$:
   ```python
   x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
   y = dist.pdf(x)
   ```
2. Create a figure: `fig, ax = plt.subplots(figsize=(10, 5))`.
3. Plot the continuous bell curve: `ax.plot(x, y, 'b-', linewidth=2.5, label='Battery Lifespan PDF')`.
4. **Shade the Warranty Region ($X < 95$ hours):**
   - Create a sub-array `x_warranty = x[x <= 95]` and `y_warranty = dist.pdf(x_warranty)`.
   - Shade the area: `ax.fill_between(x_warranty, y_warranty, color='crimson', alpha=0.5, label='Warranty (< 95 hrs)')`.
5. **Mark the Top 5% Cutoff:**
   - Draw a vertical dashed line at the 95th percentile cutoff value found in Task 3: `ax.axvline(top_5_cutoff, color='darkgreen', linestyle='--', linewidth=2, label=f'Top 5% Cutoff ({top_5_cutoff:.1f} hrs)')`.
6. Add labels, a title (`'Drone Battery Lifespan Distribution & Quality Thresholds'`), gridlines, and a legend.

---

## Submission Checklist

- [ ] Task 1: Simulated normal data and verified the 68–95–99.7% empirical proportions.
- [ ] Task 2: Calculated Z-scores, compared relative performance, and verified standardization properties ($\mu=0, \sigma=1$).
- [ ] Task 3: Computed tail probabilities and percentiles using `stats.norm.cdf` and `stats.norm.ppf`.
- [ ] Task 4: Plotted the normal PDF and shaded the warranty region with annotated thresholds.
- [ ] Script runs sequentially without issues.
