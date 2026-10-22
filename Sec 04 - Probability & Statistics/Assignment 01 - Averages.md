# Assignment 01: Measures of Central Tendency & Data Distributions

**Course:** AI & Machine Learning Foundations  
**Section:** 04 - Probability & Statistics  
**Reference Notebook:** `01 - Averages.ipynb`  
**Estimated Time to Complete:** 20 – 30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_01_averages_<your_name>.ipynb`) or Python script (`assignment_01_averages_<your_name>.py`).

---

## Overview & Objectives

In machine learning and data science, describing datasets accurately is the first critical step before feature engineering, preprocessing, and model training. Summary statistics—specifically **Mean**, **Median**, and **Mode**—condense thousands or millions of observations into representative values. However, selecting the wrong measure can severely mislead model evaluation or feature scaling.

By completing this assignment, you will demonstrate your ability to:
1. Identify and calculate the appropriate measure of central tendency for categorical, discrete, and continuous variables.
2. Quantify and explain the sensitivity of the arithmetic mean vs. the median to extreme values (outliers).
3. Diagnose distribution skewness (symmetric, right-skewed, left-skewed) by comparing mean, median, and mode.
4. Visualize distributions with overlaid central tendency markers using Matplotlib and Seaborn.

---

## Setup & Libraries

Ensure you have your environment ready with the required libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats

# Set visual style
plt.style.use('default')
sns.set_theme(style='whitegrid', palette='husl')
np.random.seed(42)
```

---

## Task 1: Categorical & Discrete Data — Finding the Mode (~5 mins)
*Reference: Part 1 & Part 2 in `01 - Averages.ipynb`*

The **Mode** is the value that occurs with the greatest frequency. Unlike the mean or median, it is the only measure of central tendency applicable to nominal categorical data.

### Scenario:
An AI startup collects user feedback ratings on a newly deployed chatbot, along with the number of prompt iterations users needed before receiving a satisfactory answer.

### Instructions:
1. Use the following sample data:
   ```python
   feedback_ratings = [
       'Satisfied', 'Neutral', 'Satisfied', 'Very Satisfied', 'Dissatisfied',
       'Satisfied', 'Neutral', 'Very Satisfied', 'Satisfied', 'Neutral',
       'Satisfied', 'Dissatisfied', 'Satisfied', 'Very Satisfied', 'Neutral'
   ]

   prompt_iterations = np.array([1, 2, 2, 3, 1, 4, 2, 3, 2, 1, 5, 2, 3, 2, 1])
   ```
2. For `feedback_ratings`:
   - Calculate the mode and its frequency using `collections.Counter` or `pd.Series().mode()`.
   - Print the most frequent rating and the number of times it appears.
3. For `prompt_iterations`:
   - Compute the mode using `Counter` or `scipy.stats.mode()`.
   - Print the modal number of iterations.
4. **Reflection Question (in a code comment or markdown cell):**  
   *Why is it mathematically impossible to compute the arithmetic mean of `feedback_ratings` directly without assigning numeric encodings?*

---

## Task 2: Mean vs. Median & Outlier Sensitivity (~8 mins)
*Reference: Part 3 & Part 4 in `01 - Averages.ipynb`*

- **Sample Mean ($\bar{x}$):** $\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$ (incorporates every value, but is vulnerable to extreme values).
- **Median:** The middle value of a sorted distribution (resistant to extreme values).

### Scenario:
You are monitoring API response latency (in milliseconds) for a production machine learning service across 12 consecutive requests.

### Instructions:
1. Create the initial baseline response times array:
   ```python
   baseline_latency_ms = np.array([45, 52, 48, 50, 47, 53, 49, 51, 46, 54, 48, 52])
   ```
2. Calculate and print the **Mean** and **Median** for `baseline_latency_ms` using `np.mean()` and `np.median()`.
3. Simulate a sudden network timeout / database cold-start by adding a single extreme outlier (e.g., `1200 ms`) to the dataset:
   ```python
   latency_with_timeout = np.append(baseline_latency_ms, [1200])
   ```
4. Re-calculate the **Mean** and **Median** for `latency_with_timeout`.
5. Compute the percentage change for both metrics:
   $$\text{Percentage Change} = \frac{\text{New Value} - \text{Original Value}}{\text{Original Value}} \times 100$$
6. Print a formatted comparison table or printout showing:
   - Baseline Mean vs. Contaminated Mean (% change)
   - Baseline Median vs. Contaminated Median (% change)
7. **Engineering Takeaway:** In 1–2 sentences, explain which metric (mean or median) should be used in a Service Level Agreement (SLA) report for API performance, and why.

---

## Task 3: Distribution Skewness & Central Tendency Ordering (~7 mins)
*Reference: Comparing Mode, Median, and Mean in `01 - Averages.ipynb`*

The relative positions of the Mean, Median, and Mode tell us about the **skewness** (asymmetry) of a distribution:
- **Symmetric:** $\text{Mean} \approx \text{Median} \approx \text{Mode}$
- **Right-Skewed (Positive Skew):** $\text{Mode} < \text{Median} < \text{Mean}$ (the tail pulls the mean to the right)
- **Left-Skewed (Negative Skew):** $\text{Mean} < \text{Median} < \text{Mode}$ (the tail pulls the mean to the left)

### Instructions:
1. Generate a synthetic right-skewed dataset of 1,000 annual tech salaries (in thousands of dollars) using a log-normal distribution:
   ```python
   np.random.seed(42)
   # Generates positive, right-skewed salary values
   salaries = np.random.lognormal(mean=4.5, sigma=0.6, size=1000)
   ```
2. Compute:
   - The sample **Mean** (`np.mean()`)
   - The **Median** (`np.median()`)
   - The approximate **Mode** (hint: round the salaries to the nearest integer and use `Counter(np.round(salaries)).most_common(1)[0][0]`)
3. Write an `if/elif/else` statement in Python that automatically checks the order of your calculated values and prints whether the distribution is right-skewed, left-skewed, or symmetric.
4. Verify if your program correctly identifies the distribution as **Right-Skewed**.

---

## Task 4: Visualization with Annotated Central Tendency Lines (~8 mins)
*Reference: Visualizing Central Tendency in `01 - Averages.ipynb`*

### Instructions:
1. Using Matplotlib (`plt.subplots()`), create a publication-quality figure of the `salaries` dataset generated in Task 3:
   - Plot a histogram of `salaries` with `bins=40`, `density=True`, `alpha=0.6`, `color='skyblue'`, and `edgecolor='white'`.
   - Add a Kernel Density Estimate (KDE) curve using `sns.kdeplot(salaries, color='navy', linewidth=2, label='KDE')`.
2. Add three vertical dashed lines (`ax.axvline(...)`) marking:
   - **Mode:** Red dashed line (`linestyle='--'`, `color='crimson'`, `label=f'Mode: ${mode_val:.1f}k'`)
   - **Median:** Orange dash-dot line (`linestyle='-.'`, `color='darkorange'`, `label=f'Median: ${median_val:.1f}k'`)
   - **Mean:** Green solid line (`linestyle='-'`, `color='forestgreen'`, `label=f'Mean: ${mean_val:.1f}k'`)
3. Set:
   - Title: `'Salary Distribution in Tech Startups (Right-Skewed)'`
   - X-axis Label: `'Salary ($ in thousands)'`
   - Y-axis Label: `'Density'`
   - Set x-limits to `(0, 350)` for clear visual focus.
4. Add `ax.legend()` and call `plt.tight_layout()`.

---

## Submission Checklist

- [ ] Task 1: Calculated and explained the mode for categorical and discrete arrays.
- [ ] Task 2: Demonstrated the dramatic impact of a single outlier on the mean vs. median.
- [ ] Task 3: Computed mean, median, and mode for a skewed distribution and verified their relative ordering.
- [ ] Task 4: Created a clean, labeled plot displaying the histogram, KDE, and all 3 central tendency markers.
- [ ] Code is organized, runnable, and documented with clear output messages.
