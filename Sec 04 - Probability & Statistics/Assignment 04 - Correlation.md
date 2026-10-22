# Assignment 04: Correlation Analysis, Heatmaps & Causation Pitfalls

**Course:** AI & Machine Learning Foundations  
**Section:** 04 - Probability & Statistics  
**Reference Notebook:** `04 - Correlation.ipynb`  
**Estimated Time to Complete:** 20 – 30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_04_correlation_<your_name>.ipynb`) or Python script (`assignment_04_correlation_<your_name>.py`).

---

## Overview & Objectives

In machine learning, understanding relationships between pairs of variables is foundational to **Exploratory Data Analysis (EDA)** and **Feature Selection**. If two predictor variables are heavily correlated with each other (multicollinearity), linear models can become unstable. Conversely, if a feature has no correlation with the target variable, it might be pruned—unless that relationship is non-linear!

Moreover, confusing correlation with causation is one of the most common and costly mistakes in data science.

By completing this assignment, you will demonstrate your ability to:
1. Compute and formally report **Pearson's Correlation Coefficient ($r$)** and its associated **p-value** using `scipy.stats.pearsonr`.
2. Construct and interpret a multi-feature **Correlation Matrix & Heatmap** using Pandas and Seaborn.
3. Demonstrate the limitations of Pearson's $r$ on **non-linear datasets** (where $r \approx 0$ despite deterministic dependence).
4. Identify **confounding variables** in observational data and articulate how to test for genuine causation using randomized experimental design.

---

## Setup & Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Set clean aesthetic style
plt.style.use('default')
sns.set_theme(style='whitegrid')
np.random.seed(42)
```

---

## Task 1: Computing & Reporting Pearson's Correlation (~7 mins)
*Reference: Part 2 & Part 3 in `04 - Correlation.ipynb`*

Pearson's $r$ quantifies the linear relationship between two continuous variables:
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- Range: $-1 \le r \le 1$
- Rule of thumb: $|r| \ge 0.7$ (strong), $0.3 \le |r| < 0.7$ (moderate), $|r| < 0.3$ (weak).

### Instructions:
1. Load the built-in Seaborn dataset `tips`:
   ```python
   tips = sns.load_dataset('tips')
   ```
2. Extract the continuous columns `total_bill` and `tip`.
3. Compute Pearson's correlation coefficient and p-value using `pearsonr`:
   ```python
   r, p_value = pearsonr(tips['total_bill'], tips['tip'])
   ```
4. Print the sample size $n = \text{len}(tips)$, the coefficient $r$, and the p-value.
5. Create a scatter plot with an overlaid regression trendline using `sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'alpha': 0.6})`.
6. **Formal Reporting:** Write a formal 1-sentence finding matching the scientific reporting format taught in the notebook:  
   *“There was a [strong/moderate/weak], [positive/negative] correlation between total bill and tip amount ($r = \dots$, $p = \dots$, $n = \dots$).”*

---

## Task 2: Correlation Matrix & Heatmap for Feature Selection (~8 mins)
*Reference: Part 3 in `04 - Correlation.ipynb`*

### Instructions:
1. Select the numerical features from `tips`:
   ```python
   num_features = tips[['total_bill', 'tip', 'size']]
   ```
2. Compute the Pearson correlation matrix using `.corr()`:
   ```python
   corr_matrix = num_features.corr()
   print(corr_matrix)
   ```
3. Create a heatmap visualization using `sns.heatmap()`:
   - Use `annot=True` to print the correlation coefficients inside each cell formatted to 2 decimal places (`fmt='.2f'`).
   - Use a diverging colormap like `'coolwarm'`, with `vmin=-1`, `vmax=1`, and `center=0`.
   - Add a meaningful title: `'Numerical Feature Correlation Heatmap'`.
4. **Analysis Questions:**
   - Which pair of features exhibits the highest positive correlation?
   - In a linear regression predicting `tip`, does party `size` provide redundant information to `total_bill`, or could both be useful? Briefly explain.

---

## Task 3: The Non-Linear Trap — When Pearson Fails (~8 mins)
*Reference: Part 2 & Part 3 in `04 - Correlation.ipynb`*

Pearson's $r$ only measures **linear** relationships. If the relationship is non-linear (e.g., quadratic, sinusoidal, exponential), Pearson's $r$ can be close to zero even when one variable is perfectly predictable from the other.

### Instructions:
1. Generate synthetic data exhibiting a deterministic parabolic relationship:
   ```python
   x = np.linspace(-10, 10, 300)
   # Parabolic relationship with minimal noise
   y = x**2 + np.random.normal(0, 3, size=len(x))
   ```
2. Calculate Pearson's $r$ and p-value between `x` and `y`.
3. Print the resulting correlation coefficient ($r$).
4. Plot a scatter plot of `x` vs `y` using `plt.subplots(figsize=(7, 4))`. Set title to `f'Non-Linear Relationship (Pearson r = {r:.3f})'`.
5. **Critical Thinking Question:**  
   *The data has an obvious, near-perfect relationship ($y \approx x^2$), yet Pearson's $r \approx 0$. Why does Pearson's coefficient give this misleading result? What should an ML engineer do when exploring features that might have non-linear dependencies?*

---

## Task 4: Correlation vs. Causation & Confounding Variables (~7 mins)
*Reference: Part 4 in `04 - Correlation.ipynb`*

### Scenario:
A data science team at an online retail company trains a churn prediction model. During exploratory data analysis, they discover a strong positive correlation ($r = +0.76$) between **the number of support tickets filed by a customer** and **annual customer lifetime spending**. 

A junior product manager reviews this finding and suggests:  
> *"Filing support tickets clearly causes customers to spend more money! We should deliberately introduce bugs and remove self-service FAQs so more customers have to call our support agents, thereby increasing total company revenue."*

### Questions to Answer:
1. **Identify the Confounding (Lurking) Variable:**  
   What underlying third variable actually explains why high spenders submit more support tickets? (Hint: Think about transaction frequency and product usage).
2. **Experimental Design for True Causation:**  
   To prove whether superior customer support *causes* higher repeat spending, how would you design a **Randomized Controlled Experiment (A/B Test)**? Specifically specify:
   - What the **Control Group** receives.
   - What the **Treatment Group** receives.
   - How subjects are assigned to prevent confounding bias.

---

## Submission Checklist

- [ ] Task 1: Calculated $r$ and p-value on `tips` and wrote a formal scientific finding sentence.
- [ ] Task 2: Constructed a labeled Seaborn correlation heatmap with cell annotations.
- [ ] Task 3: Demonstrated that Pearson's $r \approx 0$ for a quadratic dataset and explained why.
- [ ] Task 4: Identified the lurking variable in the business scenario and proposed a randomized A/B test.
- [ ] All code cells execute sequentially and plots render properly.
