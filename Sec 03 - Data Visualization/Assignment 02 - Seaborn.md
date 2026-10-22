# Assignment 02: Statistical Visualization with Seaborn

**Course:** AI & Machine Learning Foundations  
**Section:** 03 - Data Visualization  
**Reference Notebooks:** `02 - Seaborn.ipynb` & `03 - Seaborn Pandas Viz.ipynb`  
**Estimated Time to Complete:** ~30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_02_seaborn_<your_name>.ipynb`) or Python script (`assignment_02_seaborn_<your_name>.py`).

---

## Overview & Objectives

Seaborn provides an elegant, high-level interface built on top of Matplotlib, specifically designed for statistical data exploration. It excels at mapping data frames directly to visual aesthetics (color hue, marker styles, sizes) and rendering complex multi-view grids with minimal boilerplate.

By completing this assignment, you will demonstrate your ability to:
1. Distinguish between and effectively use **Axes-level** functions (e.g., `scatterplot`, `boxplot`) and **Figure-level** functions (e.g., `relplot`, `catplot`, `displot`).
2. Map multidimensional attributes to visual semantics (`hue`, `style`, `size`).
3. Diagnose univariate and bivariate statistical distributions using KDEs, empirical CDFs, and joint plots.
4. Uncover categorical variance with box, violin, and strip plots.
5. Create annotated correlation heatmaps and customized `FacetGrid` visualizations.

---

## Setup

Run the following cell to initialize the libraries, set the global Seaborn theme, and load the primary dataset:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Apply modern aesthetic theme
sns.set_theme(style="whitegrid", palette="muted")

# Load the classic Palmer Penguins dataset
penguins = sns.load_dataset("penguins")
# Drop rows with missing measurements for clean plotting
penguins = penguins.dropna()
```

---

## Task 1: Relational Visualization & Semantic Dimensions (~7 mins)
*Topics: `sns.scatterplot`, `sns.relplot`, semantic mappings (`hue`, `style`, `size`) (`Part 4 & 5`)*

In exploratory data analysis for classification tasks, you need to understand feature separability between classes.

### Instructions:
1. Using `sns.scatterplot`:
   - Plot `bill_length_mm` on the x-axis and `flipper_length_mm` on the y-axis.
   - Map `species` to `hue` (color).
   - Map `sex` to `style` (marker style).
   - Map `body_mass_g` to `size` (marker size, with `sizes=(40, 200)`).
   - Set `alpha=0.8`.
2. Add a clear title: `'Flipper Length vs Bill Length across Penguin Species'`.
3. Move the legend outside the plot or adjust its placement so it does not occlude the data points.
4. **Analysis Question (1-2 sentences in a markdown cell):** Based on the plot, which single measurement best separates the *Gentoo* species from *Adelie* and *Chinstrap*?

---

## Task 2: Distribution Diagnostics & Joint Analysis (~8 mins)
*Topics: `sns.displot`, `sns.histplot`, `sns.kdeplot`, `sns.jointplot` (`Part 3 & 6`)*

Understanding feature distributions is critical before standardizing inputs or selecting machine learning algorithms.

### Instructions:
1. **Bimodal Distribution Analysis:**
   - Use the figure-level function `sns.displot` to visualize `body_mass_g`.
   - Set `kde=True` to overlay the Kernel Density Estimate.
   - Use `hue='species'` to see the distribution per species.
   - Experiment with `col='sex'` to produce side-by-side distribution panels for Female vs Male penguins.
2. **Bivariate Joint Relationship:**
   - Create a `sns.jointplot`:
     - X-axis: `bill_depth_mm`, Y-axis: `bill_length_mm`.
     - Data: `penguins`.
     - Map `species` to `hue`.
     - Set `kind='scatter'`.
3. Observe the marginal distribution plots on the top and right edges. Notice how Simpson's paradox is visually apparent when conditioning on `species`.

---

## Task 3: Categorical Distribution & Outlier Detection (~8 mins)
*Topics: `sns.boxplot`, `sns.violinplot`, `sns.stripplot`, `sns.catplot` (`Part 7, 8 & 9`)*

Outliers and distributional skew can degrade machine learning model performance.

### Instructions:
1. Create a figure with a single axis (`fig, ax = plt.subplots(figsize=(9, 5))`).
2. Create a combined categorical visualization:
   - Draw a `sns.boxplot` comparing `body_mass_g` across `species` with `color='lightgray'` and `width=0.5`, suppressing outliers with `showfliers=False`.
   - Overlay a `sns.stripplot` on the same axis (`data=penguins`, `x='species'`, `y='body_mass_g'`, `hue='sex'`, `dodge=True`, `alpha=0.6`, `jitter=0.2`).
3. Set the title to `'Body Mass Distribution by Species and Sex'`.
4. Format the y-axis label to `'Body Mass (grams)'` and x-axis to `'Species'`.
5. Ensure the legend accurately reflects the `sex` categories.

---

## Task 4: Aggregation Heatmap & FacetGrid (~7 mins)
*Topics: `sns.heatmap`, Pivot Tables, `sns.FacetGrid` (`Part 12 & 13`, `03 - Seaborn Pandas Viz`)*

Correlation analysis and multi-panel faceted views allow fast pattern recognition across tabular datasets.

### Instructions:
1. **Correlation Heatmap:**
   - Compute the correlation matrix for the numeric features of `penguins`:
     ```python
     num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
     corr = penguins[num_cols].corr()
     ```
   - Plot `sns.heatmap` with `annot=True`, `cmap='vlag'`, `vmin=-1`, `vmax=1`, `fmt='.2f'`, and `linewidths=0.5`.
   - Title the heatmap: `'Penguin Morphological Feature Correlations'`.
2. **FacetGrid Histogram Mapping:**
   - Initialize a `sns.FacetGrid` conditioned on `island` in columns (`col='island'`) and `species` in rows (`row='species'`), using `penguins`.
   - Map `sns.histplot` to the grid for the column `'flipper_length_mm'` with `bins=10`.
   - Inspect the resulting grid: On which island(s) does the *Gentoo* species live?

---

## Submission Checklist

- [ ] All code cells execute sequentially without warnings or errors.
- [ ] Both Axes-level functions and Figure-level functions are correctly demonstrated.
- [ ] All plots have descriptive titles, axis labels, and legible legends.
- [ ] The written observations in Task 1 and Task 4 are answered concisely in markdown cells.
- [ ] Save and submit `assignment_02_seaborn_<your_name>.ipynb`.
