# Assignment 01: Matplotlib Essentials

**Course:** AI & Machine Learning Foundations  
**Section:** 03 - Data Visualization  
**Reference Notebook:** `01 - Matplotlib.ipynb`  
**Estimated Time to Complete:** ~30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_01_matplotlib_<your_name>.ipynb`) or Python script (`assignment_01_matplotlib_<your_name>.py`).

---

## Overview & Objectives

In machine learning and AI, Matplotlib is the bedrock of Python visualization. Whether monitoring deep neural network training loss, diagnosing validation curves, or visualizing multidimensional embeddings, mastering Matplotlib's **Object-Oriented (OO) API** (`fig, ax = plt.subplots(...)`) is an essential engineering skill.

By completing this assignment, you will demonstrate your ability to:
1. Use Matplotlib's Object-Oriented interface to create modular, publication-ready figures.
2. Construct side-by-side subplots with customized twin axes, legends, gridlines, and annotations.
3. Visualize distributions and categorical comparisons using bar plots, histograms, and scatter plots with colormaps.
4. Render matrix data (such as correlation matrices) using `imshow()` with colorbars and cell annotations.

---

## Setup

Run the following cell to import libraries and prepare the environment:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set a consistent aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
np.random.seed(42)
```

---

## Task 1: Training Diagnostics & Dual-Axis Visualization (~8 mins)
*Topics: OO Interface, Subplots, Multi-line plotting, Twin Axes (`Part 2, 3 & 4`)*

In deep learning, you frequently track both **Loss** (which decreases) and **Accuracy** (which increases) across training epochs.

### Instructions:
1. Generate synthetic training progress data:
   ```python
   epochs = np.arange(1, 21)
   train_loss = 2.5 * np.exp(-0.2 * epochs) + np.random.normal(0, 0.03, size=len(epochs))
   val_loss = 2.7 * np.exp(-0.18 * epochs) + 0.15 + np.random.normal(0, 0.04, size=len(epochs))
   val_accuracy = 0.5 + 0.45 * (1 - np.exp(-0.15 * epochs)) + np.random.normal(0, 0.02, size=len(epochs))
   ```
2. Create a figure with a single Axes (`fig, ax1 = plt.subplots(figsize=(8, 5))`).
3. On `ax1`:
   - Plot `epochs` vs `train_loss` with a solid blue line (`color='tab:blue'`, `label='Train Loss'`).
   - Plot `epochs` vs `val_loss` with a dashed red line (`color='tab:red'`, `linestyle='--'`, `label='Val Loss'`).
   - Set the x-axis label to `'Epoch'` and the y-axis label to `'Loss'`.
   - Set the title to `'Model Training Dynamics'`.
4. Create a twin y-axis `ax2 = ax1.twinx()`:
   - Plot `epochs` vs `val_accuracy` with a green dash-dot line (`color='tab:green'`, `linestyle='-.', marker='o'`, `markersize=4`, `label='Val Accuracy'`).
   - Set `ax2` y-axis label to `'Accuracy'`.
   - Set the y-limits of `ax2` from `0.4` to `1.0`.
5. Combine the legends from both `ax1` and `ax2` into a single, clean legend in the upper-right corner.

---

## Task 2: Comparative Bar & Category Analysis (~7 mins)
*Topics: Bar plots, horizontal bars, custom ticks, annotations (`Part 8 & 11`)*

Model latency and throughput vary across hardware accelerators. You will create a clear comparative horizontal bar chart.

### Instructions:
1. Use the following benchmark dictionary:
   ```python
   models = ['ResNet-50', 'EfficientNet-B0', 'Vision Transformer', 'MobileNet-V3', 'ConvNeXt']
   latency_ms = [14.2, 8.5, 32.1, 4.8, 18.7]
   ```
2. Create a figure and axis (`figsize=(8, 4)`).
3. Generate a horizontal bar plot (`ax.barh`) using a cool-to-warm colormap or a soft color like `'teal'`.
4. Invert the y-axis so the fastest model or top model is at the top (`ax.invert_yaxis()`).
5. Add labels directly at the end of each bar showing the exact millisecond value formatted as `f"{val:.1f} ms"`.
6. Set appropriate title (`'Model Inference Latency (Batch Size = 1)'`), x-label (`'Latency (ms)'`), and remove unnecessary top and right spines for a modern minimalist look.

---

## Task 3: Feature Distribution & Subplot Grids (~8 mins)
*Topics: Subplots grid, Histograms, Scatter with colormap (`Part 3, 7 & 9`)*

### Instructions:
1. Load the dataset provided in the course directory:
   ```python
   # Load static dataset
   df = pd.read_json('data/df_static.json', lines=True)
   ```
2. Create a 1-row by 2-column subplot grid (`fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))`).
3. **Left Subplot (`ax1`): Age Distribution Histogram**
   - Plot a histogram of `df['Age']` with `bins=15`, `color='skyblue'`, `edgecolor='black'`, and `alpha=0.7`.
   - Add a vertical red dashed line at the **mean** age (`ax1.axvline(...)`), labeled `'Mean Age'`.
   - Set title to `'Age Distribution'` and label axes.
4. **Right Subplot (`ax2`): Attractiveness vs Height Scatter**
   - Plot `df['Height_cm']` on the x-axis and `df['Attractiveness']` on the y-axis.
   - Map point colors to `df['Income']` using the `'viridis'` colormap (`scatter(..., c=df['Income'], cmap='viridis', alpha=0.7)`).
   - Add a colorbar linked to this scatter plot with the label `'Income ($)'`.
   - Set title to `'Attractiveness vs Height'` and label axes.
5. Use `plt.tight_layout()` to ensure labels do not overlap.

---

## Task 4: Heatmap / Correlation Matrix with Text Annotations (~7 mins)
*Topics: imshow, matrix visualization, annotations (`Part 14`)*

Feature correlation analysis prevents multicollinearity in predictive modeling.

### Instructions:
1. Select numerical columns from `df`:
   ```python
   num_cols = ['Age', 'Height_cm', 'Income', 'Attractiveness']
   corr = df[num_cols].corr().values
   ```
2. Create a figure and axis (`figsize=(6, 5)`).
3. Display the correlation matrix using `ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)`.
4. Add a colorbar alongside the heatmap.
5. Set the x-ticks and y-ticks to match the column names (`ax.set_xticks(range(len(num_cols)))`, `ax.set_xticklabels(num_cols, rotation=45)`).
6. Use a nested loop to overlay the numerical correlation coefficient (formatted to 2 decimal places, e.g. `0.45` or `-0.12`) centered inside each cell using `ax.text(...)`.
7. Give the figure the title `'Numerical Feature Correlation Matrix'`.

---

## Submission Checklist

- [ ] All code runs sequentially without errors from top to bottom.
- [ ] Figures use the Object-Oriented syntax (`fig, ax = plt.subplots()`).
- [ ] Every plot includes informative titles, axis labels, and proper scale limits.
- [ ] Save and submit `assignment_01_matplotlib_<your_name>.ipynb`.
