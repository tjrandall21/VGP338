# Assignment 04: Interactive Visualizations with Plotly

**Course:** AI & Machine Learning Foundations  
**Section:** 03 - Data Visualization  
**Reference Notebook:** `05 - Plotly.ipynb`  
**Estimated Time to Complete:** ~30 Minutes  
**Deliverables:** Submit a completed Jupyter Notebook (`assignment_04_plotly_<your_name>.ipynb`) or Python script (`assignment_04_plotly_<your_name>.py`).

---

## Overview & Objectives

Plotly enables data scientists and AI engineers to build interactive, web-ready visualizations with pan, zoom, hover tooltips, and 3D rotations out of the box. Plotly offers two distinct APIs:
1. **Plotly Express (`px`)**: A high-level, concise API for rapid data exploration.
2. **Graph Objects (`go`)**: A low-level, object-oriented API that provides granular control over individual traces, subplots, layouts, and geographic projections.

By completing this assignment, you will demonstrate your ability to:
1. Generate interactive 2D and 3D multivariate charts with custom hover templates using Plotly Express.
2. Create statistical diagnostic plots (box plots, interactive correlation heatmaps).
3. Architect multi-panel dashboards using `make_subplots` and `plotly.graph_objects`.
4. Render interactive global choropleth maps displaying country-level metrics.

---

## Setup

Run the following cell to import Plotly modules, Pandas, and load the World Happiness Report 2024 dataset:

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/WHR_2024.csv')

# Verify the shape and columns
print(df.shape)
df.head(3)
```

---

## Task 1: Multivariate 3D Exploration with Plotly Express (~8 mins)
*Topics: `px.scatter_3d`, 3D visual encodings, hover data (`Part 4`)*

High-dimensional data in AI often benefits from 3D projections before dimensionality reduction (like PCA or t-SNE).

### Instructions:
1. Using `px.scatter_3d`:
   - Data: `df`.
   - X-axis: `'GDP_per_Capita'`.
   - Y-axis: `'Healthy_Life_Expectancy'`.
   - Z-axis: `'Happiness_Score'`.
   - Color: `'Continent'`.
   - Size: `'Freedom'` (with size limits appropriate for clear visibility).
   - Hover Name: `'Country'`.
   - Additional Hover Data: `['Rank', 'Social_Support']`.
2. Customize the layout:
   - Set the title to `'3D Landscape: Happiness vs Economy & Health (2024)'`.
   - Set a clean template (e.g., `template='plotly_white'`).
   - Adjust marker opacity to `0.85`.
3. Display the interactive figure (`fig.show()`). Rotate and inspect the 3D space.

---

## Task 2: Statistical Heatmaps & Distribution with Express (~7 mins)
*Topics: `px.imshow`, `px.box`, statistical diagnostics (`Part 2 & 3`)*

Analyze the distribution of happiness scores and collinearity between explanatory variables.

### Instructions:
1. **Categorical Box Plot with Jittered Points:**
   - Create a box plot using `px.box`:
     - X-axis: `'Continent'`.
     - Y-axis: `'Happiness_Score'`.
     - Color: `'Continent'`.
     - Enable all underlying points with `points='all'`.
     - Add hover data: `['Country', 'Rank']`.
     - Title: `'Happiness Score Distribution by Continent'`.
2. **Interactive Correlation Matrix (`px.imshow`):**
   - Select key numerical features:
     ```python
     features = ['Happiness_Score', 'GDP_per_Capita', 'Social_Support', 
                 'Healthy_Life_Expectancy', 'Freedom', 'Generosity', 'Trust_in_Government']
     corr_matrix = df[features].corr()
     ```
   - Render the correlation heatmap using `px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)`.
   - Title: `'Interactive Correlation Matrix'`.

---

## Task 3: Dashboards with Graph Objects & Subplots (~8 mins)
*Topics: `plotly.graph_objects`, `make_subplots`, `add_trace`, `update_layout` (`Part 5, 6, 7 & 8`)*

Build a dual-panel dashboard combining a ranked bar chart and a statistical histogram using low-level Graph Objects.

### Instructions:
1. Prepare the top 10 countries:
   ```python
   top_10 = df.sort_values(by='Rank').head(10)
   ```
2. Initialize a 1-row by 2-column subplot structure:
   ```python
   fig = make_subplots(
       rows=1, cols=2,
       subplot_titles=('Top 10 Happiest Countries', 'Global Happiness Score Distribution'),
       horizontal_spacing=0.15
   )
   ```
3. **Trace 1 (Left Panel):**
   - Add a horizontal bar chart (`go.Bar`):
     - `x=top_10['Happiness_Score']`
     - `y=top_10['Country']`
     - `orientation='h'`
     - Marker color: `'#2E86AB'`
     - Custom hover template: `%{y}: %{x:.2f}<extra></extra>`
   - Place in `row=1, col=1`.
4. **Trace 2 (Right Panel):**
   - Add a histogram (`go.Histogram`):
     - `x=df['Happiness_Score']`
     - `nbinsx=15`
     - Marker color: `'#A23B72'`
     - Custom hover template: `Score Range: %{x}<br>Count: %{y}<extra></extra>`
   - Place in `row=1, col=2`.
5. Update figure layout:
   - Invert the y-axis of the left panel so Rank 1 is at the top (`fig.update_yaxes(autorange='reversed', row=1, col=1)`).
   - Set overall title: `'Global Happiness Analysis Dashboard (2024)'`.
   - Set `template='plotly_white'`, `height=500`, `width=950`, and hide the legend (`showlegend=False`).
6. Display the figure (`fig.show()`).

---

## Task 4: Interactive Global Choropleth Map (~7 mins)
*Topics: Geographic visualization, `px.choropleth` or `go.Choropleth`, projection types (`Part 9`)*

Map geospatial patterns in happiness scores across all sovereign countries.

### Instructions:
1. Build a choropleth map using `px.choropleth`:
   - Data: `df`.
   - Locations: `'Country'` (set `locationmode='country names'`).
   - Color: `'Happiness_Score'`.
   - Hover Name: `'Country'`.
   - Hover Data: `['Rank', 'GDP_per_Capita', 'Healthy_Life_Expectancy']`.
   - Continuous color scale: `'Viridis'` or `'Plasma'`.
2. Configure layout and projections:
   - Projection type: `'natural earth'` or `'orthographic'`.
   - Set title: `'Global Happiness Index 2024 - Choropleth Map'`.
   - Update colorbar title to `'Happiness Score'`.
   - Apply `template='plotly_white'`.
3. Inspect the map: Hover over different nations, zoom into specific continents, and observe global geographic disparities.

---

## Submission Checklist

- [ ] All Plotly Express and Graph Objects cells render without runtime errors.
- [ ] 3D scatter chart is interactive with legible axis labels and customized hover details.
- [ ] Subplots dashboard using `make_subplots` correctly combines both chart types with inverted rank order.
- [ ] Choropleth map renders country boundaries and binds scores accurately.
- [ ] Save and submit `assignment_04_plotly_<your_name>.ipynb`.
